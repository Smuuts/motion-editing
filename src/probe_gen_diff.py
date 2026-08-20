"""
Does DIFFING two shared-noise generations localise the instruction?

The premise: this backbone's generation path is measured-good while its editing
conditioner on inverted latents is measured-blind to the instruction. So take the mask
from the strong pathway — generate under the edit instruction and under a reference
prompt from ONE shared noise path, and read the group selector off the difference.
Nothing is trained, nothing is inverted, and no attention is read; the whole route is
gated on one question nobody has asked this checkpoint:

    does our own generator carry left/right at all?

If it does, this hands us an automatic instruction-driven mask on a frozen backbone. If
it does not, that is the cleaner finding — laterality is absent from the WEIGHTS, not
merely from the editing read-out — and attention supervision is confirmed mandatory.

WHAT IS MEASURED. The four contrasting instructions every probe here uses (two
laterality pairs x two limb categories), so the numbers land in the same table as
M1/M2's. Three read-outs of the same generations, the last two being controls that can
take the result away:

  paired    D = |g_instruction - g_reference|, SHARED noise     <- the statistic itself
  energy    the generation's own |delta| motion energy          <- does differencing add
                                                                  anything over just
                                                                  looking at the sample?
  unpaired  the same difference against a reference from a      <- does the shared noise
            DIFFERENT seed                                         do any work?

each scored in decoded joint space and in normalised feature space, and each reported as
lat_acc / cat_acc (forced choices, chance 0.5), top1 and align (chance 1/G), plus
r_lat/r_cat instruction-invariance. Both forced choices are between two options over an
instruction set symmetric in side and in limb, so a constant preference ("always left")
scores exactly chance and cannot fake a pass.

Controls: the batch carries the reference prompt TWICE and those two rows must come out
bit-identical (D == 0) — the proof the noise really is shared, without which every paired
number is meaningless. `D~act` reports how much D just tracks the reference generation's
motion energy.

One honest caveat on the default reference: a null context makes eps_cond == eps_uncond
for that row, so the reference is generated UNGUIDED while the instruction rows carry the
CFG scale. D therefore mixes "what the text does" with "what guidance does" — but
guidance points along the text direction by construction, so it amplifies the real
localisation rather than inventing one. `--reference caption` puts both rows on the same
guidance footing.

    python src/probe_gen_diff.py --checkpoint runs/exp_hml3d_x0/checkpoint_latest \\
        --data_root data/HumanML3D/HumanML3D --seeds 5
    # instructions composed onto a source clip's caption (the realistic editing setting)
    python src/probe_gen_diff.py --checkpoint ... --data_root ... --clip 012698 \\
        --reference caption --compose
"""

import os
import json
import argparse

import numpy as np
import torch

from analysis.gen_diff import (
    feature_divergence, joint_divergence, part_channels, readout_stats,
    temporal_activity, verdict,
)
from analysis.gen_diff_report import (family_label, print_reading, print_table,
                                      print_verdicts)
from analysis.instructions import DEFAULT_TARGETS, PHRASINGS
from data.clips import read_caption
from model.body_groups import GROUP_NAMES
from model.sampler import DDPMSampler
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from utils.cli import add_data_args, add_logging_args, add_model_args, configure_logging, resolve_device
from utils.decode import recover_joints, smplh_body_model
from utils.logger import get_logger
from utils.model_io import load_model
from utils.probe import flat_corr
from utils.visualise import plot_gen_diff, plot_gen_diff_summary

log = get_logger(__name__)

# Read-out families, in report order. The first is the statistic itself; the other
# two are the controls that decide whether it is doing anything the samples don't
# already say on their own.
READOUTS = ("paired", "energy", "unpaired")
SPACES = ("joint", "feat")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_data_args(p, split=False, smplh=True)
    p.add_argument("--clip", default=None,
                   help="Source clip id, for --reference caption / --compose. The probe "
                        "itself needs no clip: it only generates.")
    p.add_argument("--reference", default="null",
                   help="Reference prompt c_src: 'null' (the model's null context — the "
                        "default), 'caption' (the --clip's own annotation), or a literal "
                        "prompt string.")
    p.add_argument("--compose", action="store_true",
                   help="Send '<caption>, <instruction>' instead of the bare instruction "
                        "— the realistic editing setting. Needs --clip.")
    p.add_argument("--phrasing", default="instruction", choices=sorted(PHRASINGS),
                   help="'instruction' (default) sends the editor's imperatives; "
                        "'descriptive' sends the same four contrasts phrased like a "
                        "HumanML3D caption. The control for 'is a negative just the "
                        "generator being asked in a phrasing it never saw?'")
    p.add_argument("--length", type=int, default=120, help="Frames to generate.")
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seeds", type=int, default=5,
                   help="Independent shared-noise draws. Each is one full paired batch; "
                        "≥2 are needed for the 'unpaired' control.")
    p.add_argument("--seed0", type=int, default=42, help="First seed (then +1 each).")
    p.add_argument("--lambda_mask", type=float, default=70.0,
                   help="Percentile threshold for the alignment number (higher = "
                        "sparser), matching the editor's --lambda_noise default.")
    p.add_argument("--out_dir", default="eval_results/gen_diff")
    add_logging_args(p)
    return configure_logging(p.parse_args())


# ── prompts ──────────────────────────────────────────────────────────────────────

def build_prompts(args, caption):
    """(reference_label, reference_text_or_None, instruction prompts).

    A `None` reference text means the model's own null context — not a string the text
    encoder can produce, which is why it is carried separately.
    """
    if args.compose and not caption:
        raise SystemExit("--compose needs a --clip with a caption.")
    prompts = [f"{caption}, {e}" if args.compose else e
               for e in PHRASINGS[args.phrasing]]

    if args.reference == "null":
        return "∅ (null context)", None, prompts
    if args.reference == "caption":
        if not caption:
            raise SystemExit("--reference caption needs a --clip with a caption.")
        return caption, caption, prompts
    return args.reference, args.reference, prompts


def encode_prompts(text_encoder, model, ref_text, prompts, device):
    """(B, L, d) contexts for [reference, reference-again, *instructions].

    The reference is sent twice on purpose: the two rows share their prompt AND their
    noise, so their outputs must be identical — the plumbing check.
    """
    with torch.no_grad():
        ctx = text_encoder.encode(prompts)                        # (n, L, d)
        if ref_text is None:
            ref = model.null_text_emb.detach().to(ctx.dtype)      # (1, L, d)
        else:
            ref = text_encoder.encode([ref_text])
    return torch.cat([ref, ref.clone(), ctx], dim=0).to(device)


# ── one seed ─────────────────────────────────────────────────────────────────────

def generate(sampler, contexts, args, seed, device):
    """One shared-noise batch → (B, F, D) normalised motions as numpy."""
    generator = torch.Generator(device=device).manual_seed(seed)
    out = sampler.sample_paired(contexts, length=args.length,
                                guidance_scale=args.guidance_scale,
                                generator=generator, show_progress=False)
    return out.float().cpu().numpy()


def to_joints(gen, mean, std, feature_mode):
    """(B, F, D) normalised → [(F, 22, 3)] world joints, one per row."""
    return [recover_joints(g * std + mean, feature_mode) for g in gen]


def divergence_fns(feature_mode):
    """{space: (a, b) -> (F, G)} for the two spaces D is read in."""
    channels = part_channels(feature_mode)
    return {"feat": lambda a, b: feature_divergence(a, b, channels),
            "joint": joint_divergence}


def seed_readouts(gen, joints, prev, fns):
    """All read-out maps for one seed: {(readout, space): [ (F,G) per instruction ]}.

    `prev` is the previous seed's rows (or None on the first seed) — the 'unpaired'
    control differences against a reference generated from *different* noise, which is
    what isolates how much of the paired signal the shared noise is responsible for.
    """
    rows = {"feat": gen, "joint": joints}
    prev_rows = None if prev is None else {"feat": prev[0], "joint": prev[1]}
    maps = {}
    for space, fn in fns.items():
        r = rows[space]
        maps[("paired", space)] = [fn(r[i], r[0]) for i in range(2, len(r))]
        maps[("energy", space)] = [temporal_activity(fn, r[i]) for i in range(2, len(r))]
        if prev_rows is not None:
            maps[("unpaired", space)] = [fn(r[i], prev_rows[space][0])
                                         for i in range(2, len(r))]
    return maps



# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    log.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device=device, use_ema=not args.no_ema)
    feature_mode = config.get("feature_mode", "humanml3d")
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)
    schedule = NoiseSchedule.from_config(config, device=device)
    sampler = DDPMSampler(model, schedule, device)
    text_encoder = build_text_encoder(config, device=device)

    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))

    caption = read_caption(args.data_root, args.clip) if args.clip else ""
    ref_label, ref_text, prompts = build_prompts(args, caption)
    contexts = encode_prompts(text_encoder, model, ref_text, prompts, device)

    log.info(f"checkpoint   {args.checkpoint}")
    log.info(f"feature_mode {feature_mode}  arch={config.get('arch', 'dit')}  "
          f"predict_type={schedule.predict_type}")
    log.info(f"generation   {args.length} frames  guidance {args.guidance_scale}  "
          f"{args.seeds} seed(s) from {args.seed0}")
    log.info(f"reference    {ref_label!r}   phrasing {args.phrasing}")
    for e in prompts:
        log.info(f"  instruction {e!r}")

    fns = divergence_fns(feature_mode)
    families = {family_label(r, s): [] for r in READOUTS for s in SPACES}
    control_corr = {name: [] for name in families}
    plumbing, motions, prev = [], [], None
    last = {}

    for k in range(args.seeds):
        seed = args.seed0 + k
        gen = generate(sampler, contexts, args, seed, device)
        joints = to_joints(gen, mean, std, feature_mode)
        motions.append(gen)

        # Plumbing: rows 0 and 1 are the same prompt on the same noise path.
        plumbing.append(float(np.abs(gen[0] - gen[1]).max()))

        ref_energy = temporal_activity(fns["joint"], joints[0])
        last = {"maps": {}, "ref_energy": ref_energy}
        for (readout, space), maps in seed_readouts(gen, joints, prev, fns).items():
            name = family_label(readout, space)
            families[name].append(
                readout_stats(maps, instructions=prompts, percentile=args.lambda_mask))
            ref_act = (ref_energy if space == "joint"
                       else temporal_activity(fns["feat"], gen[0]))
            control_corr[name].append(float(np.mean([flat_corr(m, ref_act) for m in maps])))
            last["maps"][name] = maps

        prev = (gen, joints)
        log.info(f"  seed {seed}: generated {len(gen)} clips   "
              f"identical-rows |Δ|max = {plumbing[-1]:.2e}")

    families = {k: v for k, v in families.items() if v}          # drop empty 'unpaired'
    controls = {k: float(np.mean(v)) for k, v in control_corr.items() if v}
    verdicts = verdict(families)

    worst_plumbing = max(plumbing)
    paired_ok = ("OK" if worst_plumbing == 0.0 else
                 "NOT bit-identical — the pairing is broken and every paired number "
                 "below is suspect")
    log.info(f"\nplumbing check: two identical prompts on one shared noise path differ by "
          f"at most {worst_plumbing:.2e}  → {paired_ok}")
    if args.seeds < 2:
        log.info("note: --seeds 1 ⇒ the 'unpaired' control could not be computed.")

    print_table(families, controls)
    print_verdicts(verdicts)
    print_reading(verdicts)

    # Figures + JSON from the last seed's paired·joint maps (the headline read-out). The
    # tag carries the variant so a phrasing/reference sweep does not overwrite itself.
    tag = "_".join(filter(None, [
        os.path.basename(os.path.dirname(args.checkpoint.rstrip("/"))) or "ckpt",
        args.phrasing,
        "composed" if args.compose else "",
        "" if args.reference == "null" else "refcap"]))
    headline = family_label("paired", "joint")
    head = families[headline][-1]
    plot_gen_diff(prompts, DEFAULT_TARGETS, last["maps"][headline],
                  [np.array(head["profile"][e]) for e in prompts], last["ref_energy"],
                  GROUP_NAMES, ref_label,
                  os.path.join(args.out_dir, f"{tag}_gen_diff.png"),
                  title_extra=f"{tag} · {args.seeds} seeds · guidance "
                              f"{args.guidance_scale} · reference {ref_label!r}")
    plot_gen_diff_summary(prompts, {k: v[-1] for k, v in families.items()}, verdicts,
                          GROUP_NAMES, os.path.join(args.out_dir, f"{tag}_gen_diff_summary.png"))

    out = os.path.join(args.out_dir, f"{tag}_gen_diff.json")
    with open(out, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "feature_mode": feature_mode,
            "predict_type": schedule.predict_type, "arch": config.get("arch", "dit"),
            "length": args.length, "guidance_scale": args.guidance_scale,
            "seeds": [args.seed0 + k for k in range(args.seeds)],
            "reference": ref_label, "compose": args.compose, "clip": args.clip,
            "phrasing": args.phrasing,
            "prompts": prompts, "group_labels": GROUP_NAMES,
            "align_chance": 1.0 / len(GROUP_NAMES),
            "plumbing_max_abs_diff": worst_plumbing,
            "control_corr_with_reference_energy": controls,
            "per_seed": {k: v for k, v in families.items()},
            "verdict": verdicts,
        }, f, indent=2)
    log.info(f"\nWrote {out}")

    npz = os.path.join(args.out_dir, f"{tag}_gen_diff_motions.npz")
    np.savez_compressed(npz, motions=np.stack(motions).astype(np.float32),
                        prompts=np.array(["__reference__", "__reference_dup__"] + prompts))
    log.info(f"Wrote {npz}   (rows: reference, reference-dup, then the 4 instructions — "
          "render these to check the generator actually performs the instruction)")


if __name__ == "__main__":
    main()
