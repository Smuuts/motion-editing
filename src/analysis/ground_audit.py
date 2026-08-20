"""
The independent auditor for the caption->body-part parser: does the GENERATOR agree?

A single generation's per-group motion energy picks the named group at 0.990 and the
named side at 1.000, so it is the one signal in this project known to resolve laterality.
Generating from a sample of tier-1 captions and checking where the energy lands therefore
validates the parser at scale, and the disagreements are a concrete list of vocabulary
bugs rather than an opinion about them.
"""

import os

import numpy as np
import torch

from analysis.gen_diff import joint_divergence, temporal_activity
from data.body_part_labels import parse_caption
from model.body_groups import GROUP_NAMES
from model.sampler import DDPMSampler
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder
from utils.decode import recover_joints, smplh_body_model
from utils.logger import get_logger
from utils.model_io import load_model
from utils.probe import accuracy_block, group_profile

log = get_logger(__name__)

def audit_with_generator(args, annotations, device):
    """Do the parser's tier-1 labels agree with where the generator puts motion energy?

    Reuses the measured-good generation-space read-out verbatim (analysis/gen_diff):
    generate the caption, decode to joints, take the per-group motion energy, and ask two
    questions a constant bias cannot win.
    """
    # Tier-1 captions only: the audit is about laterality, which is what tier 1 carries.
    pool = []
    for _, caption in annotations:
        lat = [m for m in parse_caption(caption) if m.lat]
        if len(lat) == 1:                      # one unambiguous target to score against
            pool.append((caption, lat[0].groups[0]))
    rng = np.random.default_rng(args.audit_seed)
    idx = rng.permutation(len(pool))[:args.audit_n]
    sample = [pool[i] for i in idx]
    log.section(f"3. generator audit: {len(sample)} tier-1 captions "
                f"(pool {len(pool)})")

    model, config = load_model(args.audit_checkpoint, device=device,
                               use_ema=not args.no_ema)
    feature_mode = config.get("feature_mode", "humanml3d")
    if feature_mode == "smplh":
        smplh_body_model(args.smplh_model_path)
    schedule = NoiseSchedule.from_config(config, device=device)
    sampler = DDPMSampler(model, schedule, device)
    text_encoder = build_text_encoder(config, device=device)
    mean = np.load(os.path.join(args.data_root, "Mean.npy"))
    std = np.load(os.path.join(args.data_root, "Std.npy"))
    mirror = {"left_arm": "right_arm", "right_arm": "left_arm",
              "left_leg": "right_leg", "right_leg": "left_leg"}

    top1_wins, lat_wins, cat_wins, rows = [], [], [], []
    for start in range(0, len(sample), args.audit_batch):
        chunk = sample[start:start + args.audit_batch]
        with torch.no_grad():
            ctx = text_encoder.encode([c for c, _ in chunk]).to(device)
        gen = sampler.sample_paired(
            ctx, length=args.audit_length, guidance_scale=args.audit_guidance,
            generator=torch.Generator(device=device).manual_seed(args.audit_seed + start),
            show_progress=False).float().cpu().numpy()

        for (caption, target), g in zip(chunk, gen):
            joints = recover_joints(g * std + mean, feature_mode)
            profile = group_profile(temporal_activity(joint_divergence, joints))
            pred = GROUP_NAMES[int(np.argmax(profile))]
            ti, mi = GROUP_NAMES.index(target), GROUP_NAMES.index(mirror[target])
            own = [ti, mi]
            other = ([GROUP_NAMES.index("left_leg"), GROUP_NAMES.index("right_leg")]
                     if target.endswith("_arm") else
                     [GROUP_NAMES.index("left_arm"), GROUP_NAMES.index("right_arm")])
            top1_wins.append(pred == target)
            lat_wins.append(bool(profile[ti] > profile[mi]))
            cat_wins.append(bool(profile[own].sum() > profile[other].sum()))
            rows.append({"caption": caption, "parser": target, "generator_top1": pred,
                         "profile": profile.tolist()})
        log.info(f"  generated {min(start + args.audit_batch, len(sample))}/{len(sample)}")

    blocks = {
        "laterality": accuracy_block(lat_wins, "generator agrees on the SIDE"),
        "category": accuracy_block(cat_wins, "generator agrees on the LIMB"),
        "top1": accuracy_block(top1_wins, "generator's top-1 group == parser's group",
                               chance=1.0 / len(GROUP_NAMES)),
    }
    for b in blocks.values():
        flag = ("PASS" if b["beats_chance"] else
                "BELOW chance" if b["below_chance"] else "at chance")
        log.info(f"  {b['label']:44s} {b['accuracy']:.3f}  "
              f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]  n={b['n']}  "
              f"chance {b['chance']:.3f}  → {flag}")

    disagree = [r for r in rows if r["parser"] != r["generator_top1"]]
    if disagree:
        # Read these carefully before blaming the parser. The auditor scores where the
        # generation puts the most MOTION, which is not the same question as which part
        # the caption names: in a compound caption ("walks down stairs while holding the
        # rail with their left hand") the locomotion dominates the energy and the named
        # arm loses, even though the parser's label is the correct one to supervise. A
        # disagreement is a parser bug only when the caption names one part and the
        # generator picks a different *named-able* one.
        log.info(f"\n  {len(disagree)} disagreements; first 15 (inspect — a compound caption "
              f"whose named limb is not its dominant motion is EXPECTED here, and is not "
              f"a parser bug):")
        for r in disagree[:15]:
            log.info(f"    parser={r['parser']:10s} gen={r['generator_top1']:10s} "
                  f"| {r['caption'][:70]}")
    return {"blocks": blocks, "rows": rows}
