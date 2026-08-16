"""
Argument definitions and parsing shared by the entry-point scripts in src/.

The `add_*_args` helpers define the flag groups that several scripts have in common,
so a default can't drift between two scripts that claim to do the same thing. The rest
parses the shared CLI shapes (a device string, a "one value or one per edit" list, a
body-part group spec) identically everywhere.
"""

import torch

from model.body_groups import group_names, parse_axis_spec


# ── shared flag groups ───────────────────────────────────────────────────────────

def add_model_args(parser, *, multi_checkpoint=False):
    """--checkpoint / --no_ema / --device: which weights to run and where."""
    if multi_checkpoint:
        parser.add_argument("--checkpoint", action="append", required=True,
                            help="Checkpoint dir (config.json + ema.pt). Repeat to A/B "
                                 "several checkpoints.")
    else:
        parser.add_argument("--checkpoint", required=True,
                            help="Checkpoint dir (config.json + ema.pt).")
    parser.add_argument("--no_ema", action="store_true",
                        help="Load model.pt instead of ema.pt.")
    parser.add_argument("--device", default=None,
                        help="'cuda'/'cpu' (default: cuda when available).")
    return parser


def add_data_args(parser, *, split=True, source=False, max_frames=196, smplh=False):
    """--data_root (+ optional --split / --source / --smplh_model_path) / --max_frames."""
    parser.add_argument("--data_root", required=True,
                        help="Data root (Mean/Std.npy, <split>.txt, new_joint_vecs/).")
    if source:
        parser.add_argument("--source", required=True,
                            help="Source clip: an index into --split, or a path to a "
                                 "raw (T, D) .npy file.")
    if split:
        parser.add_argument("--split", default="val")
    parser.add_argument("--max_frames", type=int, default=max_frames)
    if smplh:
        parser.add_argument("--smplh_model_path",
                            default="data/motionfix/data/body_models/smplh",
                            help="smplh checkpoints: SMPLHLayer dir (SMPLH_NEUTRAL.npz).")
    return parser


def add_mask_args(parser, *, mask_timesteps=40, thresholds=True, windows=True,
                  edit_space=True, alpha_floor=False):
    """The Stage-2 mask-collection knobs (see editing/masking.collect_statistics).

    `alpha_floor=True` additionally defines --guidance_alpha_floor, which is a Stage-3
    guidance knob rather than a mask one — opt-in for the same reason `thresholds`/`windows`
    are, so the probes that only build masks do not carry a flag they never read.
    """
    if edit_space:
        parser.add_argument("--edit_space", default="auto", choices=["auto", "eps", "x0"],
                            help="Space the editor does ψ/M2 and SEGA guidance in. "
                                 "'auto' (default) = the checkpoint's own predict_type, "
                                 "so an x0 checkpoint edits x0-natively "
                                 "(docs/AttentionGrounding_Options.md §5.3). Force a "
                                 "value to A/B the space against the checkpoint.")
    parser.add_argument("--m1_layers", default="auto",
                        help="Blocks the M1 read-out averages over. 'auto' (default) = "
                             "the checkpoint's own --attn_ground_layers when it was "
                             "trained with the grounding loss, else all blocks; 'all' "
                             "forces every block (the historical behaviour); '3,4,5' "
                             "picks explicitly. Averaging all 8 blocks on a checkpoint "
                             "that supervised 3 dilutes the grounded signal ~8/3x.")
    parser.add_argument("--m1_columns", default="auto",
                        choices=["auto", "span", "semantic", "content"],
                        help="Text columns the M1 read-out reads. 'content' = every "
                             "content token, the historical read -- note this INCLUDES "
                             "'a'/'person'/'the', since the default 'raw' readout never "
                             "applied masking._STOP_WORDS. 'semantic' strips those stop "
                             "words and keeps the verb. 'span' keeps only the body-part "
                             "words the grounding loss supervised (best measured, but it "
                             "uses the caption parser at inference). 'auto' (default) = "
                             "'semantic' on a grounded checkpoint, 'content' otherwise. "
                             "See docs/FINDINGS.md 'COLUMN dilution'.")
    parser.add_argument("--psi_readout", default="energy", choices=["abs", "energy"],
                        help="What the psi/M2 contrast measures: 'energy' (default since "
                             "2026-08-15) = the SIGNED change in per-group motion energy, "
                             "which separates 'the edit adds motion here' from 'the edit "
                             "stills the source here'; 'abs' = |x0_c - x0_ref|, the "
                             "LEDITS++ magnitude and the historical default (pass it to "
                             "reproduce any result recorded before that date). Size-"
                             "matched M2-alone comparison on 9 clips x 3 checkpoints, "
                             "identical 327 cells: alignment 0.24 -> 0.33, recall 0.50 -> "
                             "0.69, instruction-invariance 0.69 -> 0.13, source coupling "
                             "+0.42 -> -0.21. See docs/FINDINGS.md 'psi is a mixture'.")
    parser.add_argument("--mask_timesteps", type=int, default=mask_timesteps,
                        help="Sweep this many evenly-spaced timesteps for mask "
                             "collection" + (" (default: all 1000; 40 is much faster "
                                             "and nearly identical)." if mask_timesteps is None
                                             else "."))
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the Stage-1 inversion noise. The inversion draws "
                             "an independent x_t ladder per call, so an UNSEEDED probe "
                             "reports a sample, not a value -- two identical runs on clip "
                             "005675 differed by 0.48 in per-clip align_attn. Fixed at 42 "
                             "by default so probe runs are comparable; vary it to measure "
                             "the spread. Paired within-run contrasts were never affected "
                             "(one inversion is shared by all instructions).")
    parser.add_argument("--m1_select", default="percentile",
                        choices=["percentile", "rank"],
                        help="How M1's GROUP component is cut. 'percentile' (default) = "
                             "one global quantile at --lambda_attn over every (frame, "
                             "group) cell; bit-identical to results recorded before "
                             "2026-08-15, but it is a cell BUDGET, not a selector -- at "
                             "70 it must hand out 0.30*G = 2.1 group-rows whatever the "
                             "map says, so a one-group instruction always spills into the "
                             "runner-up. 'rank' = keep the groups holding at least "
                             "--m1_rank_ratio of the top group's mass (capped at "
                             "--m1_rank_max), then threshold psi INSIDE those rows only. "
                             "Rank adapts to how many groups the instruction actually "
                             "names instead of being told. docs/FINDINGS.md 'Two mask "
                             "defects with different causes'.")
    parser.add_argument("--m1_rank_ratio", type=float, default=0.5,
                        help="--m1_select rank: keep a group if its total M1 mass is at "
                             "least this fraction of the top group's. ->1 = top-1 only.")
    parser.add_argument("--m1_rank_max", type=int, default=3,
                        help="--m1_select rank: hard cap on selected groups. Stops a flat "
                             "(ungrounded) map from selecting the whole body.")
    if alpha_floor:
        parser.add_argument("--guidance_alpha_floor", type=float, default=None,
                            help="Apply edit guidance only where sqrt(alpha_cumprod_t) >= "
                                 "this. Default resolves per space (MotionEditor."
                                 "resolve_alpha_floor): 0.03 in eps space, where reaching x0 "
                                 "divides by a vanishing sqrt(alpha_cumprod_t) and guidance "
                                 "at the highest-noise steps diverges; 0 in x0 space, which "
                                 "has no such factor to gate and where an x0-trained model's "
                                 "text conditioning is strongest.")
    if thresholds:
        parser.add_argument("--lambda_attn", type=float, default=70.0,
                            help="M1 percentile threshold (higher = sparser mask). "
                                 "Unused when --m1_select rank.")
        parser.add_argument("--lambda_noise", type=float, default=70.0,
                            help="M2 percentile threshold (higher = sparser mask).")
    if windows:
        parser.add_argument("--m1_window", type=int, nargs=2, metavar=("LO", "HI"),
                            default=None,
                            help="Restrict M1's sweep to [LO, HI], resampled to "
                                 "--mask_timesteps steps INSIDE the window. M1 and M2 "
                                 "do not carry their signal at the same noise levels — "
                                 "see docs/FINDINGS.md.")
        parser.add_argument("--m2_window", type=int, nargs=2, metavar=("LO", "HI"),
                            default=None, help="Same, for M2/ψ.")
        parser.add_argument("--per_step_norm", action="store_true",
                            help="Weight every swept timestep equally instead of by its "
                                 "magnitude (an even grid is otherwise not an even "
                                 "average).")
    return parser


# ── shared parsing ───────────────────────────────────────────────────────────────

def resolve_device(name: str | None = None) -> torch.device:
    """`--device` string → torch.device, defaulting to cuda when available."""
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def per_edit_lookup(values, keys, name):
    """Resolve a CLI list that holds either one value (applied to all keys) or exactly
    one value per key into a `key -> value` lookup function. Returns None if `values`
    is None. Used for --scales and --target_groups, which both take this shape."""
    if values is None:
        return None
    if len(values) == 1:
        return lambda k: values[0]
    if len(values) == len(keys):
        vmap = dict(zip(keys, values))
        return lambda k: vmap[k]
    raise SystemExit(f"--{name} must be 1 value or {len(keys)} (got {len(values)}).")


def parse_group_names(spec: str, group_mode: str = "parts") -> list[str]:
    """`parse_axis_spec` at the CLI boundary: an unknown group name becomes a clean
    SystemExit rather than a traceback."""
    try:
        return parse_axis_spec(spec, group_mode)
    except ValueError as e:
        raise SystemExit(str(e))


def parse_group_mask(spec: str, is_group: bool, group_mode: str = "parts") -> torch.Tensor:
    """Same spec → (G,) bool tensor over the model's token axis, for --mask_mode groups
    (the user names the targets instead of relying on M1/an LLM to find them)."""
    if not is_group:
        raise SystemExit("--mask_mode groups requires a body-part-grouped model "
                         "(feature_mode humanml3d/smplh/group); this checkpoint has G=1.")
    if not spec.strip():
        raise SystemExit(f"--target_groups {spec!r} named no groups.")
    axis = group_names(group_mode)
    mask = torch.zeros(len(axis), dtype=torch.bool)
    mask[[axis.index(n) for n in parse_group_names(spec, group_mode)]] = True
    return mask
