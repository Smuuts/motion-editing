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
                  edit_space=True):
    """The Stage-2 mask-collection knobs (see editing/masking.collect_statistics)."""
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
    parser.add_argument("--psi_readout", default="abs", choices=["abs", "energy"],
                        help="What the psi/M2 contrast measures: 'abs' (default) = "
                             "|x0_c - x0_ref|, the LEDITS++ magnitude; 'energy' = the "
                             "SIGNED change in per-group motion energy, which separates "
                             "'the edit adds motion here' from 'the edit stills the "
                             "source here'. Measured better inside M1 & M2 at matched "
                             "mask size (0.452 -> 0.583); default stays 'abs' until the "
                             "end-to-end MotionFix comparison.")
    parser.add_argument("--mask_timesteps", type=int, default=mask_timesteps,
                        help="Sweep this many evenly-spaced timesteps for mask "
                             "collection" + (" (default: all 1000; 40 is much faster "
                                             "and nearly identical)." if mask_timesteps is None
                                             else "."))
    if thresholds:
        parser.add_argument("--lambda_attn", type=float, default=70.0,
                            help="M1 percentile threshold (higher = sparser mask).")
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
