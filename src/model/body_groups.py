"""
Body-part group definitions for the 22-joint HumanML3D/SMPL skeleton.

These constants are shared by:
  - GroupDiT (model/dit.py)        — tokenises the 263-dim feature vector into
                                      one token per body-part group.
  - analysis/ + utils/visualise/   — label the G-axis of attention heatmaps.
  - LEDITS++ Stage 2 masking       — reshapes (F*G) attention to (F, G) and maps
                                      LLM-named groups (e.g. "right_arm") to joints.

Keeping them in one place means the implicit attention mask (M1), the explicit
LLM-derived fallback mask, and the model all agree on the same group ordering
and channel partition.
"""

# Body-part groups: (name, joint_indices into the 21-joint body tensor).
# Joint indices are 0-based (SMPL joints 1–21, zero-indexed):
#   0=L_Hip 1=R_Hip 2=Spine1 3=L_Knee 4=R_Knee 5=Spine2 6=L_Ankle 7=R_Ankle
#   8=Spine3 9=L_Foot 10=R_Foot 11=Neck 12=L_Collar 13=R_Collar 14=Head
#   15=L_Shoulder 16=R_Shoulder 17=L_Elbow 18=R_Elbow 19=L_Wrist 20=R_Wrist
BODY_PART_GROUPS = [
    ("left_leg",  [0, 3, 6, 9]),      # L_Hip, L_Knee, L_Ankle, L_Foot
    ("right_leg", [1, 4, 7, 10]),     # R_Hip, R_Knee, R_Ankle, R_Foot
    ("spine",     [2, 5, 8]),          # Spine1, Spine2, Spine3
    ("left_arm",  [12, 15, 17, 19]),  # L_Collar, L_Shoulder, L_Elbow, L_Wrist
    ("right_arm", [13, 16, 18, 20]),  # R_Collar, R_Shoulder, R_Elbow, R_Wrist
    ("head",      [11, 14]),           # Neck, Head
]
N_GROUPS = 1 + len(BODY_PART_GROUPS)  # 7: root + 6 body-part groups

# Public: ordered group names matching the G-dimension of GroupDiT token sequences.
# Token g in frame f is at position f*G + g; GROUP_NAMES[g] is its name.
GROUP_NAMES: list[str] = ["root"] + [name for name, _ in BODY_PART_GROUPS]


# ── shared HumanML3D-263 channel arithmetic ─────────────────────────────────────
# The load-bearing offsets (4/67/193) and the foot-contact channel assignment live
# here once; both the body-part partition (build_group_channels) and the per-joint
# partition (build_joint_channels) are derived from them. For non-root body joint b
# (0-indexed into the 21-joint body array, = SMPL joint b+1):
#   position  → [4 + b*3   : +3]     rotation → [67 + b*6  : +6]
#   velocity  → [193 + (b+1)*3 : +3] (velocity array is 22-joint; index 0 = root).
def _hml3d_pos_ch(b):      return list(range(4   + b * 3,     4   + b * 3 + 3))
def _hml3d_rot_ch(b):      return list(range(67  + b * 6,     67  + b * 6 + 6))
def _hml3d_vel_ch(smpl_j): return list(range(193 + smpl_j*3, 193 + smpl_j*3 + 3))
def _hml3d_joint_ch(b):    return _hml3d_pos_ch(b) + _hml3d_rot_ch(b) + _hml3d_vel_ch(b + 1)

_HML3D_ROOT_CH = list(range(4)) + _hml3d_vel_ch(0)   # kinematics [0:4] + root vel → 7D
# Foot-contact labels [259:263] in (L_Ankle, L_Foot, R_Ankle, R_Foot) order; each is
# owned by its own leg body joint (index into the 21-joint body array).
_HML3D_FOOT_CONTACT_CH = {6: 259, 9: 260, 7: 261, 10: 262}


def build_group_channels() -> list[list[int]]:
    """
    Partition all 263 HumanML3D channels into 7 per-group index lists
    (root, left_leg, right_leg, spine, left_arm, right_arm, head).

    Root group gets root kinematics [0:4] + root velocity [193:196]. Left/right leg
    groups additionally get the two foot-contact channels for their ankle/foot joints,
    appended at the end of the group (HumanML3D concatenates feet_l=(L_Ankle,L_Foot)
    then feet_r=(R_Ankle,R_Foot)). All 263 channels appear exactly once across the 7 lists.
    """
    body_groups: list[list[int]] = []
    for _, joint_ids in BODY_PART_GROUPS:
        ch: list[int] = []
        for j in joint_ids:
            ch += _hml3d_joint_ch(j)
        body_groups.append(ch)

    # left_leg (body index 0) ← L_Ankle, L_Foot contacts; right_leg (1) ← R_Ankle, R_Foot.
    body_groups[0] += [_HML3D_FOOT_CONTACT_CH[6], _HML3D_FOOT_CONTACT_CH[9]]
    body_groups[1] += [_HML3D_FOOT_CONTACT_CH[7], _HML3D_FOOT_CONTACT_CH[10]]

    channels = [list(_HML3D_ROOT_CH)] + body_groups
    assert sorted(sum(channels, [])) == list(range(263)), \
        "Group channel indices must be a partition of [0, 263)"
    return channels


# Per-group channel index lists and their dims: [7, 50, 50, 36, 48, 48, 24] → 263
GROUP_CHANNELS: list[list[int]] = build_group_channels()
GROUP_DIMS:     list[int]       = [len(ch) for ch in GROUP_CHANNELS]


# ── SMPL-H 135-d layout ─────────────────────────────────────────────────────────
# Feature vector: [trans_delta(3) | body_pose_6d(126) | global_orient_6d(6)].
# Each of the 21 body joints is one contiguous 6D rotation block — far cleaner than
# HumanML3D's four scattered arrays (positions/rotations/velocities/contacts).
SMPLH_FEAT_DIM = 135


# ── shared SMPL-H-135 channel arithmetic ────────────────────────────────────────
# Body joint b (0-indexed, = SMPL joint b+1) owns one contiguous 6D rotation block;
# the root gets the pelvis translation delta [0:3] + global orientation [129:135].
def _smplh_rot6d(b):  return list(range(3 + b * 6, 3 + b * 6 + 6))
_SMPLH_ROOT_CH = [0, 1, 2] + list(range(129, 135))   # trans_delta + global_orient_6d


def build_group_channels_smpl() -> list[list[int]]:
    """
    Partition the 135 SMPL-H channels into the same 7 body-part groups.
    Dims: [9, 24, 24, 18, 24, 24, 12] → 135.
    """
    body_groups = [[c for b in joint_ids for c in _smplh_rot6d(b)]
                   for _, joint_ids in BODY_PART_GROUPS]
    channels = [list(_SMPLH_ROOT_CH)] + body_groups
    assert sorted(sum(channels, [])) == list(range(SMPLH_FEAT_DIM)), \
        "SMPL-H group channels must be a partition of [0, 135)"
    return channels


GROUP_CHANNELS_SMPL: list[list[int]] = build_group_channels_smpl()
GROUP_DIMS_SMPL:     list[int]       = [len(ch) for ch in GROUP_CHANNELS_SMPL]


# ── per-joint (fine) tokenisation ────────────────────────────────────────────────
# An alternative to the 6 coarse body-part groups: one token per SMPL joint (22:
# root/pelvis + 21 body joints), selected by group_mode="joints" (default "parts").
# Motivation: finer query resolution gives attention/supervision more to bind to —
# the "finer group axis" route. The
# token AXIS is the only thing that changes; the channel partition is a strict
# refinement of the body-part one (every part is a union of its joint tokens), so the
# whole editing/masking stack — which is already parametrised on `group_channels`
# and the token count — works with either.
#
# JOINT_NAMES[k] names token k. Index 0 is the root (pelvis kinematics); tokens 1..21
# are SMPL body joints 1..21 (= body-array index k−1), in the standard SMPL order.
N_JOINTS = 22
JOINT_NAMES: list[str] = [
    "root",                                   # 0  Pelvis
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",          # 1..6
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",       # 7..12
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",       # 13..17
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",                       # 18..21
]
assert len(JOINT_NAMES) == N_JOINTS


def build_joint_channels() -> list[list[int]]:
    """Partition the 263 HumanML3D channels into 22 per-joint index lists.

    Body joint b (0-indexed, = SMPL joint b+1) is token b+1; the root/pelvis is token 0.
    Same shared arithmetic as build_group_channels(), but one token per joint instead
    of aggregating into body-part groups; foot contacts attach to their own joint token.
    """
    tokens: list[list[int]] = [list(_HML3D_ROOT_CH)]        # token 0: root kin + root vel
    tokens += [_hml3d_joint_ch(b) for b in range(21)]        # tokens 1..21: body joints 0..20
    for b, contact in _HML3D_FOOT_CONTACT_CH.items():
        tokens[b + 1].append(contact)

    assert sorted(sum(tokens, [])) == list(range(263)), \
        "Per-joint channel indices must be a partition of [0, 263)"
    return tokens


def build_joint_channels_smpl() -> list[list[int]]:
    """Partition the 135 SMPL-H channels into 22 per-joint index lists (token 0 = root,
    tokens 1..21 = body joints 0..20, each its own 6D rotation block)."""
    tokens: list[list[int]] = [list(_SMPLH_ROOT_CH)] + [_smplh_rot6d(b) for b in range(21)]
    assert sorted(sum(tokens, [])) == list(range(SMPLH_FEAT_DIM)), \
        "Per-joint SMPL-H channels must be a partition of [0, 135)"
    return tokens


JOINT_CHANNELS:      list[list[int]] = build_joint_channels()
JOINT_DIMS:          list[int]       = [len(ch) for ch in JOINT_CHANNELS]
JOINT_CHANNELS_SMPL: list[list[int]] = build_joint_channels_smpl()
JOINT_DIMS_SMPL:     list[int]       = [len(ch) for ch in JOINT_CHANNELS_SMPL]


# Which body-part group each per-joint token belongs to — lets group_mode="joints"
# still accept coarse body-part names (e.g. "left_arm") in --target_groups / LLM masks
# by expanding them to their constituent joint tokens. root → token 0.
_PART_TO_JOINT_TOKENS: dict[str, list[int]] = {"root": [0]}
for _name, _joint_ids in BODY_PART_GROUPS:
    _PART_TO_JOINT_TOKENS[_name] = [b + 1 for b in _joint_ids]


# feature_mode strings that select body-part-grouped tokenisation (GroupDiT).
# 'group' is a legacy alias for 'humanml3d' kept for old checkpoint configs.
GROUPED_FEATURE_MODES = ("humanml3d", "smplh", "group")


def is_grouped_mode(feature_mode: str) -> bool:
    """True if feature_mode selects GroupDiT (vs. legacy flat MotionDiT)."""
    return feature_mode in GROUPED_FEATURE_MODES


GROUP_MODES = ("parts", "joints")


def group_layout(feature_mode: str, group_mode: str = "parts",
                 ) -> tuple[list[list[int]], list[int], int]:
    """(token_channels, token_dims, total_feature_dim) for a grouped feature_mode.

    feature_mode picks the representation ('humanml3d'/legacy 'group' → 263-d;
    'smplh' → 135-d). group_mode picks the token AXIS:
      'parts'  (default) → 7 body-part group tokens (root + 6 groups).
      'joints'           → 22 per-joint tokens (root + 21 body joints).
    Both partition the same channels; only the token grouping differs.
    """
    if group_mode not in GROUP_MODES:
        raise ValueError(f"Unknown group_mode {group_mode!r}; choices: {GROUP_MODES}")
    is_hml3d = feature_mode in ("humanml3d", "group")
    is_smplh = feature_mode == "smplh"
    if not (is_hml3d or is_smplh):
        raise ValueError(f"Unknown grouped feature_mode: {feature_mode!r}")
    if group_mode == "joints":
        if is_hml3d:
            return JOINT_CHANNELS, JOINT_DIMS, 263
        return JOINT_CHANNELS_SMPL, JOINT_DIMS_SMPL, SMPLH_FEAT_DIM
    if is_hml3d:
        return GROUP_CHANNELS, GROUP_DIMS, 263
    return GROUP_CHANNELS_SMPL, GROUP_DIMS_SMPL, SMPLH_FEAT_DIM


def group_names(group_mode: str = "parts") -> list[str]:
    """Ordered token-axis names for a group_mode: GROUP_NAMES ('parts', 7) or
    JOINT_NAMES ('joints', 22). Token k in frame f sits at position f*G + k."""
    if group_mode not in GROUP_MODES:
        raise ValueError(f"Unknown group_mode {group_mode!r}; choices: {GROUP_MODES}")
    return JOINT_NAMES if group_mode == "joints" else GROUP_NAMES


def named_token_indices(names: list[str], group_mode: str = "parts") -> list[int]:
    """Resolve axis names → token indices for the given group_mode (deduplicated,
    sorted). 'parts' mode: names must be body-part group names. 'joints' mode: names
    may be joint names OR body-part group names (the latter expand to their joint
    tokens), so an instruction can still target "left_arm" with per-joint tokens.
    Raises ValueError on an unknown name (listing the valid choices)."""
    axis = group_names(group_mode)
    idxs: list[int] = []
    for n in names:
        if n in axis:
            idxs.append(axis.index(n))
        elif group_mode == "joints" and n in _PART_TO_JOINT_TOKENS:
            idxs.extend(_PART_TO_JOINT_TOKENS[n])
        else:
            choices = axis if group_mode == "parts" else axis + list(_PART_TO_JOINT_TOKENS)
            raise ValueError(f"Unknown group/joint name {n!r}; choices: {choices}")
    return sorted(set(idxs))


def axis_labels(names: list[str], group_mode: str = "parts") -> list[str]:
    """Resolve part/joint names to the token-axis names of `group_mode` (deduplicated,
    in axis order). The name-level counterpart of `named_token_indices` — in 'joints'
    mode a coarse part name expands to its joint tokens. Raises ValueError on an
    unknown name."""
    axis = group_names(group_mode)
    return [axis[i] for i in named_token_indices(names, group_mode)]


def parse_axis_spec(spec: str, group_mode: str = "parts") -> list[str]:
    """"left_arm,right_arm" / "left_arm right_arm" → token-axis names (see axis_labels).
    The one place a user-typed group specification is interpreted."""
    return axis_labels([n.strip() for n in spec.replace(",", " ").split() if n.strip()],
                       group_mode)


def resolve_group_context(config: dict) -> tuple[str, bool, str, list[str]]:
    """(feature_mode, is_group, group_mode, token_axis_names) read from a checkpoint
    config — the common preamble the editing/analysis entry points need to interpret a
    loaded model's token axis. Defaults ('humanml3d'/'parts') keep old configs working."""
    feature_mode = config.get("feature_mode", "humanml3d")
    group_mode   = config.get("group_mode", "parts")
    return feature_mode, is_grouped_mode(feature_mode), group_mode, group_names(group_mode)
