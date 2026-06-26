"""
Body-part group definitions for the 22-joint HumanML3D/SMPL skeleton.

These constants are shared by:
  - GroupDiT (model/dit.py)        — tokenises the 263-dim feature vector into
                                      one token per body-part group.
  - analyse_attention.py           — labels the G-axis of attention heatmaps.
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


def build_group_channels() -> list[list[int]]:
    """
    Partition all 263 HumanML3D channels into 7 per-group index lists
    (root, left_leg, right_leg, spine, left_arm, right_arm, head).

    For each non-root joint j (0-indexed in the 21-joint body array):
      position  → channels [4 + j*3  : 4 + j*3  + 3]
      rotation  → channels [67 + j*6 : 67 + j*6 + 6]
      velocity  → channels [193 + (j+1)*3 : 193 + (j+1)*3 + 3]
        (velocity array is 22-joint; index 0 = root, so body joint j maps to vel index j+1)

    Root group gets: root kinematics [0:4] + root velocity [193:196].
    Left/right leg groups additionally get the two foot-contact channels
    for their ankle and foot joints ([259:263] in L_Ankle/L_Foot/R_Ankle/R_Foot
    order — HumanML3D concatenates feet_l=(L_Ankle,L_Foot) then feet_r=(R_Ankle,R_Foot)).

    All 263 channels appear exactly once across the 7 lists.
    """
    def pos_ch(j):      return list(range(4   + j * 3,     4   + j * 3 + 3))
    def rot_ch(j):      return list(range(67  + j * 6,     67  + j * 6 + 6))
    def vel_ch(smpl_j): return list(range(193 + smpl_j*3, 193 + smpl_j*3 + 3))

    root_ch = list(range(4)) + vel_ch(0)  # kinematics [0:4] + root vel [193:196] → 7D

    body_groups: list[list[int]] = []
    for _, joint_ids in BODY_PART_GROUPS:
        ch: list[int] = []
        for j in joint_ids:
            ch += pos_ch(j) + rot_ch(j) + vel_ch(j + 1)
        body_groups.append(ch)

    # Foot-contact labels [259:263]: L_Ankle, L_Foot, R_Ankle, R_Foot
    # L_Ankle (joint 6) and L_Foot (joint 9) belong to left_leg  (body index 0)
    # R_Ankle (joint 7) and R_Foot (joint 10) belong to right_leg (body index 1)
    body_groups[0] += [259, 260]   # left_leg  ← L_Ankle, L_Foot contact
    body_groups[1] += [261, 262]   # right_leg ← R_Ankle, R_Foot contact

    channels = [root_ch] + body_groups
    assert sorted(sum(channels, [])) == list(range(263)), \
        "Group channel indices must be a partition of [0, 263)"
    return channels


# Per-group channel index lists and their dims: [7, 50, 50, 36, 48, 48, 24] → 263
GROUP_CHANNELS: list[list[int]] = build_group_channels()
GROUP_DIMS:     list[int]       = [len(ch) for ch in GROUP_CHANNELS]
