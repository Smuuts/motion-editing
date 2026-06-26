"""
MDM-style geometric losses and MPJPE for the 22-joint HumanML3D skeleton,
operating directly on the 263-dim feature representation.
"""

import torch

# Parent joint index for each of the 22 joints (-1 = root).
# Kept for the explicit LLM-derived fallback mask: given joint group names from an
# LLM (e.g., "right_arm"), map them to joint indices via BODY_PART_GROUPS in
# model/body_groups.py, then use PARENTS to optionally expand the mask to connecting joints.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


# HumanML3D 263-dim channel layout:
#   [0]      root angular velocity
#   [1:3]    root XZ linear velocity
#   [3]      root height Y
#   [4:67]   21 joint positions (root-relative, local frame), 21×3
#   [67:193] 21 joint rotations 6D, 21×6
#   [193:259] 22 joint velocity vectors, 22×3
#   [259:263] foot contact binary labels: L_Ankle, L_Foot, R_Ankle, R_Foot
#     (HumanML3D concatenates feet_l=(L_Ankle,L_Foot) then feet_r=(R_Ankle,R_Foot))
#
# Foot joint indices in the 21-joint position array (0-indexed, i.e. joint_id - 1),
# ordered to match the [259:263] contact channels above:
_HML3D_FOOT_JOINTS = [6, 9, 7, 10]  # L_Ankle=7, L_Foot=10, R_Ankle=8, R_Foot=11


def hml3d_geometric_losses(
    x_pred_norm: torch.Tensor,          # (B, T, 263) predicted clean features, normalised
    x_gt_norm:   torch.Tensor,          # (B, T, 263) ground-truth clean features, normalised
    mean:        torch.Tensor,          # (263,)
    std:         torch.Tensor,          # (263,)
    mask:        torch.Tensor | None = None,  # (B, T) True = real frame
) -> dict[str, torch.Tensor]:
    """
    MDM-style geometric losses (Eqs. 3-5) for HumanML3D (263-dim) features.

    L_pos  — MSE on the 21 local joint positions directly stored in the features.
    L_vel  — MSE on frame-to-frame position differences (penalises jitter).
    L_foot — penalises predicted foot velocity on frames where the GT says foot
             is in contact with the ground (mitigates foot sliding).

    Returns a dict {\"pos\": ..., \"vel\": ..., \"foot\": ...} of scalar tensors.
    x_pred_norm is clamped to ±5 before denormalisation to avoid gradient blow-up
    at high noise timesteps.
    """
    x_pred = x_pred_norm.clamp(-5, 5) * std + mean
    x_gt   = x_gt_norm              * std + mean

    # Joint positions: channels [4:67], 21 joints × 3, root-relative local frame
    pos_pred = x_pred[..., 4:67].reshape(*x_pred.shape[:-1], 21, 3)  # (B, T, 21, 3)
    pos_gt   = x_gt[...,   4:67].reshape(*x_gt.shape[:-1],   21, 3)

    # L_pos: per-joint position MSE
    sq_err = (pos_pred - pos_gt) ** 2
    if mask is not None:
        sq_err = sq_err * mask[:, :, None, None].float()
        l_pos  = sq_err.sum() / (mask.float().sum() * 21 * 3).clamp(min=1)
    else:
        l_pos = sq_err.mean()

    # L_vel: frame-to-frame position-difference MSE (MDM Eq. 5)
    vel_pred   = pos_pred[:, 1:] - pos_pred[:, :-1]   # (B, T-1, 21, 3)
    vel_gt     = pos_gt[:,   1:] - pos_gt[:,   :-1]
    vel_sq_err = (vel_pred - vel_gt) ** 2
    if mask is not None:
        vel_mask   = (mask[:, :-1] & mask[:, 1:]).float()  # both frames must be real
        vel_sq_err = vel_sq_err * vel_mask[:, :, None, None]
        l_vel      = vel_sq_err.sum() / (vel_mask.sum() * 21 * 3).clamp(min=1)
    else:
        l_vel = vel_sq_err.mean()

    # L_foot: penalise foot velocity when GT contact label is active (MDM Eq. 4)
    # Contact channels [259:263]: L_Ankle, L_Foot, R_Ankle, R_Foot (binary in GT),
    # paired element-wise with _HML3D_FOOT_JOINTS in the same order.
    foot_contact  = (x_gt[..., 259:263] >= 0.5).float()          # (B, T, 4)
    foot_pos_pred = pos_pred[:, :, _HML3D_FOOT_JOINTS, :]         # (B, T, 4, 3)
    foot_vel_pred = foot_pos_pred[:, 1:] - foot_pos_pred[:, :-1]  # (B, T-1, 4, 3)
    f_i           = foot_contact[:, :-1, :, None]                  # (B, T-1, 4, 1)
    foot_sq_err   = (foot_vel_pred * f_i) ** 2
    if mask is not None:
        vel_mask    = (mask[:, :-1] & mask[:, 1:]).float()
        foot_sq_err = foot_sq_err * vel_mask[:, :, None, None]
        l_foot      = foot_sq_err.sum() / (vel_mask.sum() * 4 * 3).clamp(min=1)
    else:
        l_foot = foot_sq_err.mean()

    return {"pos": l_pos, "vel": l_vel, "foot": l_foot}


# SMPL 22-joint feet (world-space FK joints): L_Ankle, L_Foot, R_Ankle, R_Foot.
_SMPLH_FOOT_JOINTS = [7, 10, 8, 11]


def smplh_geometric_losses(
    x_pred_norm: torch.Tensor,          # (B, T, 135) predicted clean SMPL-H features, normalised
    x_gt_norm:   torch.Tensor,          # (B, T, 135) ground-truth clean features, normalised
    mean:        torch.Tensor,          # (135,)
    std:         torch.Tensor,          # (135,)
    J_rest:      torch.Tensor,          # (22, 3) neutral rest joints
    parents:     torch.Tensor,          # (22,)   kinematic-tree parents
    mask:        torch.Tensor | None = None,  # (B, T) True = real frame
    foot_thre:   float = 0.01,          # world foot speed (m/frame) below which GT foot = in contact
) -> dict[str, torch.Tensor]:
    """
    MDM-style geometric losses for the SMPL-H (135-d) rep, on world-space joints obtained by forward
    kinematics (`smplh_world_joints`). Mirrors `hml3d_geometric_losses`, but the foot-contact label
    is derived from GT foot velocity (the SMPL-H rep has no explicit contact channels).
    """
    from data.smplh_features import smplh_world_joints

    x_pred = x_pred_norm.clamp(-5, 5) * std + mean
    x_gt   = x_gt_norm               * std + mean
    Jp = smplh_world_joints(x_pred, J_rest, parents)   # (B, T, 22, 3)
    Jg = smplh_world_joints(x_gt,   J_rest, parents)

    # L_pos: per-joint position MSE
    sq_err = (Jp - Jg) ** 2
    if mask is not None:
        sq_err = sq_err * mask[:, :, None, None].float()
        l_pos  = sq_err.sum() / (mask.float().sum() * 22 * 3).clamp(min=1)
    else:
        l_pos = sq_err.mean()

    # L_vel: frame-to-frame position-difference MSE
    vel_pred, vel_gt = Jp[:, 1:] - Jp[:, :-1], Jg[:, 1:] - Jg[:, :-1]
    vel_sq_err = (vel_pred - vel_gt) ** 2
    if mask is not None:
        vel_mask   = (mask[:, :-1] & mask[:, 1:]).float()
        vel_sq_err = vel_sq_err * vel_mask[:, :, None, None]
        l_vel      = vel_sq_err.sum() / (vel_mask.sum() * 22 * 3).clamp(min=1)
    else:
        l_vel = vel_sq_err.mean()

    # L_foot: penalise predicted foot velocity where the GT foot is in contact
    foot_pred = Jp[:, :, _SMPLH_FOOT_JOINTS, :]
    foot_gt   = Jg[:, :, _SMPLH_FOOT_JOINTS, :]
    fvel_pred = foot_pred[:, 1:] - foot_pred[:, :-1]                    # (B, T-1, 4, 3)
    fvel_gt   = foot_gt[:, 1:]   - foot_gt[:, :-1]
    contact   = (fvel_gt.norm(dim=-1) < foot_thre).float()[..., None]  # (B, T-1, 4, 1)
    foot_sq_err = (fvel_pred * contact) ** 2
    if mask is not None:
        vel_mask    = (mask[:, :-1] & mask[:, 1:]).float()
        foot_sq_err = foot_sq_err * vel_mask[:, :, None, None]
        l_foot      = foot_sq_err.sum() / (vel_mask.sum() * 4 * 3).clamp(min=1)
    else:
        l_foot = foot_sq_err.mean()

    return {"pos": l_pos, "vel": l_vel, "foot": l_foot}


def _joint_positions_humanml3d(x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, 263) denormalized → (B, T, 22, 3) root-relative local-frame positions."""
    joints_21 = x[..., 4:67].reshape(*x.shape[:-1], 21, 3)
    root = torch.zeros(*x.shape[:-1], 1, 3, device=x.device, dtype=x.dtype)
    return torch.cat([root, joints_21], dim=-2)


def compute_mpjpe(
    x_pred_norm: torch.Tensor,
    x_gt_norm:   torch.Tensor,
    mean:        torch.Tensor,
    std:         torch.Tensor,
    mask:        torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean per-joint position error (metres) between predicted and GT clean features.

    Both inputs are clamped to ±5 before denormalisation to avoid blow-up at
    high noise timesteps. Uses root-relative local-frame positions from channels
    [4:67] of the 263D HumanML3D feature vector.

    LEDITS++ evaluation: restrict the joint set to those NOT in the edited
    body-part group (joints whose mask column was all-zero) to measure source
    preservation.
    """
    x_pred = x_pred_norm.clamp(-5, 5).float() * std + mean
    x_gt   = x_gt_norm.clamp(-5, 5).float()   * std + mean

    joints_pred = _joint_positions_humanml3d(x_pred)
    joints_gt   = _joint_positions_humanml3d(x_gt)

    dist = torch.norm(joints_pred - joints_gt, dim=-1)  # (B, T, 22)

    if mask is not None:
        dist = dist * mask[:, :, None].float()
        return dist.sum() / (mask.float().sum() * 22).clamp(min=1)

    return dist.mean()
