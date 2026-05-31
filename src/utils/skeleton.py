"""
Differentiable forward kinematics for the 22-joint HumanML3D/SMPL skeleton.
Used to compute an auxiliary FK position loss during SMPL-mode training.
"""

import torch
import torch.nn.functional as F

# Parent joint index for each of the 22 joints (-1 = root).
# Used for FK and for the explicit LLM-derived fallback mask: given joint group names
# from an LLM (e.g., "right_arm"), map them to joint indices via _BODY_PART_GROUPS in
# dit.py, then use PARENTS to optionally expand the mask to include connecting joints.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Bone offsets (child position - parent position) derived from the mean of
# first frames across 200 HumanML3D training clips, in metres.
OFFSETS = torch.tensor([
    [ 0.0000,  0.0000,  0.0000],   #  0  pelvis (root)
    [ 0.0597, -0.0803, -0.0097],   #  1  L_Hip
    [-0.0620, -0.0878, -0.0066],   #  2  R_Hip
    [ 0.0015,  0.1234, -0.0276],   #  3  Spine1
    [ 0.0619, -0.3574,  0.0104],   #  4  L_Knee
    [-0.0606, -0.3543,  0.0038],   #  5  R_Knee
    [ 0.0056,  0.1342,  0.0243],   #  6  Spine2
    [-0.0155, -0.4107, -0.0587],   #  7  L_Ankle
    [ 0.0162, -0.4036, -0.0665],   #  8  R_Ankle
    [-0.0014,  0.0502,  0.0141],   #  9  Spine3
    [ 0.0505, -0.0534,  0.0946],   # 10  L_Foot
    [-0.0465, -0.0557,  0.0983],   # 11  R_Foot
    [-0.0091,  0.2114, -0.0089],   # 12  Neck
    [ 0.0710,  0.1131, -0.0067],   # 13  L_Collar
    [-0.0797,  0.1137, -0.0145],   # 14  R_Collar
    [ 0.0061,  0.0753,  0.0492],   # 15  Head
    [ 0.1255, -0.0057, -0.0093],   # 16  L_Shoulder
    [-0.1163, -0.0036, -0.0046],   # 17  R_Shoulder
    [ 0.0557, -0.2216, -0.0255],   # 18  L_Elbow
    [-0.0622, -0.2197, -0.0248],   # 19  R_Elbow
    [ 0.0247, -0.1697,  0.0793],   # 20  L_Wrist
    [-0.0432, -0.1675,  0.0713],   # 21  R_Wrist
], dtype=torch.float32)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion [w,x,y,z] → 3×3 rotation matrix. q: (..., 4) → (..., 3, 3)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def _cont6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """6D continuous rotation → 3×3 rotation matrix via Gram-Schmidt.
    x: (..., 6) → (..., 3, 3)  (columns are the orthonormal basis vectors)
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1.float(), dim=-1, eps=1e-6)
    b2 = F.normalize((a2.float() - (b1 * a2.float()).sum(dim=-1, keepdim=True) * b1), dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # (..., 3, 3) columns = [b1|b2|b3] = R


def _recover_root_torch(data: torch.Tensor):
    """
    Differentiable recovery of root orientation and world position from
    the 130-dim SMPL feature vector (denormalized).

    data : (B, T, 130)
    Returns
        r_rot_quat : (B, T, 4)  Y-axis rotation quaternion [w, x, y, z]
        r_pos      : (B, T, 3)  world-space root position
    """
    # ── root rotation ────────────────────────────────────────────────────────
    rot_vel = data[..., 0]                                     # (B, T)

    # Shift velocities by one frame so frame 0 starts at angle 0
    zeros   = torch.zeros(*data.shape[:-2], 1, device=data.device, dtype=data.dtype)
    ang_vel = torch.cat([zeros, rot_vel[..., :-1]], dim=-1)    # (B, T)
    r_rot_ang = torch.cumsum(ang_vel, dim=-1)                  # (B, T)

    r_rot_quat = torch.stack([
        torch.cos(r_rot_ang),
        torch.zeros_like(r_rot_ang),
        torch.sin(r_rot_ang),
        torch.zeros_like(r_rot_ang),
    ], dim=-1)                                                 # (B, T, 4)

    # ── root position ────────────────────────────────────────────────────────
    # XZ velocities are in the root-local frame; shift by one frame
    xz_vel = data[..., :-1, 1:3]                              # (B, T-1, 2)
    zeros2  = torch.zeros(*data.shape[:-2], 1, 2,
                          device=data.device, dtype=data.dtype)
    xz_vel  = torch.cat([zeros2, xz_vel], dim=-2)             # (B, T, 2)

    r_pos = torch.cat([
        xz_vel[..., 0:1],
        torch.zeros(*data.shape[:-2], data.shape[-2], 1,
                    device=data.device, dtype=data.dtype),
        xz_vel[..., 1:2],
    ], dim=-1)                                                 # (B, T, 3)

    # Rotate local velocities → world frame  (qrot(qinv(q), v))
    qinv  = torch.cat([r_rot_quat[..., :1], -r_rot_quat[..., 1:]], dim=-1)
    qvec  = qinv[..., 1:]
    uv    = torch.cross(qvec, r_pos, dim=-1)
    uuv   = torch.cross(qvec, uv, dim=-1)
    r_pos = r_pos + 2 * (qinv[..., :1] * uv + uuv)

    r_pos = torch.cumsum(r_pos, dim=-2)                        # integrate → world position

    # Y is stored directly (not as velocity)
    root_y = data[..., 3:4]
    r_pos  = torch.cat([r_pos[..., 0:1], root_y, r_pos[..., 2:3]], dim=-1)

    return r_rot_quat, r_pos


def _forward_kinematics(
    root_quat:  torch.Tensor,   # (B, T, 4)
    body_pose:  torch.Tensor,   # (B, T, 21, 6)
    root_pos:   torch.Tensor,   # (B, T, 3)
    offsets:    torch.Tensor,   # (22, 3)
) -> torch.Tensor:
    """
    Differentiable forward kinematics.
    Returns world-space joint positions: (B, T, 22, 3)
    """
    root_rotmat  = _quat_to_rotmat(root_quat)                  # (B, T, 3, 3)
    body_rotmats = _cont6d_to_rotmat(body_pose)                # (B, T, 21, 3, 3)
    all_rotmats  = torch.cat([root_rotmat.unsqueeze(2), body_rotmats], dim=2)  # (B, T, 22, 3, 3)

    positions   = [None] * 22
    global_rots = [None] * 22

    positions[0]   = root_pos
    global_rots[0] = all_rotmats[:, :, 0]

    for i in range(1, 22):
        p = PARENTS[i]
        # (B, T, 3, 3) @ (3,) → (B, T, 3)
        positions[i]   = positions[p] + torch.einsum('...ij,j->...i', global_rots[p], offsets[i])
        global_rots[i] = global_rots[p] @ all_rotmats[:, :, i]

    return torch.stack(positions, dim=2)                       # (B, T, 22, 3)


def fk_position_loss(
    x_pred_norm: torch.Tensor,   # (B, T, 130) predicted clean features, normalised
    x_gt_norm:   torch.Tensor,   # (B, T, 130) ground-truth clean features, normalised
    mean:        torch.Tensor,   # (130,)
    std:         torch.Tensor,   # (130,)
    mask:        torch.Tensor | None = None,  # (B, T) True = real frame
) -> torch.Tensor:
    """
    Compute mean-squared joint position error between the FK of the predicted
    and ground-truth clean SMPL feature sequences.

    Clamps x_pred_norm to ±5 before denormalisation to avoid gradient blow-up
    at high noise timesteps where predict_x0_from_eps amplifies errors.
    """
    x_pred = x_pred_norm.clamp(-5, 5) * std + mean
    x_gt   = x_gt_norm                * std + mean

    quat_pred, pos_pred = _recover_root_torch(x_pred)
    quat_gt,   pos_gt   = _recover_root_torch(x_gt)

    # channels [4:130] → (B, T, 21, 6)
    pose_pred = x_pred[..., 4:].reshape(*x_pred.shape[:-1], 21, 6)
    pose_gt   = x_gt[...,   4:].reshape(*x_gt.shape[:-1],   21, 6)

    offsets = OFFSETS.to(x_pred.device)
    joints_pred = _forward_kinematics(quat_pred, pose_pred, pos_pred, offsets)  # (B, T, 22, 3)
    joints_gt   = _forward_kinematics(quat_gt,   pose_gt,  pos_gt,   offsets)

    sq_err = (joints_pred - joints_gt) ** 2  # (B, T, 22, 3)

    if mask is not None:
        sq_err = sq_err * mask[:, :, None, None].float()
        return sq_err.sum() / (mask.float().sum() * 22 * 3).clamp(min=1)

    return sq_err.mean()


def _joint_positions_smpl(x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, 130) denormalized → (B, T, 22, 3) ROOT-RELATIVE world-space positions.
    Root is subtracted so joint 0 (pelvis) is always at origin — correct for MPJPE.
    For visualisation use recover_world_positions_smpl instead."""
    root_quat, root_pos = _recover_root_torch(x)
    body_pose = x[..., 4:].reshape(*x.shape[:-1], 21, 6)
    offsets = OFFSETS.to(x.device)
    joints = _forward_kinematics(root_quat, body_pose, root_pos, offsets)
    return joints - joints[:, :, 0:1, :]


def recover_world_positions_smpl(x: torch.Tensor) -> torch.Tensor:
    """x: (B, T, 130) denormalized → (B, T, 22, 3) WORLD-SPACE positions (root NOT subtracted).
    Use this for visualisation so global translation (walking forward etc.) is preserved.
    This matches recover_from_ric behaviour for HumanML3D mode."""
    root_quat, root_pos = _recover_root_torch(x)
    body_pose = x[..., 4:].reshape(*x.shape[:-1], 21, 6)
    offsets = OFFSETS.to(x.device)
    return _forward_kinematics(root_quat, body_pose, root_pos, offsets)


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
    feature_mode: str,
    mask:        torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean per-joint position error (metres) between predicted and GT clean features.

    Both inputs are clamped to ±5 before denormalisation to match the behaviour
    of fk_position_loss and avoid blow-up at high noise timesteps.

    SMPL mode:       root-relative world-space positions via FK (22 joints)
    HumanML3D mode:  root-relative local-frame positions from channels [4:67] (22 joints)

    The two frames are not identical, but both are root-relative and in metres,
    so the numbers are directly comparable across training modes.

    LEDITS++ evaluation: the proposal measures MPJPE on UNEDITED joints only
    (source preservation). Compute this by restricting the joint set to those
    NOT in the edited body-part group, i.e., joints whose mask column was all-zero.
    """
    x_pred = x_pred_norm.clamp(-5, 5).float() * std + mean
    x_gt   = x_gt_norm.clamp(-5, 5).float()   * std + mean

    if feature_mode in ("smpl", "group"):
        joints_pred = _joint_positions_smpl(x_pred)
        joints_gt   = _joint_positions_smpl(x_gt)
    else:
        joints_pred = _joint_positions_humanml3d(x_pred)
        joints_gt   = _joint_positions_humanml3d(x_gt)

    dist = torch.norm(joints_pred - joints_gt, dim=-1)  # (B, T, 22)

    if mask is not None:
        dist = dist * mask[:, :, None].float()
        return dist.sum() / (mask.float().sum() * 22).clamp(min=1)

    return dist.mean()
