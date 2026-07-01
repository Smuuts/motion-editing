"""
SMPL-H feature helpers for building a HumanML3D-aligned SMPL training set (Path A).

Produces the exact 135-d feature MotionFix's TMR consumes —
`[trans_delta(3) | body_pose_6d(126) | global_orient_6d(6)]` — from raw SMPL-H
`{rots:(T,66) axis-angle (22 body joints), trans:(T,3)}`, plus the left/right mirror
(for HumanML3D's `M`-prefixed clips). einops/roma-free so it runs in the `ma` env.

Layout matches `data/motionfix/src/data/features.py` (`_get_body_pose` / `_get_body_orient`
/ `_get_body_transl_delta_pelv`) and `tmr_evaluator`'s `collect_gen_samples`, validated to
~1e-6 against MotionFix's own loader (see smplh_features_selftest in this repo's tests/runbook).
"""

import numpy as np
import torch
import torch.nn.functional as F
from smplx.lbs import batch_rodrigues, batch_rigid_transform

# SMPL 22-joint order; left<->right swap permutation for sagittal mirroring.
#  0 pelvis 1 L_hip 2 R_hip 3 spine1 4 L_knee 5 R_knee 6 spine2 7 L_ankle 8 R_ankle
#  9 spine3 10 L_foot 11 R_foot 12 neck 13 L_collar 14 R_collar 15 head
# 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist 21 R_wrist
_MIRROR_PERM = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10,
                         12, 14, 13, 15, 17, 16, 19, 18, 21, 20])

# AMASS/SMPL world joints come out Z-up (head high +Z, feet low -Z), but the HumanML3D
# convention — and everything downstream that consumes joints (recover_from_ric,
# extract_hml3d_features, the animation renderer) — is Y-up. This proper rotation about X
# (+90°) maps (x, y, z) -> (x, z, -y) so the body stands upright with gravity along -Y.
# Determinant +1 (no reflection) so chirality is preserved. Applied as `joints @ AMASS_TO_HML3D`.
AMASS_TO_HML3D = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]], dtype=np.float32)


def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle (...,3) -> rotation matrix (...,3,3)."""
    return batch_rodrigues(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)


def aa_to_6d(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle (...,3) -> 6D (first two rows of the matrix), (...,6) — pytorch3d convention."""
    m = aa_to_rotmat(aa)
    return m[..., :2, :].reshape(*aa.shape[:-1], 6)


def smplh_to_features(rots, trans) -> np.ndarray:
    """
    Raw SMPL-H -> (T,135) [trans_delta | body_pose_6d | global_orient_6d], un-normalised.

    rots  : (T,66) axis-angle, 22 body joints (joint 0 = global orient, 1..21 = body pose).
    trans : (T,3)  pelvis translation.
    """
    rots = torch.as_tensor(np.asarray(rots), dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(trans), dtype=torch.float32)
    T = rots.shape[0]
    go = rots[:, :3]                                   # global orient
    bp = rots[:, 3:66].reshape(T, 21, 3)               # 21 body joints
    orient6d = aa_to_6d(go)                            # (T,6)
    pose6d = aa_to_6d(bp).reshape(T, 126)              # (T,126)
    # pelvis-frame translation delta: v_i = R_{i-1}^T (t_i - t_{i-1}), first frame zeroed.
    # matches MotionFix change_for (forward=True): einsum('...di,...d->...i', R, p) = R^T p.
    R = aa_to_rotmat(go)                               # (T,3,3)
    tv = trans - torch.roll(trans, 1, 0)
    tdelta = torch.einsum("tdi,td->ti", torch.roll(R, 1, 0), tv)
    tdelta[0] = 0
    return torch.cat([tdelta, pose6d, orient6d], dim=-1).numpy().astype(np.float32)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D rotation (..., 6) -> rotation matrix (..., 3, 3) via Gram-Schmidt (pytorch3d
    convention — inverse of `aa_to_6d`/`matrix_to_rotation_6d`)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def features_to_rotmats(feat: torch.Tensor):
    """135-d feature (..., 135) -> (rotmats (..., 22, 3, 3), trans_delta (..., 3)).

    rotmats[..., 0] = global orient; [..., 1:] = the 21 body joints.
    """
    lead = feat.shape[:-1]
    go = rotation_6d_to_matrix(feat[..., 129:135])                       # (...,3,3)
    bp = rotation_6d_to_matrix(feat[..., 3:129].reshape(*lead, 21, 6))   # (...,21,3,3)
    rotmats = torch.cat([go.unsqueeze(-3), bp], dim=-3)                   # (...,22,3,3)
    return rotmats, feat[..., 0:3]


def smplh_rest_joints_and_parents(body_model):
    """Neutral-shape rest joints (22, 3) and kinematic-tree parents (22,) from a SMPLHLayer."""
    J_rest = (body_model.J_regressor @ body_model.v_template)[:22].clone()   # betas=0
    parents = body_model.parents[:22].clone().long()
    return J_rest, parents


def smplh_world_joints(feat: torch.Tensor, J_rest: torch.Tensor, parents: torch.Tensor):
    """135-d features (B, T, 135) -> world-space joints (B, T, 22, 3).

    Cheap joints-only forward kinematics (`batch_rigid_transform`, no LBS/vertices). The
    pelvis-frame `trans_delta` is rotated to the world frame by the per-frame global orient and
    integrated over time so foot-contact velocities are measured in world space.
    """
    B, T, _ = feat.shape
    rotmats, trans_delta = features_to_rotmats(feat)            # (B,T,22,3,3), (B,T,3)
    go = rotmats[:, :, 0]                                       # (B,T,3,3) global orient
    # world velocity v_i = R_{i-1} @ trans_delta_i (inverse of change_for), then cumulative sum
    v = torch.zeros_like(trans_delta)
    v[:, 1:] = torch.einsum("btij,btj->bti", go[:, :-1], trans_delta[:, 1:])
    t_world = torch.cumsum(v, dim=1)                            # (B,T,3)
    J = J_rest.to(feat).unsqueeze(0).expand(B * T, -1, -1)      # (B*T,22,3)
    posed, _ = batch_rigid_transform(rotmats.reshape(B * T, 22, 3, 3), J, parents.to(feat.device))
    return posed.reshape(B, T, 22, 3) + t_world[:, :, None]


def smplh_decode_to_joints(feat_raw: np.ndarray, body_model, to_hml3d_frame: bool = True) -> np.ndarray:
    """RAW (T, 135) features -> (T, 22, 3) world joints (numpy). For decode / visualization.

    `to_hml3d_frame=True` rotates the native Z-up SMPL joints into HumanML3D's Y-up frame
    (via `AMASS_TO_HML3D`), which is what recover_from_ric / extract_hml3d_features / the
    renderer all expect. Set False only if you need the raw SMPL-frame joints.
    """
    J_rest, parents = smplh_rest_joints_and_parents(body_model)
    feat = torch.as_tensor(feat_raw, dtype=torch.float32)[None]   # (1,T,135)
    with torch.no_grad():
        joints = smplh_world_joints(feat, J_rest, parents)[0]
    joints = joints.cpu().numpy()
    if to_hml3d_frame:
        joints = joints @ AMASS_TO_HML3D
    return joints


def features_to_smpl(feat_raw: np.ndarray):
    """RAW (T,135) features -> (rots (T,66) axis-angle, trans (T,3) absolute).

    Inverse of `smplh_to_features` up to a constant translation offset (the pelvis-frame
    `trans_delta` integrates to a world trajectory anchored at trans[0]=0). MotionFix's TMR
    re-derives `trans_delta` from (global_orient, trans), so that offset is irrelevant. Used
    to decode the editor's output back to raw SMPL for fps resampling + the gen layout.
    """
    from scipy.spatial.transform import Rotation

    feat = torch.as_tensor(np.asarray(feat_raw), dtype=torch.float32)[None]   # (1,T,135)
    rotmats, trans_delta = features_to_rotmats(feat)                          # (1,T,22,3,3),(1,T,3)
    go = rotmats[:, :, 0]                                                     # (1,T,3,3)
    # invert change_for: world velocity v_i = R_{i-1} @ trans_delta_i, then integrate.
    v = torch.zeros_like(trans_delta)
    v[:, 1:] = torch.einsum("btij,btj->bti", go[:, :-1], trans_delta[:, 1:])
    trans = torch.cumsum(v, dim=1)[0].numpy().astype(np.float32)             # (T,3)
    T = rotmats.shape[1]
    aa = Rotation.from_matrix(rotmats[0].reshape(-1, 3, 3).numpy()).as_rotvec()
    rots = aa.reshape(T, 66).astype(np.float32)
    return rots, trans


def resample_motion(rots, trans, src_fps: float, dst_fps: float):
    """Resample raw SMPL-H (rots (T,66) axis-angle, trans (T,3)) from src_fps to dst_fps.

    Rotations via quaternion Slerp (per joint), translation via linear interpolation, over
    the same clip duration. The editor runs at 20 fps (HumanML3D) but MotionFix/TMR is native
    30 fps, so sources are 30->20 on input and edits 20->30 on output for comparable scoring.
    """
    from scipy.spatial.transform import Rotation, Slerp

    rots = np.asarray(rots, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    T = rots.shape[0]
    if src_fps == dst_fps or T < 2:
        return rots.copy(), trans.copy()
    T_dst = max(2, int(round(T * dst_fps / src_fps)))
    t_src = np.arange(T) / src_fps
    t_dst = np.linspace(0.0, t_src[-1], T_dst)

    r = rots.reshape(T, 22, 3)
    out = np.empty((T_dst, 22, 3), dtype=np.float32)
    for j in range(22):
        out[:, j] = Slerp(t_src, Rotation.from_rotvec(r[:, j]))(t_dst).as_rotvec()
    rots_dst = out.reshape(T_dst, 66)
    trans_dst = np.stack([np.interp(t_dst, t_src, trans[:, a]) for a in range(3)], axis=-1)
    return rots_dst.astype(np.float32), trans_dst.astype(np.float32)


def smpl_to_gen_layout(rots, trans) -> np.ndarray:
    """Raw SMPL-H (rots (T,66) aa, trans (T,3)) -> MotionFix gen layout (T,135):
    [trans(3) | global_orient_6d(6) | body_pose_6d(126)].

    This is the layout `tmr_evaluator.collect_gen_samples` expects (it re-derives trans_delta
    and reorders internally) — NOT the training `smplh_to_features` layout.
    """
    rots = torch.as_tensor(np.asarray(rots), dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(trans), dtype=torch.float32)
    T = rots.shape[0]
    go6d = aa_to_6d(rots[:, :3])                            # (T,6)
    bp6d = aa_to_6d(rots[:, 3:66].reshape(T, 21, 3)).reshape(T, 126)
    return torch.cat([trans, go6d, bp6d], dim=-1).numpy().astype(np.float32)


def mirror_smplh(rots, trans):
    """Sagittal (left<->right) mirror of raw SMPL-H, matching HumanML3D's `M`-clips.

    Standard SMPL pose flip: swap L/R joints, negate the y,z axis-angle components; negate trans-x.
    """
    rots = np.asarray(rots, dtype=np.float32).copy()
    trans = np.asarray(trans, dtype=np.float32).copy()
    T = rots.shape[0]
    r = rots.reshape(T, 22, 3)[:, _MIRROR_PERM]        # swap left/right joints
    r[..., 1] *= -1                                    # negate y component
    r[..., 2] *= -1                                    # negate z component
    trans[:, 0] *= -1                                  # mirror across the x=0 plane
    return r.reshape(T, 66), trans
