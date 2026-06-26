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
from smplx.lbs import batch_rodrigues

# SMPL 22-joint order; left<->right swap permutation for sagittal mirroring.
#  0 pelvis 1 L_hip 2 R_hip 3 spine1 4 L_knee 5 R_knee 6 spine2 7 L_ankle 8 R_ankle
#  9 spine3 10 L_foot 11 R_foot 12 neck 13 L_collar 14 R_collar 15 head
# 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist 21 R_wrist
_MIRROR_PERM = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10,
                         12, 14, 13, 15, 17, 16, 19, 18, 21, 20])


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
