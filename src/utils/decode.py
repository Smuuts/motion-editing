"""
Raw (denormalised) motion features → world-space joint positions.

Two representations, one entry point (`recover_joints`):
  humanml3d (263-d) — RIC recovery: integrate the root velocity channels, then rotate
                      the stored local joint positions into the world frame.
  smplh     (135-d) — SMPL-H forward kinematics via a cached neutral body model.

`extract_hml3d_features` (data/hml3d_features.py) is the inverse of `recover_from_ric`.
"""

import numpy as np


# ── HumanML3D RIC recovery ───────────────────────────────────────────────────────

def _qinv(q: np.ndarray) -> np.ndarray:
    """Unit-quaternion inverse (conjugate). q: (..., 4) as [w, x, y, z]."""
    inv = q.copy()
    inv[..., 1:] *= -1
    return inv


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by unit quaternion(s) q.
    q: (..., 4), v: (..., 3) — shapes must broadcast on all but the last dim.
    """
    qvec = q[..., 1:]
    uv   = np.cross(qvec, v)
    uuv  = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _recover_root_rot_pos(data: np.ndarray):
    """Recover root Y-axis orientation and world position from velocity features.

    data : (..., T, 263)
    Returns
        r_rot_quat : (..., T, 4)  per-frame Y-axis rotation quaternion [w,x,y,z]
        r_pos      : (..., T, 3)  per-frame world-space root position
    """
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]        # shift by one frame
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)      # integrate velocity → angle

    r_rot_quat = np.zeros(data.shape[:-1] + (4,))
    r_rot_quat[..., 0] = np.cos(r_rot_ang)        # w
    r_rot_quat[..., 2] = np.sin(r_rot_ang)        # y  (Y-axis rotation)

    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]  # XZ velocity in root frame (shifted)
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)        # rotate to world frame
    r_pos = np.cumsum(r_pos, axis=-2)              # integrate → world XZ position
    r_pos[..., 1] = data[..., 3]                  # Y height stored directly (not velocity)

    return r_rot_quat, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """
    Convert raw HumanML3D features → world-space joint positions.
    Must be called on RAW (denormalised) features.

    data       : (..., T, 263)
    Returns    : (..., T, joints_num, 3)
    """
    r_rot_quat, r_pos = _recover_root_rot_pos(data)

    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (-1, 3))  # (..., T, J, 3)

    # rotate local joint positions from root-facing frame → world frame
    # r_rot_quat[..., None, :] → (..., T, 1, 4) broadcasts over joint dim
    positions = _qrot(_qinv(r_rot_quat)[..., None, :], positions)

    # add root XZ offset; joint Y is already world-space height
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # prepend root as joint 0
    return np.concatenate([r_pos[..., None, :], positions], axis=-2)


# ── SMPL-H forward kinematics ────────────────────────────────────────────────────

_SMPLH_BODY_MODEL = None


def smplh_body_model(model_path: str = "data/motionfix/data/body_models/smplh"):
    """Lazily build + cache a neutral SMPLHLayer for decoding smplh features to joints.

    Call once with the configured path before `recover_joints` on an smplh checkpoint;
    later calls reuse the cached model and ignore their argument.
    """
    global _SMPLH_BODY_MODEL
    if _SMPLH_BODY_MODEL is None:
        import smplx
        _SMPLH_BODY_MODEL = smplx.SMPLHLayer(
            model_path=model_path, gender="neutral", ext="npz").eval()
    return _SMPLH_BODY_MODEL


def recover_joints(raw_features: np.ndarray, feature_mode: str) -> np.ndarray:
    """Raw (denormalised) features → world-space joint positions (T, 22, 3).

    'humanml3d' (263) via RIC recovery; 'smplh' (135) via SMPL forward kinematics.
    """
    if feature_mode == "smplh":
        from data.smplh_features import smplh_decode_to_joints
        return smplh_decode_to_joints(raw_features, smplh_body_model())
    return recover_from_ric(raw_features, joints_num=22)
