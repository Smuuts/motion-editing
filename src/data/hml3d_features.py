"""
Forward HumanML3D feature extraction: world-space joint positions (T, 22, 3)
-> the 263-dim HumanML3D motion representation.

This is the inverse of `utils.visualise.recover_from_ric` and a faithful, pure-NumPy
port of the official HumanML3D pipeline (uniform-skeleton retarget -> put on floor ->
canonicalise facing to Z+ -> IK -> cont6d rotations + RIC positions + velocities +
foot contacts). Quaternions are [w, x, y, z]; the channel layout matches
`utils.skeleton` (see its module docstring).

Use it to bring *any* 22-joint motion (e.g. MotionFix's SMPL-H joint_positions) into the
representation the GroupDiT/MotionDiT model was trained on.

    feats = extract_hml3d_features(joints_22, tgt_offsets)   # (T-1, 263), RAW (un-normalised)

`tgt_offsets` is the rest-pose bone-length skeleton everything is retargeted onto; get it
once from a reference HumanML3D clip via `get_tgt_offsets(ref_joints)`.
"""

import numpy as np

# ── HumanML3D / SMPL 22-joint skeleton definition ───────────────────────────────
# Unit bone directions in the rest pose (paramUtil.t2m_raw_offsets).
T2M_RAW_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0],
    [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
], dtype=np.float64)

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]

# r_hip, l_hip, sdr_r, sdr_l — used to define the body's forward direction.
FACE_JOINT_INDX = [2, 1, 17, 16]
# Lower-leg joints used to compute the leg-length scale ratio for retargeting.
L_IDX1, L_IDX2 = 5, 8
# Foot joints for contact detection: right = [R_Ankle, R_Foot], left = [L_Ankle, L_Foot].
FID_R, FID_L = [8, 11], [7, 10]
FEET_THRE = 0.002


# ── quaternion ops (numpy, [w, x, y, z]) ────────────────────────────────────────
def _qmul(q, r):
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], axis=-1)


def _qrot(q, v):
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _qinv(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


def _qnormalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _qbetween(v0, v1):
    """Shortest-arc quaternion rotating v0 onto v1 (vectors need not be unit)."""
    v = np.cross(v0, v1)
    w = (np.sqrt((v0 ** 2).sum(-1, keepdims=True) * (v1 ** 2).sum(-1, keepdims=True))
         + (v0 * v1).sum(-1, keepdims=True))
    return _qnormalize(np.concatenate([w, v], axis=-1))


def _quaternion_to_cont6d(q):
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    m = np.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    # cont6d = first two columns of the rotation matrix
    return np.concatenate([m[..., 0], m[..., 1]], axis=-1)


# ── minimal numpy Skeleton (IK / FK / rest-pose offsets) ────────────────────────
class _Skeleton:
    def __init__(self, raw_offset, kinematic_tree):
        self._raw_offset = np.asarray(raw_offset, dtype=np.float64)
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [-1] + [0] * (len(self._raw_offset) - 1)
        for chain in kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def set_offset(self, offsets):
        self._offset = np.asarray(offsets, dtype=np.float64)

    def get_offsets_joints(self, joints):
        """Rest-pose bone offsets scaled to the bone lengths of one frame (J, 3)."""
        offsets = self._raw_offset.copy()
        for i in range(1, len(self._raw_offset)):
            offsets[i] = np.linalg.norm(joints[i] - joints[self._parents[i]]) * offsets[i]
        self._offset = offsets
        return offsets

    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(-1))[:, np.newaxis]

        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            from scipy.ndimage import gaussian_filter1d
            forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
        forward = forward / np.sqrt((forward ** 2).sum(-1))[:, np.newaxis]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0).astype(np.float64)
        root_quat = _qbetween(forward, target)

        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        quat_params[:, 0] = root_quat
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset[chain[j + 1]][np.newaxis].repeat(len(joints), axis=0)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v ** 2).sum(-1))[:, np.newaxis]
                rot_u_v = _qbetween(u, v)
                R_loc = _qmul(_qinv(R), rot_u_v)
                quat_params[:, chain[j + 1]] = R_loc
                R = _qmul(R, R_loc)
        return quat_params

    def forward_kinematics_np(self, quat_params, root_pos, do_root_R=True):
        offsets = np.broadcast_to(self._offset, quat_params.shape[:-1] + (3,)).copy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            R = quat_params[:, 0] if do_root_R else \
                np.array([[1.0, 0, 0, 0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = _qmul(R, quat_params[:, chain[i]])
                joints[:, chain[i]] = _qrot(R, offsets[:, chain[i]]) + joints[:, chain[i - 1]]
        return joints


def get_tgt_offsets(ref_joints):
    """Rest-pose skeleton offsets (22, 3) from one reference HumanML3D clip (T, 22, 3)."""
    return _Skeleton(T2M_RAW_OFFSETS, T2M_KINEMATIC_CHAIN).get_offsets_joints(ref_joints[0])


def _uniform_skeleton(positions, tgt_offsets):
    src = _Skeleton(T2M_RAW_OFFSETS, T2M_KINEMATIC_CHAIN)
    src_offset = src.get_offsets_joints(positions[0])
    src_leg = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg = np.abs(tgt_offsets[L_IDX1]).max() + np.abs(tgt_offsets[L_IDX2]).max()
    scale = tgt_leg / src_leg
    tgt_root_pos = positions[:, 0] * scale
    quat_params = src.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    src.set_offset(tgt_offsets)
    return src.forward_kinematics_np(quat_params, tgt_root_pos)


def extract_hml3d_features(positions, tgt_offsets, feet_thre=FEET_THRE,
                           return_transform=False):
    """
    (T, 22, 3) world-space joints -> (T-1, 263) RAW HumanML3D features.

    `positions` must be at 20 fps (HumanML3D's rate) for the velocity channels to be
    consistent with the trained model. Returns un-normalised features; apply
    (x - Mean) / Std before feeding the model.

    If `return_transform=True`, also returns the per-clip canonicalisation quaternion
    `root_quat_init` (4,) [w,x,y,z] — the Y-axis rotation that turned the body's frame-0
    facing onto Z+. Because canonicalisation is a global rigid transform, only an absolute
    orientation (e.g. SMPL global_orient) needs it undone later; local joint angles
    (body_pose) and pelvis-relative translation deltas are invariant to it.
    """
    positions = np.asarray(positions, dtype=np.float64).copy()

    positions = _uniform_skeleton(positions, tgt_offsets)

    # put on floor
    positions[:, :, 1] -= positions.min(axis=0).min(axis=0)[1]
    # XZ root of frame 0 to origin
    root_pos_init = positions[0]
    positions = positions - root_pos_init[0] * np.array([1, 0, 1])

    # canonicalise: frame-0 facing -> Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = (root_pos_init[r_hip] - root_pos_init[l_hip]) + \
             (root_pos_init[sdr_r] - root_pos_init[sdr_l])
    across = across / np.sqrt((across ** 2).sum(-1))
    forward_init = np.cross(np.array([0, 1, 0]), across)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(-1))
    root_quat_init = _qbetween(forward_init[np.newaxis], np.array([[0, 0, 1.0]]))  # (1,4)
    positions = _qrot(np.ones(positions.shape[:-1] + (4,)) * root_quat_init, positions)

    global_positions = positions.copy()

    # foot contacts
    def _foot(pos, fid):
        d = ((pos[1:, fid] - pos[:-1, fid]) ** 2).sum(-1)
        return (d < feet_thre).astype(np.float64)
    feet_l = _foot(positions, FID_L)
    feet_r = _foot(positions, FID_R)

    # cont6d rotations + root velocities (smooth forward)
    skel = _Skeleton(T2M_RAW_OFFSETS, T2M_KINEMATIC_CHAIN)
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)
    cont_6d_params = _quaternion_to_cont6d(quat_params)
    r_rot = quat_params[:, 0].copy()
    velocity = _qrot(r_rot[1:], positions[1:, 0] - positions[:-1, 0])
    r_velocity = _qmul(r_rot[1:], _qinv(r_rot[:-1]))

    # root-invariant joint positions (RIFKE): strip root XZ, face every frame to Z+
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = _qrot(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])           # y-axis angular velocity
    l_velocity = velocity[:, [0, 2]]                     # xz linear velocity
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)   # (T, 21*6)
    ric_data = positions[:, 1:].reshape(len(positions), -1)            # (T, 21*3)
    local_vel = _qrot(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                      global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)                  # (T-1, 22*3)

    data = np.concatenate([
        root_data,            # (T-1, 4)
        ric_data[:-1],        # (T-1, 63)
        rot_data[:-1],        # (T-1, 126)
        local_vel,            # (T-1, 66)
        feet_l, feet_r,       # (T-1, 4)
    ], axis=-1).astype(np.float32)
    if return_transform:
        return data, root_quat_init[0].astype(np.float32)
    return data
