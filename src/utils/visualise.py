"""
Visualise a HumanML3D motion as a skeleton animation and save as a GIF.

Usage:
    from utils.visualise import recover_from_ric, save_animation
    joints = recover_from_ric(raw_263dim_features)  # (T, 22, 3)
    save_animation(joints, save_path="output.gif", title="walk forward")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# HumanML3D / SMPL 22-joint ordering — each list is one chain drawn as one colour.
# Indices match the output of recover_from_ric (joint 0 = pelvis).
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],       # right leg:  pelvis→R_Hip→R_Knee→R_Ankle→R_Foot
    [0, 1, 4, 7, 10],       # left leg:   pelvis→L_Hip→L_Knee→L_Ankle→L_Foot
    [0, 3, 6, 9, 12, 15],   # spine:      pelvis→Spine1→Spine2→Spine3→Neck→Head
    [9, 14, 17, 19, 21],    # right arm:  Spine3→R_Collar→R_Shoulder→R_Elbow→R_Wrist
    [9, 13, 16, 18, 20],    # left arm:   Spine3→L_Collar→L_Shoulder→L_Elbow→L_Wrist
]

CHAIN_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


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


def save_animation(
    joints: np.ndarray,
    save_path: str,
    title: str = "",
    fps: int = 20,
    figsize: tuple = (6, 6),
):
    """
    Render a skeleton animation from joint positions and save as MP4 or GIF.

    joints    : (T, 22, 3) numpy array, world-space metres (SMPL axes: X=right, Y=up, Z=fwd)
    save_path : output path (.mp4 requires ffmpeg, .gif uses pillow)
    """
    T = joints.shape[0]

    fig = plt.figure(figsize=figsize, facecolor="black")
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False

    # SMPL Y=up → matplotlib Z=vertical; SMPL Z=fwd → matplotlib Y=depth
    margin = 0.3
    x_min, x_max = joints[:, :, 0].min() - margin, joints[:, :, 0].max() + margin
    y_min, y_max = joints[:, :, 2].min() - margin, joints[:, :, 2].max() + margin  # SMPL Z
    z_min, z_max = joints[:, :, 1].min() - margin, joints[:, :, 1].max() + margin  # SMPL Y

    lines = []
    for chain, color in zip(KINEMATIC_CHAIN, CHAIN_COLORS):
        line, = ax.plot([], [], [], "-o", color=color, markersize=3, linewidth=2)
        lines.append((chain, line))

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=20, azim=-70)
        ax.set_xlabel("X", color="gray", fontsize=7)
        ax.set_ylabel("Z (fwd)", color="gray", fontsize=7)
        ax.set_zlabel("Y (up)", color="gray", fontsize=7)
        ax.tick_params(colors="gray", labelsize=6)
        if title:
            ax.set_title(title, color="white", fontsize=9, pad=4)
        return [l for _, l in lines]

    def update(frame):
        for chain, line in lines:
            xs = joints[frame, chain, 0]
            ys = joints[frame, chain, 2]   # SMPL Z → mpl Y
            zs = joints[frame, chain, 1]   # SMPL Y → mpl Z
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        return [l for _, l in lines]

    ani = animation.FuncAnimation(
        fig, update, frames=T,
        init_func=init, blit=True, interval=1000 // fps,
    )

    writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
    ani.save(save_path, writer=writer, fps=fps, dpi=100,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"Saved animation: {save_path}")


def show_animation(
    joints: np.ndarray,
    title: str = "",
    fps: int = 20,
    figsize: tuple = (6, 6),
):
    """Display a skeleton animation interactively (blocking)."""
    T = joints.shape[0]

    fig = plt.figure(figsize=figsize, facecolor="black")
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False

    margin = 0.3
    x_min, x_max = joints[:, :, 0].min() - margin, joints[:, :, 0].max() + margin
    y_min, y_max = joints[:, :, 2].min() - margin, joints[:, :, 2].max() + margin
    z_min, z_max = joints[:, :, 1].min() - margin, joints[:, :, 1].max() + margin

    lines = []
    for chain, color in zip(KINEMATIC_CHAIN, CHAIN_COLORS):
        line, = ax.plot([], [], [], "-o", color=color, markersize=3, linewidth=2)
        lines.append((chain, line))

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=20, azim=-70)
        ax.set_xlabel("X", color="gray", fontsize=7)
        ax.set_ylabel("Z (fwd)", color="gray", fontsize=7)
        ax.set_zlabel("Y (up)", color="gray", fontsize=7)
        ax.tick_params(colors="gray", labelsize=6)
        if title:
            ax.set_title(title, color="white", fontsize=9, pad=4)
        return [l for _, l in lines]

    def update(frame):
        for chain, line in lines:
            xs = joints[frame, chain, 0]
            ys = joints[frame, chain, 2]
            zs = joints[frame, chain, 1]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        return [l for _, l in lines]

    ani = animation.FuncAnimation(  # noqa: F841 — must be kept alive for plt.show()
        fig, update, frames=T,
        init_func=init, blit=True, interval=1000 // fps,
    )
    plt.show()
