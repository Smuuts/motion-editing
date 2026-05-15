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


def recover_from_ric(data: np.ndarray, joints_num: int = 22) -> np.ndarray:
    """
    Convert (T, 263) raw HumanML3D features → world-space joint positions (T, 22, 3).
    Must be called on RAW (denormalised) features.
    """
    T = len(data)
    rot_vel = data[:, 0]    # angular velocity around Y
    r_vel   = data[:, 1:3]  # XZ linear velocity in root frame

    # integrate rotation — prepend 0 so angle[0]=0 (no rotation at first frame),
    # consistent with position integration which also starts at 0.
    r_rot_ang = np.cumsum(np.concatenate([[0.0], rot_vel[:-1]]))  # (T,)
    cos_a = np.cos(r_rot_ang)
    sin_a = np.sin(r_rot_ang)

    # rotate XZ velocity into world frame, then integrate → root XZ position
    world_vx = r_vel[:, 0] * cos_a - r_vel[:, 1] * sin_a
    world_vz = r_vel[:, 0] * sin_a + r_vel[:, 1] * cos_a
    root_x = np.cumsum(np.concatenate([[0], world_vx[:-1]]))
    root_z = np.cumsum(np.concatenate([[0], world_vz[:-1]]))
    root_y = data[:, 3]  # root height stored directly

    root_pos = np.stack([root_x, root_y, root_z], axis=-1)  # (T, 3)

    # (joints_num-1)*3 = 63 non-root joint positions at channels 4:67 (root-relative)
    j_pos = data[:, 4:4 + (joints_num - 1) * 3].reshape(T, joints_num - 1, 3)

    x = j_pos[:, :, 0].copy()
    z = j_pos[:, :, 2].copy()
    j_pos[:, :, 0] = x * cos_a[:, None] - z * sin_a[:, None]
    j_pos[:, :, 2] = x * sin_a[:, None] + z * cos_a[:, None]
    # Only add root XZ — joint Y values are already world-space height
    j_pos[:, :, 0] += root_x[:, None]
    j_pos[:, :, 2] += root_z[:, None]

    # prepend root so output indices match the kinematic chain
    return np.concatenate([root_pos[:, None, :], j_pos], axis=1)  # (T, 22, 3)


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
