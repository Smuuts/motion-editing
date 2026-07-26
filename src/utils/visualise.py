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


def mpjpe_from_joints(joints_a: np.ndarray, joints_b: np.ndarray):
    """Root-relative Mean Per-Joint Position Error between two joint sequences.

    joints_a, joints_b : (T, J, 3) world-space positions (any joint count/order,
                          as long as both arrays use the same one). Joint 0 is
                          treated as the root and subtracted out before comparing.
    Compares only the overlapping frame range if the two sequences differ in length.

    Returns (per_frame (T_common,), mean: float, T_common: int).
    """
    T_common = min(len(joints_a), len(joints_b))
    a = joints_a[:T_common] - joints_a[:T_common, 0:1]
    b = joints_b[:T_common] - joints_b[:T_common, 0:1]
    per_frame = np.sqrt(((a - b) ** 2).sum(axis=-1)).mean(axis=-1)  # (T_common,)
    return per_frame, float(per_frame.mean()), T_common


# ── Shared rendering helpers ────────────────────────────────────────────────────
#
# save_animation / show_animation / save_comparison_animation all draw the same
# KINEMATIC_CHAIN skeleton on one or two black 3D axes and recentre the viewport
# on the root joint every frame; these helpers hold that common setup/update logic
# so the three entry points below only differ in how many axes they use and
# whether the result is saved or shown.

def _style_3d_axis(ax):
    """Black background + hidden panes — the shared look of every skeleton axis."""
    ax.set_facecolor("black")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False


def _init_3d_axis(ax, z_min, z_max, title=None):
    """One-time axis setup (view angle, labels, height range) — call from init_func."""
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=20, azim=-70)
    ax.set_xlabel("X",       color="gray", fontsize=7)
    ax.set_ylabel("Z (fwd)", color="gray", fontsize=7)
    ax.set_zlabel("Y (up)",  color="gray", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)


def _make_skeleton_lines(ax):
    """Create one empty Line3D per KINEMATIC_CHAIN entry on `ax`. Returns [(chain, line), ...]."""
    return [
        (chain, ax.plot([], [], [], "-o", color=color, markersize=3, linewidth=2)[0])
        for chain, color in zip(KINEMATIC_CHAIN, CHAIN_COLORS)
    ]


def _update_skeleton(ax, lines, joints, frame, hw, z_min, z_max):
    """Recentre the viewport on the current pelvis position and redraw one frame's pose.

    joints : (T, J, 3) SMPL axes (X=right, Y=up, Z=fwd), mapped to matplotlib's
             (X, Y=depth, Z=vertical) by swapping SMPL's Y and Z.
    """
    cx, cy = joints[frame, 0, 0], joints[frame, 0, 2]
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - hw, cy + hw)
    ax.set_zlim(z_min, z_max)
    for chain, line in lines:
        line.set_data(joints[frame, chain, 0], joints[frame, chain, 2])
        line.set_3d_properties(joints[frame, chain, 1])


def _height_range(*joint_arrays):
    """Global Y (height) range across one or more (T, J, 3) joint arrays, padded by 0.2 m."""
    lo = min(j[:, :, 1].min() for j in joint_arrays)
    hi = max(j[:, :, 1].max() for j in joint_arrays)
    return lo - 0.2, hi + 0.2


def _animate_skeleton(joints, title, fps, figsize, save_path):
    """Shared driver for save_animation / show_animation.

    joints    : (T, 22, 3) numpy array, world-space metres.
    save_path : output path (.mp4/.gif) to save, or None to show interactively (blocking).
    """
    T = joints.shape[0]
    hw = 1.0   # viewport half-width in metres (body ~0.5 m wide, arms ~0.8 m)
    z_min, z_max = _height_range(joints)

    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_axis(ax)
    lines = _make_skeleton_lines(ax)

    def init():
        _init_3d_axis(ax, z_min, z_max, title=title)
        return [l for _, l in lines]

    def update(frame):
        _update_skeleton(ax, lines, joints, frame, hw, z_min, z_max)
        return [l for _, l in lines]

    # blit=False required because axis limits change every frame
    ani = animation.FuncAnimation(
        fig, update, frames=T,
        init_func=init, blit=False, interval=1000 // fps,
    )

    if save_path is None:
        plt.show()   # blocks until the window is closed; `ani` must stay alive until then
        return

    writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
    ani.save(save_path, writer=writer, fps=fps, dpi=100,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"Saved animation: {save_path}")


# ── Public entry points ──────────────────────────────────────────────────────────

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

    The viewport follows the root joint (joint 0 = pelvis) with a fixed half-width
    of 1.0 m, so a walking character stays centred rather than appearing as a tiny
    dot in a large axis when the trajectory is long.
    """
    _animate_skeleton(joints, title, fps, figsize, save_path)


def show_animation(
    joints: np.ndarray,
    title: str = "",
    fps: int = 20,
    figsize: tuple = (6, 6),
):
    """Display a skeleton animation interactively (blocking). Viewport follows root."""
    _animate_skeleton(joints, title, fps, figsize, save_path=None)


def save_comparison_animation(
    joints_gen: np.ndarray,
    joints_gt: np.ndarray,
    mpjpe_per_frame: np.ndarray,
    total_mpjpe: float,
    save_path: str,
    title: str = "",
    clip_id: str = "",
    fps: int = 20,
    figsize: tuple = (12, 6),
    gen_label: str = "Generated",
    gt_label: str = None,
    edit_mask: np.ndarray = None,
):
    """
    Side-by-side animation: generated (left) vs ground truth (right).

    joints_gen, joints_gt  : (T, 22, 3) world-space positions.
    mpjpe_per_frame        : (T_common,) root-relative MPJPE in metres.
    total_mpjpe            : mean MPJPE over T_common frames.
    gen_label, gt_label    : panel captions. gt_label defaults to "Ground Truth [clip_id]".
    edit_mask              : (T,) bool — per-frame edit mask. When given, a timeline strip
                             (green = edited) with a moving cursor is drawn under the panels
                             and the per-frame readout shows EDIT / frozen for the current
                             frame. None → no strip (backward-compatible).
    """
    T_common = len(mpjpe_per_frame)
    T_gen    = len(joints_gen)
    T_gt     = len(joints_gt)
    T        = max(T_gen, T_gt)
    hw       = 1.0
    z_min, z_max = _height_range(joints_gen, joints_gt)

    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")
    ax_gen = fig.add_subplot(121, projection="3d")
    ax_gt  = fig.add_subplot(122, projection="3d")
    for ax in (ax_gen, ax_gt):
        _style_3d_axis(ax)
        ax.tick_params(colors="gray", labelsize=6)

    if title:
        short = title[:72] + "…" if len(title) > 72 else title
        fig.suptitle(short, color="white", fontsize=8, y=0.99)

    def _wrap(s, n=46):
        return s if len(s) <= n else s[:n] + "…"
    ax_gen.set_title(_wrap(gen_label), color="white", fontsize=9, pad=4)
    ax_gt.set_title(_wrap(gt_label if gt_label is not None else f"Ground Truth  [{clip_id}]"),
                    color="white", fontsize=9, pad=4)

    mpjpe_txt = fig.text(
        0.5, 0.01,
        f"Avg MPJPE: {total_mpjpe * 1000:.1f} mm",
        ha="center", color="cyan", fontsize=9,
    )

    lines_gen = _make_skeleton_lines(ax_gen)
    lines_gt  = _make_skeleton_lines(ax_gt)

    # Optional per-frame edit-mask timeline (green = edited) with a moving cursor,
    # drawn as a thin strip under the two panels.
    cursor = None
    if edit_mask is not None:
        edit_mask = np.asarray(edit_mask).astype(float).reshape(-1)
        strip_ax = fig.add_axes((0.15, 0.05, 0.70, 0.03))
        strip_ax.imshow(edit_mask[None, :], aspect="auto", cmap="Greens",
                        vmin=0, vmax=1, extent=(0, len(edit_mask), 0, 1))
        strip_ax.set_yticks([]); strip_ax.set_xticks([])
        strip_ax.set_xlim(0, len(edit_mask))
        for sp in strip_ax.spines.values():
            sp.set_color("gray")
        strip_ax.text(-0.012, 0.5, "edit mask", transform=strip_ax.transAxes,
                      ha="right", va="center", color="white", fontsize=7)
        cursor = strip_ax.axvline(0, color="red", lw=1.5)

    def _init():
        _init_3d_axis(ax_gen, z_min, z_max)
        _init_3d_axis(ax_gt,  z_min, z_max)
        return [l for _, l in lines_gen] + [l for _, l in lines_gt] + [mpjpe_txt]

    def _update(frame):
        if frame < T_gen:
            _update_skeleton(ax_gen, lines_gen, joints_gen, frame, hw, z_min, z_max)
        if frame < T_gt:
            _update_skeleton(ax_gt, lines_gt, joints_gt, frame, hw, z_min, z_max)
        if frame < T_common:
            status = ""
            if edit_mask is not None and frame < len(edit_mask):
                status = "  |  ● EDIT" if edit_mask[frame] > 0.5 else "  |  ○ frozen"
            mpjpe_txt.set_text(
                f"Frame {frame:3d}: {mpjpe_per_frame[frame] * 1000:.1f} mm  |  "
                f"Avg: {total_mpjpe * 1000:.1f} mm{status}"
            )
        extra = []
        if cursor is not None:
            cursor.set_xdata([frame, frame])
            extra = [cursor]
        return [l for _, l in lines_gen] + [l for _, l in lines_gt] + [mpjpe_txt] + extra

    ani = animation.FuncAnimation(
        fig, _update, frames=T,
        init_func=_init, blit=False, interval=1000 // fps,
    )
    writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
    ani.save(save_path, writer=writer, fps=fps, dpi=100,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"Saved comparison: {save_path}")
