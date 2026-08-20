"""
Skeleton animations: render (T, 22, 3) world-space joints as an MP4/GIF.

    from utils.visualise import save_animation
    save_animation(joints, "output.gif", title="walk forward")

Feature → joints decoding lives in utils/decode.py (`recover_joints`).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 (registers the 3d projection)
from utils.logger import get_logger

log = get_logger(__name__)


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

VIEWPORT_HALF_WIDTH = 1.0   # metres; body ~0.5 m wide, arms ~0.8 m


# ── shared rendering helpers ────────────────────────────────────────────────────
# The three entry points below all draw the same KINEMATIC_CHAIN skeleton on black 3D
# axes and recentre the viewport on the root joint every frame; they differ only in
# how many axes they use and whether the result is saved or shown.

def _style_3d_axis(ax):
    """Black background + hidden panes — the shared look of every skeleton axis."""
    ax.set_facecolor("black")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.tick_params(colors="gray", labelsize=6)


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
    """One empty Line3D per KINEMATIC_CHAIN entry on `ax`. Returns [(chain, line), ...]."""
    return [
        (chain, ax.plot([], [], [], "-o", color=color, markersize=3, linewidth=2)[0])
        for chain, color in zip(KINEMATIC_CHAIN, CHAIN_COLORS)
    ]


def _update_skeleton(ax, lines, joints, frame, z_min, z_max):
    """Recentre the viewport on the current pelvis position and redraw one frame's pose.

    joints : (T, J, 3) SMPL axes (X=right, Y=up, Z=fwd), mapped to matplotlib's
             (X, Y=depth, Z=vertical) by swapping SMPL's Y and Z.
    """
    hw = VIEWPORT_HALF_WIDTH
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


def _run(fig, update, init, frames, fps, save_path):
    """Drive a FuncAnimation and either save it or show it (blocking).

    blit=False is required because the axis limits change every frame.
    """
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                  blit=False, interval=1000 // fps)
    if save_path is None:
        plt.show()   # blocks until closed; `ani` must stay alive until then
        return
    writer = "ffmpeg" if save_path.endswith(".mp4") else "pillow"
    ani.save(save_path, writer=writer, fps=fps, dpi=100,
             savefig_kwargs={"facecolor": "black"})
    plt.close(fig)


def _animate_one(joints, title, fps, figsize, save_path):
    """Single-skeleton driver shared by save_animation / show_animation."""
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
        _update_skeleton(ax, lines, joints, frame, z_min, z_max)
        return [l for _, l in lines]

    _run(fig, update, init, joints.shape[0], fps, save_path)
    if save_path is not None:
        log.info(f"Saved animation: {save_path}")


def _ellipsis(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


# ── public entry points ──────────────────────────────────────────────────────────

def save_animation(joints: np.ndarray, save_path: str, title: str = "",
                   fps: int = 20, figsize: tuple = (6, 6)):
    """Render a skeleton animation and save it as MP4 (needs ffmpeg) or GIF (pillow).

    joints : (T, 22, 3) world-space metres, SMPL axes (X=right, Y=up, Z=fwd).
    The viewport follows the pelvis, so a long trajectory doesn't shrink the figure.
    """
    _animate_one(joints, title, fps, figsize, save_path)


def show_animation(joints: np.ndarray, title: str = "", fps: int = 20,
                   figsize: tuple = (6, 6)):
    """Display a skeleton animation interactively (blocking). Viewport follows the root."""
    _animate_one(joints, title, fps, figsize, save_path=None)


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
    Side-by-side animation: generated (left) vs ground truth / source (right).

    mpjpe_per_frame : (T_common,) root-relative MPJPE in metres; total_mpjpe its mean.
    gt_label        : defaults to "Ground Truth [clip_id]".
    edit_mask       : (T,) bool — when given, a timeline strip (green = edited) with a
                      moving cursor is drawn under the panels and the per-frame readout
                      shows EDIT / frozen. None → no strip.
    """
    T_common, T_gen, T_gt = len(mpjpe_per_frame), len(joints_gen), len(joints_gt)
    z_min, z_max = _height_range(joints_gen, joints_gt)

    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")
    ax_gen = fig.add_subplot(121, projection="3d")
    ax_gt  = fig.add_subplot(122, projection="3d")
    for ax in (ax_gen, ax_gt):
        _style_3d_axis(ax)

    if title:
        fig.suptitle(_ellipsis(title, 73), color="white", fontsize=8, y=0.99)
    ax_gen.set_title(_ellipsis(gen_label, 47), color="white", fontsize=9, pad=4)
    ax_gt.set_title(_ellipsis(gt_label if gt_label is not None
                              else f"Ground Truth  [{clip_id}]", 47),
                    color="white", fontsize=9, pad=4)

    mpjpe_txt = fig.text(0.5, 0.01, f"Avg MPJPE: {total_mpjpe * 1000:.1f} mm",
                         ha="center", color="cyan", fontsize=9)

    lines_gen = _make_skeleton_lines(ax_gen)
    lines_gt  = _make_skeleton_lines(ax_gt)

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

    def init():
        _init_3d_axis(ax_gen, z_min, z_max)
        _init_3d_axis(ax_gt,  z_min, z_max)
        return [l for _, l in lines_gen] + [l for _, l in lines_gt] + [mpjpe_txt]

    def update(frame):
        if frame < T_gen:
            _update_skeleton(ax_gen, lines_gen, joints_gen, frame, z_min, z_max)
        if frame < T_gt:
            _update_skeleton(ax_gt, lines_gt, joints_gt, frame, z_min, z_max)
        if frame < T_common:
            status = ""
            if edit_mask is not None and frame < len(edit_mask):
                status = "  |  ● EDIT" if edit_mask[frame] > 0.5 else "  |  ○ frozen"
            mpjpe_txt.set_text(
                f"Frame {frame:3d}: {mpjpe_per_frame[frame] * 1000:.1f} mm  |  "
                f"Avg: {total_mpjpe * 1000:.1f} mm{status}")
        extra = []
        if cursor is not None:
            cursor.set_xdata([frame, frame])
            extra = [cursor]
        return [l for _, l in lines_gen] + [l for _, l in lines_gt] + [mpjpe_txt] + extra

    _run(fig, update, init, max(T_gen, T_gt), fps, save_path)
    log.info(f"Saved comparison: {save_path}")
