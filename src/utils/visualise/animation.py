"""
Motion animations: render (T, 22, 3) world-space joints as an MP4/GIF.

    from utils.visualise import save_animation
    save_animation(joints, "output.gif", title="walk forward")

Every entry point draws either a **skeleton** (the kinematic chains, the default) or a **SMPL
body mesh**, chosen per panel by whether `vertices` were passed:

    from utils.visualise import vertices_from_features
    verts = vertices_from_features(raw_feat, feature_mode, smooth_sigma)   # None unless smplh
    save_animation(joints, "output.mp4", vertices=verts, skeleton_overlay=True)

`joints` stays required either way — the mesh panels follow the pelvis with it, and it is what
the overlay draws — so a caller that cannot build vertices (`humanml3d`, which stores joint
positions rather than rotations) simply passes None and gets the old behaviour. Mesh frames are
rendered offscreen by `mesh.py` and drawn into an ordinary 2D axis, so the figure furniture —
titles, the MPJPE readout, the edit-mask strip — is identical in both modes.

Feature → joints decoding lives in utils/decode.py (`recover_joints`).
"""

from collections import namedtuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401 (registers the 3d projection)

from utils.logger import get_logger
from utils.visualise.mesh import DEFAULT_MESH_SIZE, mesh_renderer

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

# View angle, shared by both renderers so the skeleton and the mesh cannot drift apart:
# matplotlib takes it through view_init, MeshRenderer through the same spherical convention.
CAMERA_ELEV, CAMERA_AZIM = 20.0, -70.0


# ── shared rendering helpers ────────────────────────────────────────────────────
# The three entry points below all recentre the viewport on the root joint every frame and
# differ only in how many panels they use and whether the result is saved or shown.

def _style_3d_axis(ax):
    """Black background + hidden panes — the shared look of every skeleton axis."""
    ax.set_facecolor("black")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.tick_params(colors="gray", labelsize=6)


def _init_3d_axis(ax, z_min, z_max, title=None):
    """One-time axis setup (view angle, labels, height range) — call from init_func."""
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
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


# Vertical framing shared by every panel of one figure, so two panels of a comparison are drawn
# at the same scale: z_min/z_max are the skeleton axes' height limits, centre_y/half_height the
# mesh camera's. They are computed from different arrays on purpose — a body's feet and scalp
# lie outside its ankle and head *joints*, and framing the mesh on joints would clip them.
_Framing = namedtuple("_Framing", "z_min z_max centre_y half_height")


def _framing(joint_arrays, vertex_arrays):
    z_min, z_max = _height_range(*joint_arrays)
    if vertex_arrays:
        lo = min(v[:, :, 1].min() for v in vertex_arrays)
        hi = max(v[:, :, 1].max() for v in vertex_arrays)
    else:
        lo, hi = z_min, z_max
    return _Framing(z_min, z_max, (lo + hi) / 2.0,
                    max((hi - lo) / 2.0 + 0.15, VIEWPORT_HALF_WIDTH))


# ── panels ───────────────────────────────────────────────────────────────────────
# A panel owns one axis and the data drawn on it, and exposes init()/update(frame)/artists.
# The two kinds are interchangeable, which is what keeps the entry points below mode-agnostic.

class _SkeletonPanel:
    """A 3D axis drawing the kinematic chains, viewport following the pelvis."""

    def __init__(self, fig, pos, joints, framing, title=None):
        self.joints, self.framing, self.title = joints, framing, title
        self.ax = fig.add_subplot(pos, projection="3d")
        _style_3d_axis(self.ax)
        self.lines = _make_skeleton_lines(self.ax)

    @property
    def artists(self):
        return [line for _, line in self.lines]

    def init(self):
        _init_3d_axis(self.ax, self.framing.z_min, self.framing.z_max, title=self.title)
        return self.artists

    def update(self, frame):
        if frame < len(self.joints):
            _update_skeleton(self.ax, self.lines, self.joints, frame,
                             self.framing.z_min, self.framing.z_max)
        return self.artists


class _MeshPanel:
    """A 2D axis showing offscreen-rendered SMPL bodies, optionally with the chains on top.

    The axis is 2D because its content is an image; `imshow` is given the renderer's pixel
    extent so the projected overlay can be plotted in raw pixel coordinates.
    """

    def __init__(self, fig, pos, joints, vertices, framing, renderer, title=None,
                 skeleton_overlay=False):
        self.joints, self.vertices = joints, vertices
        self.framing, self.renderer, self.title = framing, renderer, title

        size = renderer.size
        self.ax = fig.add_subplot(pos)
        self.ax.set_facecolor("black")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.image = self.ax.imshow(np.zeros((size, size, 3), dtype=np.uint8),
                                    extent=(0, size, size, 0), interpolation="bilinear")
        self.ax.set_xlim(0, size)
        self.ax.set_ylim(size, 0)

        self.lines = [] if not skeleton_overlay else [
            (chain, self.ax.plot([], [], "-o", color=color, markersize=2, linewidth=1.2,
                                 alpha=0.9)[0])
            for chain, color in zip(KINEMATIC_CHAIN, CHAIN_COLORS)
        ]

    @property
    def artists(self):
        return [self.image] + [line for _, line in self.lines]

    def init(self):
        if self.title:
            self.ax.set_title(self.title, color="white", fontsize=9, pad=4)
        return self.artists

    def update(self, frame):
        if frame < len(self.vertices):
            # The camera tracks the pelvis in X/Z (as the skeleton viewport does) but holds a
            # fixed height, so vertical motion shows as vertical motion instead of being
            # cancelled out by the camera.
            pose_frame = min(frame, len(self.joints) - 1)
            target = self.joints[pose_frame, 0].copy()
            target[1] = self.framing.centre_y
            rgb, pixels = self.renderer.render(
                self.vertices[frame], target, self.framing.half_height,
                overlay_points=self.joints[pose_frame] if self.lines else None)
            self.image.set_data(rgb)
            for chain, line in self.lines:
                line.set_data(pixels[chain, 0], pixels[chain, 1])
        return self.artists


def _make_panel(fig, pos, joints, vertices, framing, renderer, title=None,
                skeleton_overlay=False):
    """A mesh panel when this clip has vertices, the skeleton panel when it does not."""
    if vertices is None:
        return _SkeletonPanel(fig, pos, joints, framing, title=title)
    return _MeshPanel(fig, pos, joints, vertices, framing, renderer, title=title,
                      skeleton_overlay=skeleton_overlay)


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


def _animate_one(joints, title, fps, figsize, save_path, vertices=None,
                 skeleton_overlay=False, mesh_size=DEFAULT_MESH_SIZE):
    """Single-body driver shared by save_animation / show_animation."""
    framing = _framing([joints], [] if vertices is None else [vertices])

    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")

    with mesh_renderer(vertices is not None, size=mesh_size,
                       elev=CAMERA_ELEV, azim=CAMERA_AZIM) as renderer:
        panel = _make_panel(fig, 111, joints, vertices, framing, renderer, title=title,
                            skeleton_overlay=skeleton_overlay)
        _run(fig, panel.update, panel.init, joints.shape[0], fps, save_path)

    if save_path is not None:
        log.info(f"Saved animation: {save_path}")


def _ellipsis(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


# ── public entry points ──────────────────────────────────────────────────────────

def save_animation(joints: np.ndarray, save_path: str, title: str = "",
                   fps: int = 20, figsize: tuple = (6, 6),
                   vertices: np.ndarray = None, skeleton_overlay: bool = False,
                   mesh_size: int = DEFAULT_MESH_SIZE):
    """Render an animation and save it as MP4 (needs ffmpeg) or GIF (pillow).

    joints   : (T, 22, 3) world-space metres, SMPL axes (X=right, Y=up, Z=fwd).
    vertices : (T, V, 3) SMPL body vertices from `vertices_from_features` — renders the body
               mesh instead of the skeleton. None (default) keeps the skeleton.
    skeleton_overlay : draw the kinematic chains on top of the mesh (ignored without vertices).

    The viewport follows the pelvis, so a long trajectory doesn't shrink the figure.
    """
    _animate_one(joints, title, fps, figsize, save_path, vertices=vertices,
                 skeleton_overlay=skeleton_overlay, mesh_size=mesh_size)


def show_animation(joints: np.ndarray, title: str = "", fps: int = 20,
                   figsize: tuple = (6, 6), vertices: np.ndarray = None,
                   skeleton_overlay: bool = False, mesh_size: int = DEFAULT_MESH_SIZE):
    """Display an animation interactively (blocking). Viewport follows the root.

    See `save_animation` for `vertices` / `skeleton_overlay`. Note that a mesh panel is an
    image, so it cannot be rotated with the mouse the way the skeleton's 3D axis can.
    """
    _animate_one(joints, title, fps, figsize, save_path=None, vertices=vertices,
                 skeleton_overlay=skeleton_overlay, mesh_size=mesh_size)


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
    vertices_gen: np.ndarray = None,
    vertices_gt: np.ndarray = None,
    skeleton_overlay: bool = False,
    mesh_size: int = DEFAULT_MESH_SIZE,
):
    """
    Side-by-side animation: generated (left) vs ground truth / source (right).

    mpjpe_per_frame : (T_common,) root-relative MPJPE in metres; total_mpjpe its mean.
    gt_label        : defaults to "Ground Truth [clip_id]".
    edit_mask       : (T,) bool — when given, a timeline strip (green = edited) with a
                      moving cursor is drawn under the panels and the per-frame readout
                      shows EDIT / frozen. None → no strip.
    vertices_gen / vertices_gt : (T, V, 3) SMPL vertices from `vertices_from_features`; each
                      panel renders the body mesh when its array is given and the skeleton
                      when it is None, so the two sides can differ.
    skeleton_overlay : draw the kinematic chains on top of whichever panels are meshes.
    """
    T_common, T_gen, T_gt = len(mpjpe_per_frame), len(joints_gen), len(joints_gt)
    framing = _framing([joints_gen, joints_gt],
                       [v for v in (vertices_gen, vertices_gt) if v is not None])

    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.patch.set_facecolor("black")

    if title:
        fig.suptitle(_ellipsis(title, 73), color="white", fontsize=8, y=0.99)

    mpjpe_txt = fig.text(0.5, 0.01, f"Avg MPJPE: {total_mpjpe * 1000:.1f} mm",
                         ha="center", color="cyan", fontsize=9)

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

    with mesh_renderer(vertices_gen is not None or vertices_gt is not None, size=mesh_size,
                       elev=CAMERA_ELEV, azim=CAMERA_AZIM) as renderer:
        # Both panel kinds take their label through the constructor and set it in init().
        panel_gen = _make_panel(fig, 121, joints_gen, vertices_gen, framing, renderer,
                                title=_ellipsis(gen_label, 47),
                                skeleton_overlay=skeleton_overlay)
        panel_gt = _make_panel(fig, 122, joints_gt, vertices_gt, framing, renderer,
                               title=_ellipsis(gt_label if gt_label is not None
                                               else f"Ground Truth  [{clip_id}]", 47),
                               skeleton_overlay=skeleton_overlay)

        def init():
            return panel_gen.init() + panel_gt.init() + [mpjpe_txt]

        def update(frame):
            artists = panel_gen.update(frame) + panel_gt.update(frame)
            if frame < T_common:
                status = ""
                if edit_mask is not None and frame < len(edit_mask):
                    status = "  |  ● EDIT" if edit_mask[frame] > 0.5 else "  |  ○ frozen"
                mpjpe_txt.set_text(
                    f"Frame {frame:3d}: {mpjpe_per_frame[frame] * 1000:.1f} mm  |  "
                    f"Avg: {total_mpjpe * 1000:.1f} mm{status}")
            if cursor is not None:
                cursor.set_xdata([frame, frame])
                artists = artists + [cursor]
            return artists + [mpjpe_txt]

        _run(fig, update, init, max(T_gen, T_gt), fps, save_path)

    log.info(f"Saved comparison: {save_path}")
