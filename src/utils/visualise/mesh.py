"""
SMPL body meshes for the animations in `animation.py`.

    from utils.visualise import vertices_from_features, save_animation
    verts = vertices_from_features(raw_feat, "smplh")     # (T, 6890, 3), or None
    save_animation(joints, "out.mp4", vertices=verts)     # None → skeleton, as before

Only the SMPL-H representation (135-d) can be meshed. It stores per-joint *rotations*, so the
body is the model's own posed surface — exact, and costing one extra LBS pass. HumanML3D's
263-d vector stores joint *positions*, from which a surface can only be recovered by fitting
(SMPLify-style optimisation, minutes per clip); `vertices_from_features` returns None there and
the caller falls back to the skeleton. Shape is the neutral body at betas=0, the same body
`utils.decode.smplh_body_model` uses for forward kinematics — verified to agree with
`smplh_decode_to_joints` to 4e-7 m, which is what lets the skeleton overlay sit exactly on the
surface rather than approximately.

Two things in the picture are the model's defaults rather than the clip's data, and should not
be read as such: the **body shape** (neutral, betas=0 — the 135-d vector carries no shape) and
the **hands** (flat, the SMPL-H zero hand pose — the representation covers the 22 body joints
only, so the fingers never move). `hands="relaxed"` draws SMPL-H's mean hand pose instead, with
naturally curled fingers; it is just as static, only less mannequin-like.

Rendering is pyrender offscreen (EGL — no display needed) into an RGB array, which
`animation.py` draws into an ordinary matplotlib axis with imshow. Everything the skeleton
figures carry — titles, the per-frame MPJPE readout, the edit-mask strip — is therefore
unchanged, and `render(..., overlay_points=)` projects world points into the very pixels it
just rendered, so the kinematic chains can be drawn back on top of the body.
"""

import os
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils.logger import get_logger

log = get_logger(__name__)

# PyOpenGL fixes its backend at import time, so this has to be set before pyrender is first
# imported (which happens lazily, inside MeshRenderer). EGL is the one that renders without an
# X display; `setdefault` leaves an explicit choice — e.g. PYOPENGL_PLATFORM=osmesa — alone.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

MESH_COLOUR = (0.62, 0.71, 0.85)    # matte blue-grey; reads on the black background
DEFAULT_MESH_SIZE = 480             # offscreen buffer, px (square: aspect ratio 1)
CAMERA_YFOV = np.pi / 4.0           # vertical field of view, radians


# ── features → body surface ──────────────────────────────────────────────────────

def smplh_faces() -> np.ndarray:
    """The neutral SMPL-H triangle list (13776, 3), from the cached body model."""
    from utils.decode import smplh_body_model
    return np.asarray(smplh_body_model().faces, dtype=np.int64)


def vertices_from_features(raw_features: np.ndarray, feature_mode: str,
                           smooth_sigma: float = 0.0, hands: str = "flat"):
    """RAW (T, D) features → (T, 6890, 3) world vertices in HumanML3D's Y-up frame.

    Returns None for any representation that carries joint positions rather than rotations
    (`humanml3d`, 263-d) — the caller's signal to render the skeleton instead.

    smooth_sigma : temporal gaussian smoothing in frames. Pass the same value the caller
                   applies to its joints, so an overlaid skeleton stays glued to the body.
                   (Smoothing vertices and smoothing joints are not the identical operation —
                   posed joints are not a linear function of posed vertices — but at the
                   sigma≈1.5 used for rendering the difference is far below a pixel.)
    hands        : "flat" (the layer's zero hand pose) or "relaxed" (SMPL-H's mean hand pose,
                   which the layer stores as left/right_hand_mean but never applies itself).
    """
    if feature_mode != "smplh":
        log.warning(f"feature_mode={feature_mode!r} stores joint positions, not rotations — "
                    "a body mesh would need fitting; rendering the skeleton instead.")
        return None

    import torch

    from data.smplh_features import AMASS_TO_HML3D, aa_to_rotmat, features_to_smpl
    from utils.decode import smplh_body_model

    body_model = smplh_body_model()
    rots, trans = features_to_smpl(np.asarray(raw_features))    # (T, 66) axis-angle, (T, 3)
    T = rots.shape[0]
    hand_kw = {}
    if hands == "relaxed":
        for side in ("left", "right"):
            mean = getattr(body_model, f"{side}_hand_mean").reshape(15, 3)
            hand_kw[f"{side}_hand_pose"] = aa_to_rotmat(mean)[None].expand(T, 15, 3, 3)
    elif hands != "flat":
        raise ValueError(f"hands must be 'flat' or 'relaxed', got {hands!r}")
    with torch.no_grad():
        posed = body_model(
            global_orient=aa_to_rotmat(torch.as_tensor(rots[:, :3]))[:, None],
            body_pose=aa_to_rotmat(torch.as_tensor(rots[:, 3:66]).reshape(T, 21, 3)),
            transl=torch.as_tensor(trans), **hand_kw)
    # Same Z-up → Y-up rotation smplh_decode_to_joints applies, so mesh and joints share a frame.
    verts = posed.vertices.cpu().numpy() @ AMASS_TO_HML3D
    if smooth_sigma > 0:
        verts = gaussian_filter1d(verts, sigma=smooth_sigma, axis=0)
    return verts.astype(np.float32)


# ── camera helpers ───────────────────────────────────────────────────────────────

def _look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera-to-world pose (4, 4) looking from `eye` at `target` (OpenGL: down local -Z)."""
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    back = eye - target
    back = back / np.linalg.norm(back)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(back @ up)) > 0.999:        # looking straight down/up: pick another up
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, back)
    right = right / np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(back, right), back, eye
    return pose


def _light_pose(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation-only pose (4, 4); a pyrender DirectionalLight shines along its own -Z."""
    pitch, yaw = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    pose = np.eye(4)
    pose[:3, :3] = ry @ rx
    return pose


# key (above-left), fill (right, dimmer), rim (behind the body, separating it from the black).
_LIGHT_RIG = [(_light_pose(-30, -35), 3.0), (_light_pose(-10, 50), 1.6), (_light_pose(25, 165), 2.0)]


class MeshRenderer:
    """Offscreen SMPL body renderer whose frames drop into a matplotlib imshow axis.

    One renderer serves every panel of a figure (the two sides of a comparison animation share
    it), so the GL context and the face list are built once. Call `close()` when the figure is
    finished — `animation.py` does this from a context manager.
    """

    def __init__(self, size: int = DEFAULT_MESH_SIZE, elev: float = 20.0, azim: float = -70.0,
                 colour: tuple = MESH_COLOUR):
        import pyrender

        self.size = int(size)
        self.faces = smplh_faces()

        # Same spherical convention as matplotlib's view_init, so the mesh and the skeleton show
        # the body from an identical angle: matplotlib's data axes (x, y, z) are our
        # (X, Z_fwd, Y_up), so its azimuth sweeps the world XZ plane and its elevation lifts Y.
        el, az = np.radians(elev), np.radians(azim)
        self._direction = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])

        self._renderer = pyrender.OffscreenRenderer(self.size, self.size)
        self._scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0, 1.0), ambient_light=(0.3, 0.3, 0.3))
        self._camera_node = self._scene.add(
            pyrender.PerspectiveCamera(yfov=CAMERA_YFOV, aspectRatio=1.0), pose=np.eye(4))
        # Parented to the camera, so the body stays lit from wherever it is being viewed.
        for pose, intensity in _LIGHT_RIG:
            self._scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                            pose=pose, parent_node=self._camera_node)
        self._material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*colour, 1.0), metallicFactor=0.05, roughnessFactor=0.75,
            alphaMode="OPAQUE")

    def render(self, vertices: np.ndarray, target, half_height: float, overlay_points=None):
        """Render one frame, and optionally project world points into the same pixels.

        vertices      : (V, 3) world-space, Y-up.
        target        : (3,) world point the camera looks at — pass the pelvis, so the viewport
                        follows the body exactly as the skeleton panels' does.
        half_height   : metres visible above and below `target`; fixed for a whole clip, so the
                        body does not breathe in and out as it moves.
        overlay_points: (N, 3) world points to project, or None.

        Returns (rgb (size, size, 3) uint8, pixels (N, 2) or None).
        """
        import pyrender
        import trimesh

        distance = half_height / np.tan(CAMERA_YFOV / 2.0)
        eye = np.asarray(target, dtype=np.float64) + self._direction * distance
        pose = _look_at(eye, target)
        self._scene.set_pose(self._camera_node, pose)

        # process=False: skip trimesh's validation/merge pass — the SMPL topology is fixed and
        # known good, and this runs once per frame.
        node = self._scene.add(pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices, self.faces, process=False), material=self._material))
        try:
            rgb, _ = self._renderer.render(self._scene)
        finally:
            self._scene.remove_node(node)

        return rgb, (None if overlay_points is None else self._project(overlay_points, pose))

    def _project(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """World points (N, 3) → pixel coordinates (N, 2) under the pose just rendered."""
        view = np.linalg.inv(pose)                              # world → camera
        cam = np.asarray(points, dtype=np.float64) @ view[:3, :3].T + view[:3, 3]
        depth = np.maximum(-cam[:, 2], 1e-6)                    # camera looks down -Z
        focal = 1.0 / np.tan(CAMERA_YFOV / 2.0)                 # aspectRatio = 1, so one focal
        x_ndc, y_ndc = focal * cam[:, 0] / depth, focal * cam[:, 1] / depth
        return np.stack([(x_ndc + 1.0) * 0.5 * self.size,
                         (1.0 - y_ndc) * 0.5 * self.size], axis=-1)

    def close(self):
        """Release the GL context. Safe to call twice."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None


@contextmanager
def mesh_renderer(enabled: bool, size: int = DEFAULT_MESH_SIZE, elev: float = 20.0,
                  azim: float = -70.0):
    """A MeshRenderer for the life of one figure, or None when the figure has no mesh panel."""
    if not enabled:
        yield None
        return
    renderer = MeshRenderer(size=size, elev=elev, azim=azim)
    try:
        yield renderer
    finally:
        renderer.close()
