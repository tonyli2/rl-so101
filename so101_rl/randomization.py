"""Domain randomization for the SO-101 pick-and-place scene (proposal sec. 3, "Training"/"Evaluation").

The four factors of the proposal, each applied by editing the compiled MuJoCo model in place
(no recompilation, so it can change every episode):

* ``texture``  - table, floor and wall appearance: one of several procedural textures with a random
  tint. Training and testing draw from disjoint texture sets ("held-out textures").
* ``lighting`` - direction, brightness, colour tint and ambient level of the scene lights.
* ``camera``   - overhead (task) camera position, rotation and vertical field of view.
* ``friction`` - object-finger sliding friction (physics: matters for RL, not for BC images).

A factor setting is a plain dict of numbers, so evaluation sweeps can set one factor to a fixed
value (``Randomizer.apply``) while the others stay nominal. ``Randomizer.sample`` draws a setting
from ranges, for randomized training.

Textures exist in the model from compile time (``add_textures`` on the MjSpec, called by
envs.build_model), because a renderer uploads textures when it is created; switching a material's
texture id afterwards is then just an array write.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Procedural textures (MuJoCo builtins). Index ranges split them into training and held-out sets.
TEXTURES = [
    # name, builtin, rgb1, rgb2, mark, random
    ("tex_checker_warm", "checker", (0.80, 0.70, 0.55), (0.60, 0.50, 0.38), "none", 0.0),
    ("tex_checker_grey", "checker", (0.70, 0.70, 0.70), (0.45, 0.45, 0.45), "none", 0.0),
    ("tex_grad_blue", "gradient", (0.55, 0.65, 0.80), (0.25, 0.30, 0.45), "none", 0.0),
    ("tex_noise_sand", "flat", (0.78, 0.72, 0.60), (0.40, 0.35, 0.28), "random", 0.25),
    ("tex_edge_green", "checker", (0.45, 0.60, 0.45), (0.30, 0.40, 0.30), "edge", 0.0),
    ("tex_noise_dark", "flat", (0.30, 0.28, 0.26), (0.10, 0.10, 0.10), "random", 0.30),
    # held out: only used by evaluation
    ("tex_checker_red", "checker", (0.75, 0.45, 0.40), (0.50, 0.25, 0.22), "none", 0.0),
    ("tex_grad_purple", "gradient", (0.65, 0.55, 0.75), (0.30, 0.22, 0.40), "none", 0.0),
    ("tex_noise_bright", "flat", (0.95, 0.95, 0.92), (0.60, 0.60, 0.55), "random", 0.35),
]
TRAIN_TEXTURES = list(range(6))
HELDOUT_TEXTURES = list(range(6, 9))
SURFACES = ("table", "floor", "wall")  # material names in the mounted scene


def add_textures(spec: mujoco.MjSpec) -> None:
    """Add the procedural textures and one switchable material per surface to a scene spec."""
    for name, builtin, rgb1, rgb2, mark, random in TEXTURES:
        tex = spec.add_texture()
        tex.name = name
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = {"checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                       "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                       "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT}[builtin]
        tex.mark = {"none": mujoco.mjtMark.mjMARK_NONE, "edge": mujoco.mjtMark.mjMARK_EDGE,
                    "random": mujoco.mjtMark.mjMARK_RANDOM}[mark]
        tex.random = random
        tex.rgb1, tex.rgb2 = list(rgb1), list(rgb2)
        tex.markrgb = list(rgb2)
        tex.width = tex.height = 256


@dataclass
class Ranges:
    """Half-ranges (or sets) sampled uniformly for randomized training."""

    textures: list = field(default_factory=lambda: list(TRAIN_TEXTURES))
    tint: float = 0.15                  # +- per RGB channel on the surface material
    light_dir_deg: float = 25.0         # tilt of each light's direction
    light_scale: tuple = (0.6, 1.4)     # brightness multiplier
    light_tint: float = 0.10            # +- per RGB channel of light colour
    ambient: tuple = (0.1, 0.35)        # headlight ambient
    camera_pos_m: float = 0.015         # +- per axis
    camera_rot_deg: float = 3.0         # +- about each axis
    camera_fovy_deg: float = 3.0        # +- vertical field of view
    friction: tuple = (0.4, 1.2)        # object-finger sliding friction
    # Per-factor cap on strength (0-1), from the calibration pilot (calibrate_ranges.py): each
    # factor's ranges are scaled by curriculum strength x its cap.
    factor_strength: dict = field(default_factory=lambda: {"texture": 1.0, "lighting": 1.0,
                                                           "camera": 1.0, "friction": 1.0})


class Randomizer:
    """Applies factor settings to one compiled model; remembers nominal values to restore them."""

    def __init__(self, world, camera: str = "overhead"):
        self.world = world
        m = world.model
        self.camera = m.camera(camera).id
        self.cam_pos0 = m.cam_pos[self.camera].copy()
        self.cam_quat0 = m.cam_quat[self.camera].copy()
        self.cam_fovy0 = float(m.cam_fovy[self.camera])
        self.light_dir0 = m.light_dir.copy()
        self.light_diffuse0 = m.light_diffuse.copy()
        self.ambient0 = m.vis.headlight.ambient.copy()
        self.surface_mats = {}
        for s in SURFACES:
            mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, s)
            if mid >= 0:
                self.surface_mats[s] = mid
        self.mat_rgba0 = {s: m.mat_rgba[i].copy() for s, i in self.surface_mats.items()}
        self.mat_texid0 = {s: m.mat_texid[i].copy() for s, i in self.surface_mats.items()}
        self.mat_texrepeat0 = {s: m.mat_texrepeat[i].copy() for s, i in self.surface_mats.items()}
        self.tex_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, t[0]) for t in TEXTURES]
        self.friction0 = world.nominal_friction

    # --- sampling -----------------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, ranges: Ranges, factors=("texture", "lighting", "camera", "friction"),
               strength: float = 1.0) -> dict:
        """Draw a setting. ``strength`` in [0, 1] shrinks every range toward nominal (curriculum):
        0 = nominal, 1 = the full ranges. Textures: each surface keeps its nominal look with
        probability 1 - strength."""
        base = float(np.clip(strength, 0.0, 1.0))
        cap = {"texture": 1.0, "lighting": 1.0, "camera": 1.0, "friction": 1.0, **ranges.factor_strength}
        p: dict = {}
        n_lights = len(self.light_dir0)
        s = base * cap["texture"]
        if "texture" in factors:
            p["texture"] = {surf: {"tex": int(rng.choice(ranges.textures)),
                                   "tint": (s * rng.uniform(-ranges.tint, ranges.tint, 3)).tolist()}
                            for surf in self.surface_mats if rng.random() < s}
        s = base * cap["lighting"]
        toward = lambda nominal, lo, hi: (nominal + s * (lo - nominal), nominal + s * (hi - nominal))  # noqa: E731
        if "lighting" in factors:
            a0 = float(np.mean(self.ambient0))
            p["lighting"] = {"tilt_deg": (s * rng.uniform(-ranges.light_dir_deg, ranges.light_dir_deg, (n_lights, 2))).tolist(),
                             "scale": float(rng.uniform(*toward(1.0, *ranges.light_scale))),
                             "tint": (s * rng.uniform(-ranges.light_tint, ranges.light_tint, 3)).tolist(),
                             "ambient": float(rng.uniform(*toward(a0, *ranges.ambient)))}
        s = base * cap["camera"]
        if "camera" in factors:
            p["camera"] = {"pos_m": (s * rng.uniform(-ranges.camera_pos_m, ranges.camera_pos_m, 3)).tolist(),
                           "rot_deg": (s * rng.uniform(-ranges.camera_rot_deg, ranges.camera_rot_deg, 3)).tolist(),
                           "fovy_deg": float(s * rng.uniform(-ranges.camera_fovy_deg, ranges.camera_fovy_deg))}
        s = base * cap["friction"]
        toward = lambda nominal, lo, hi: (nominal + s * (lo - nominal), nominal + s * (hi - nominal))  # noqa: E731
        if "friction" in factors:
            p["friction"] = float(rng.uniform(*toward(self.friction0, *ranges.friction)))
        return p

    # --- applying -----------------------------------------------------------------------------
    def nominal(self) -> None:
        m = self.world.model
        m.cam_pos[self.camera], m.cam_quat[self.camera] = self.cam_pos0, self.cam_quat0
        m.cam_fovy[self.camera] = self.cam_fovy0
        m.light_dir[:], m.light_diffuse[:] = self.light_dir0, self.light_diffuse0
        m.vis.headlight.ambient[:] = self.ambient0
        for s, i in self.surface_mats.items():
            m.mat_rgba[i], m.mat_texid[i] = self.mat_rgba0[s], self.mat_texid0[s]
            m.mat_texrepeat[i] = self.mat_texrepeat0[s]
        self.world.set_object_friction(self.friction0)

    def apply(self, p: dict) -> None:
        """Set factors in ``p`` (as produced by ``sample``); everything else returns to nominal."""
        self.nominal()
        m = self.world.model
        if "texture" in p:
            for s, cfg in p["texture"].items():
                i = self.surface_mats[s]
                m.mat_texid[i, mujoco.mjtTextureRole.mjTEXROLE_RGB] = self.tex_ids[cfg["tex"]]
                m.mat_texrepeat[i] = (6.0, 6.0)  # several pattern repeats per surface
                m.mat_rgba[i, :3] = np.clip(0.85 + np.asarray(cfg["tint"]), 0, 1)  # tint over the texture
        if "lighting" in p:
            L = p["lighting"]
            for k, (a, b) in enumerate(np.deg2rad(L["tilt_deg"])):
                d = self.light_dir0[k].copy()
                rot = np.zeros(9)
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, np.array([a, b, 0.0]), "xyz")
                mujoco.mju_quat2Mat(rot, quat)
                d = rot.reshape(3, 3) @ d
                m.light_dir[k] = d / np.linalg.norm(d)
            m.light_diffuse[:] = np.clip(self.light_diffuse0 * L["scale"] + np.asarray(L["tint"]), 0, 1)
            m.vis.headlight.ambient[:] = L["ambient"]
        if "camera" in p:
            c = p["camera"]
            m.cam_pos[self.camera] = self.cam_pos0 + np.asarray(c["pos_m"])
            dq = np.zeros(4)
            mujoco.mju_euler2Quat(dq, np.deg2rad(c["rot_deg"]), "xyz")
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self.cam_quat0, dq)
            m.cam_quat[self.camera] = q
            m.cam_fovy[self.camera] = self.cam_fovy0 + c["fovy_deg"]
        if "friction" in p:
            self.world.set_object_friction(p["friction"])
