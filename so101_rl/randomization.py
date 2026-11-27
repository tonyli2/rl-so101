"""Domain randomization for the SO-101 pick-and-place scene (proposal sec. 3, "Training"/"Evaluation").

The four factors of the proposal, each applied by editing the compiled MuJoCo model in place
(no recompilation, so it can change every episode):

* ``texture``  - table, floor and wall appearance: one of several procedural textures with a random
  tint. Training and testing draw from disjoint texture sets ("held-out textures").
* ``lighting`` - direction, brightness, colour tint and ambient level of the scene lights.
* ``camera``   - overhead (task) camera position, rotation and vertical field of view; with
                 ``wrist_*`` ranges set, the wrist camera too (its real mounting is unmeasured).
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
SURFACES = ("table", "floor", "wall", "paper")  # material names in the mounted scene (paper: enclosure)


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
    # Wrist camera (CAD pose, not measured on the robot). 0 = not randomized (the old default).
    wrist_pos_m: float = 0.0
    wrist_rot_deg: float = 0.0
    wrist_fovy_deg: float = 0.0
    friction: tuple = (0.4, 1.2)        # object-finger sliding friction
    # Control latency in 33 ms control steps, sampled uniformly from 0..this. Measured on the real
    # arm: camera ~26 ms + frame age 0-33 ms + servo dead time ~40-50 ms = ~76-110 ms (2.3-3.3 steps).
    latency_steps: int = 3
    # Sensing delay (0 = off, the default): joint readings / camera frames this many steps old,
    # sampled uniformly from 0..this, independently of the command latency above.
    obs_latency_steps: int = 0
    image_latency_steps: int = 0
    # Lower ends (the real arm is never faster than ~2 command steps) and the chance of an episode
    # with no delay at all (so the policy still works when nothing is late).
    latency_min_steps: int = 0
    image_latency_min_steps: int = 0
    zero_delay_prob: float = 0.0
    # "appearance": Tony's visual protocol (lerobot-tabletop recording/visual_randomization.py,
    # release overlay): material gloss, surface tones, light direction/intensity/colour and camera
    # effects (exposure, white balance, noise, blur), each axis on with probability = strength;
    # plus a per-episode image shift of up to +-image_shift_px pixels per camera.
    image_shift_px: int = 4
    # Per-factor cap on strength (0-1), from the calibration pilot (calibrate_ranges.py): each
    # factor's ranges are scaled by curriculum strength x its cap.
    factor_strength: dict = field(default_factory=lambda: {"texture": 1.0, "lighting": 1.0,
                                                           "camera": 1.0, "friction": 1.0,
                                                           "actuator": 1.0, "appearance": 1.0})


class Randomizer:
    """Applies factor settings to one compiled model; remembers nominal values to restore them."""

    def __init__(self, world, camera: str = "overhead"):
        self.world = world
        m = world.model
        self.camera = m.camera(camera).id
        self.wrist = m.camera("wrist_cam").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam") >= 0 else None
        cams = [c for c in (self.camera, self.wrist) if c is not None]
        self.cam_pos0 = {c: m.cam_pos[c].copy() for c in cams}
        self.cam_quat0 = {c: m.cam_quat[c].copy() for c in cams}
        self.cam_fovy0 = {c: float(m.cam_fovy[c]) for c in cams}
        # Cameras with measured intrinsics (sensorsize + focal length) ignore cam_fovy: their field
        # of view is changed through the focal length instead.
        self.cam_intrinsic0 = {c: m.cam_intrinsic[c].copy() for c in cams}
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
        # Actuators (position servos: force = kp * (target - q) - kv * qdot, clipped to forcerange)
        self.act = world.act
        self.gain0 = m.actuator_gainprm[self.act].copy()
        self.bias0 = m.actuator_biasprm[self.act].copy()
        self.forcerange0 = m.actuator_forcerange[self.act].copy()
        # Appearance factor (Tony's protocol) changes these; nominal() restores them.
        self.mat_specular0, self.mat_shininess0 = m.mat_specular.copy(), m.mat_shininess.copy()
        self.mat_rgba_all0, self.geom_rgba0 = m.mat_rgba.copy(), m.geom_rgba.copy()
        self.light_specular0, self.light_pos0 = m.light_specular.copy(), m.light_pos.copy()
        self.head_diffuse0 = m.vis.headlight.diffuse.copy()
        self.appearance_mats = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in APPEARANCE_MATS}

    # --- sampling -----------------------------------------------------------------------------
    def sample(self, rng: np.random.Generator, ranges: Ranges, factors=("texture", "lighting", "camera", "friction"),
               strength: float = 1.0) -> dict:
        """Draw a setting. ``strength`` in [0, 1] shrinks every range toward nominal (curriculum):
        0 = nominal, 1 = the full ranges. Textures: each surface keeps its nominal look with
        probability 1 - strength."""
        base = float(np.clip(strength, 0.0, 1.0))
        cap = {"texture": 1.0, "lighting": 1.0, "camera": 1.0, "friction": 1.0, "actuator": 1.0,
               "appearance": 1.0, **ranges.factor_strength}
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
            if self.wrist is not None and (ranges.wrist_pos_m or ranges.wrist_rot_deg or ranges.wrist_fovy_deg):
                p["camera"]["wrist"] = {
                    "pos_m": (s * rng.uniform(-ranges.wrist_pos_m, ranges.wrist_pos_m, 3)).tolist(),
                    "rot_deg": (s * rng.uniform(-ranges.wrist_rot_deg, ranges.wrist_rot_deg, 3)).tolist(),
                    "fovy_deg": float(s * rng.uniform(-ranges.wrist_fovy_deg, ranges.wrist_fovy_deg))}
        s = base * cap["friction"]
        toward = lambda nominal, lo, hi: (nominal + s * (lo - nominal), nominal + s * (hi - nominal))  # noqa: E731
        if "friction" in factors:
            p["friction"] = float(rng.uniform(*toward(self.friction0, *ranges.friction)))
        s = base * cap["actuator"]
        if "actuator" in factors:  # latency only (servo gain/torque changes barely mattered in the sweep)
            def draw(lo: int, hi: int) -> int:
                top = round(s * hi)
                return int(rng.integers(min(round(s * lo), top), top + 1))
            p["actuator"] = {"latency_steps": draw(ranges.latency_min_steps, ranges.latency_steps)}
            if ranges.obs_latency_steps:
                p["actuator"]["obs_latency_steps"] = draw(0, ranges.obs_latency_steps)
            if ranges.image_latency_steps:
                p["actuator"]["image_latency_steps"] = draw(ranges.image_latency_min_steps, ranges.image_latency_steps)
            if rng.random() < ranges.zero_delay_prob:
                p["actuator"] = {k: 0 for k in p["actuator"]}
        s = base * cap["appearance"]
        if "appearance" in factors:
            p["appearance"] = sample_appearance(rng, s, ranges.image_shift_px)
        return p

    # --- applying -----------------------------------------------------------------------------
    def nominal(self) -> None:
        m = self.world.model
        for c in self.cam_pos0:
            m.cam_pos[c], m.cam_quat[c] = self.cam_pos0[c], self.cam_quat0[c]
            m.cam_fovy[c], m.cam_intrinsic[c] = self.cam_fovy0[c], self.cam_intrinsic0[c]
        m.light_dir[:], m.light_diffuse[:] = self.light_dir0, self.light_diffuse0
        m.vis.headlight.ambient[:] = self.ambient0
        for s, i in self.surface_mats.items():
            m.mat_rgba[i], m.mat_texid[i] = self.mat_rgba0[s], self.mat_texid0[s]
            m.mat_texrepeat[i] = self.mat_texrepeat0[s]
        self.world.set_object_friction(self.friction0)
        m.actuator_gainprm[self.act] = self.gain0
        m.actuator_biasprm[self.act] = self.bias0
        m.actuator_forcerange[self.act] = self.forcerange0
        m.mat_specular[:], m.mat_shininess[:] = self.mat_specular0, self.mat_shininess0
        m.mat_rgba[:], m.geom_rgba[:] = self.mat_rgba_all0, self.geom_rgba0
        m.light_specular[:], m.light_pos[:] = self.light_specular0, self.light_pos0
        m.vis.headlight.diffuse[:] = self.head_diffuse0

    def _apply_camera(self, cam: int, c: dict) -> None:
        m = self.world.model
        m.cam_pos[cam] = self.cam_pos0[cam] + np.asarray(c["pos_m"])
        dq = np.zeros(4)
        mujoco.mju_euler2Quat(dq, np.deg2rad(c["rot_deg"]), "xyz")
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self.cam_quat0[cam], dq)
        m.cam_quat[cam] = q
        f0 = self.cam_fovy0[cam]
        if m.cam_sensorsize[cam].any():  # measured intrinsics: scale the focal length
            scale = np.tan(np.deg2rad(f0) / 2) / np.tan(np.deg2rad(f0 + c["fovy_deg"]) / 2)
            m.cam_intrinsic[cam, :2] = self.cam_intrinsic0[cam][:2] * scale
        else:
            m.cam_fovy[cam] = f0 + c["fovy_deg"]

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
            self._apply_camera(self.camera, p["camera"])
            if "wrist" in p["camera"] and self.wrist is not None:
                self._apply_camera(self.wrist, p["camera"]["wrist"])
        if "friction" in p:
            self.world.set_object_friction(p["friction"])
        if "actuator" in p:  # servo dynamics (latency is applied by the env, see actuator_latency)
            a = p["actuator"]
            kp, kv = a.get("kp_scale", 1.0), a.get("kv_scale", a.get("kp_scale", 1.0))
            m.actuator_gainprm[self.act, 0] = self.gain0[:, 0] * kp
            m.actuator_biasprm[self.act, 1] = self.bias0[:, 1] * kp
            m.actuator_biasprm[self.act, 2] = self.bias0[:, 2] * kv
            m.actuator_forcerange[self.act] = self.forcerange0 * a.get("torque_scale", 1.0)
        if "appearance" in p:
            apply_appearance(m, self.appearance_mats, p["appearance"])


# --- Tony's appearance protocol ---------------------------------------------------------------
# Ported from lerobot-tabletop/recording/visual_randomization.py (dataset-so101-physical-v1-200ep
# training-tools overlay): the same axes and ranges, sampled from the env's generator. Each axis is
# applied with probability = strength (curriculum), like texture swaps above.
APPEARANCE_MATS = ("red_pla", "box", "paper", "sts3215_03a_v1_material", "printed_mount_black")
APPEARANCE_AXES = ("material_gloss", "surface_tone", "light_direction", "light_intensity", "light_color",
                   "camera_effects")


def sample_appearance(rng: np.random.Generator, strength: float, shift_px: int) -> dict:
    p = {
        "material_gloss": {"plastic_specular": rng.uniform(0.05, 0.65), "plastic_shininess": rng.uniform(0.08, 0.75),
                           "arm_specular": rng.uniform(0.18, 0.75), "paper_specular": rng.uniform(0, 0.07),
                           "paper_shininess": rng.uniform(0.02, 0.16)},
        "surface_tone": {"red_hue_shift": rng.uniform(-0.015, 0.015), "red_saturation_scale": rng.uniform(0.85, 1.12),
                         "red_value_scale": rng.uniform(0.8, 1.05), "bin_grey": rng.uniform(0.34, 0.65),
                         "paper_white": rng.uniform(0.84, 0.99), "paper_warmth": rng.uniform(-0.035, 0.035),
                         "arm_black": rng.uniform(0.055, 0.16)},
        "light_direction": {"tilt_x": rng.uniform(-0.65, 0.65), "tilt_y": rng.uniform(-0.65, 0.65)},
        "light_intensity": {"scale": rng.uniform(0.6, 1.4), "ambient_scale": rng.uniform(0.7, 1.25)},
        "light_color": {"rgb_gain": rng.uniform([0.82, 0.91, 0.82], [1.1, 1.03, 1.1]).tolist()},
        "camera_effects": {"exposure": rng.uniform(0.82, 1.18), "channel_gain": rng.uniform(0.95, 1.05, 3).tolist(),
                           "noise_sigma_255": rng.uniform(1.5, 5), "blur_radius_px": rng.uniform(0.15, 0.65)},
    }
    p = {k: ({kk: (float(vv) if np.isscalar(vv) else vv) for kk, vv in v.items()}) for k, v in p.items()}
    p["axes"] = [a for a in APPEARANCE_AXES if rng.random() < strength]
    top = round(strength * shift_px)
    p["shift_px"] = rng.integers(-top, top + 1, (2, 2)).tolist()  # per camera: (dx, dy)
    p["noise_seed"] = int(rng.integers(2**31))
    return p


def apply_appearance(m: mujoco.MjModel, mats: dict, p: dict) -> None:
    """Tony's apply_profile on a model already reset to nominal (Randomizer.nominal)."""
    import colorsys

    axes = set(p["axes"])
    cube, box, paper, arm, mount = (mats[n] for n in APPEARANCE_MATS)
    if "material_gloss" in axes:
        g = p["material_gloss"]
        for i in (cube, box):
            m.mat_specular[i], m.mat_shininess[i] = g["plastic_specular"], g["plastic_shininess"]
        for i in (arm, mount):
            m.mat_specular[i], m.mat_shininess[i] = g["arm_specular"], g["plastic_shininess"]
        m.mat_specular[paper], m.mat_shininess[paper] = g["paper_specular"], g["paper_shininess"]
    if "surface_tone" in axes:
        t = p["surface_tone"]
        h, sat, v = colorsys.rgb_to_hsv(*m.mat_rgba[cube, :3])
        m.mat_rgba[cube, :3] = colorsys.hsv_to_rgb((h + t["red_hue_shift"]) % 1, min(1, sat * t["red_saturation_scale"]),
                                                   min(1, v * t["red_value_scale"]))
        m.mat_rgba[box, :3] = t["bin_grey"]
        m.mat_rgba[paper, :3] = np.clip([t["paper_white"] + t["paper_warmth"], t["paper_white"],
                                         t["paper_white"] - t["paper_warmth"]], 0, 1)
        for i in (arm, mount):
            m.mat_rgba[i, :3] = t["arm_black"]
        for g in range(m.ngeom):
            name = m.geom(g).name or ""
            if (name == "table" or name.startswith("paper_curtain_")) and not np.array_equal(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
                m.geom_rgba[g, :3] = m.mat_rgba[paper, :3]
    if "light_direction" in axes:
        d = np.array([p["light_direction"]["tilt_x"], p["light_direction"]["tilt_y"], -1.0])
        m.light_dir[0] = d / np.linalg.norm(d)
    if "light_intensity" in axes:
        li = p["light_intensity"]
        m.light_diffuse[:] *= li["scale"]
        m.light_specular[:] *= li["scale"]
        m.vis.headlight.diffuse[:] *= li["scale"]
        m.vis.headlight.ambient[:] *= li["ambient_scale"]
    if "light_color" in axes:
        gain = np.asarray(p["light_color"]["rgb_gain"])
        m.light_diffuse[:] *= gain
        m.light_specular[:] *= gain
        m.vis.headlight.diffuse[:] *= gain


def image_effects(p: dict, width: int):
    """Per-frame camera effects (Tony's augment_image, blur and noise scaled from his 640 px to
    ``width``; checked against full-resolution augmentation in eval_suite.py) and the episode's
    image shift. Returns fx(image, camera_index, step) or None."""
    from PIL import Image, ImageEnhance, ImageFilter

    k = width / 640
    on = "camera_effects" in p["axes"]
    c = p["camera_effects"]
    shifts = p.get("shift_px", [[0, 0], [0, 0]])
    if not on and not any(any(s) for s in shifts):
        return None

    def fx(image: np.ndarray, cam: int, step: int) -> np.ndarray:
        if on:
            im = ImageEnhance.Brightness(Image.fromarray(image)).enhance(c["exposure"])
            im = im.filter(ImageFilter.GaussianBlur(c["blur_radius_px"] * k))
            px = np.asarray(im, dtype=np.float32) * np.asarray(c["channel_gain"], dtype=np.float32)
            rng = np.random.default_rng([p["noise_seed"], cam, step])
            px += rng.normal(0, c["noise_sigma_255"] * k, px.shape)
            image = np.clip(px, 0, 255).astype(np.uint8)
        dx, dy = shifts[cam]
        if dx or dy:
            h, w = image.shape[:2]
            pad = np.pad(image, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode="edge")
            image = pad[abs(dy) + dy: abs(dy) + dy + h, abs(dx) + dx: abs(dx) + dx + w]
        return image

    return fx
