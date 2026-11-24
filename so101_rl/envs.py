"""SO-101 MuJoCo environments for the EECE 576 domain-randomization project.

Two Gymnasium tasks share one world (``scene_pick_place.xml`` plus Menagerie's SO-101 arm):

* ``So101ReachEnv``: stage 1 of the proposal. Move the gripper to a sampled target; dense
  distance reward; success when the gripper *ends* the episode within ``success_tol`` of it.
* ``So101PickPlaceEnv``: stage 2. Pick the red cube and leave it in the green box for a settling
  interval; staged reward (reach, grasp/lift, carry, place).

Observations (proposal sec. 3): joint positions, joint velocities and gripper state, optionally the
fixed front RGB camera, stacked over the last ``history`` control steps. Object poses are used for
rewards only; the reach target is given to the policy because it has no other way to see it.

Actions: six values in [-1, 1], five arm joint-position targets and a gripper command, tracked by
the model's position servos. ``action_mode="delta"`` moves the current targets by at most
``delta_scale`` radians per step; ``"absolute"`` maps [-1, 1] onto each actuator's control range;
``"relative"`` sets the target to the measured joints plus ``action * relative_scale`` (the motion
from where the arm is now, which BC predicts far more precisely than an absolute target).

The physical knobs the experiments vary are exposed on ``So101World``: ``set_object_friction``
(object-finger sliding friction). Visual randomization (texture, lighting, camera pose) is added in
a later milestone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import randomization

HERE = Path(__file__).absolute().parent
TABLETOP = HERE.parent / "lerobot-tabletop"
SCENE = HERE / "scene_pick_place.xml"
ARM_XML = TABLETOP / "robotstudio_so101" / "so101.xml"
MOUNT_SITE = "so101_mount"
ARM_COLOR = (0.30, 0.62, 0.68)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = slice(0, 5)
GRIPPER = 5
# Arm folded above the table, gripper half open; LeRobot so101 units (degrees, gripper 0-100).
HOME_DEG = (0.0, -60.0, 60.0, 35.0, 0.0, 30.0)

EE_SITE = "gripperframe"
CUBE = "red_cube"
BOX = "box"
TARGET = "reach_target"
CUBE_HALF = 0.015
BOX_INNER_HALF = 0.045
IN_BOX_MAX_HEIGHT = 0.06


VARIANTS = ("mounted", "legacy")
MOUNTED_DIR = TABLETOP / "mounts" / "generated"


def prepare_mounted(rebuild: bool = True) -> Path:
    """Generate the mounted scene/arm XML (lerobot-tabletop/mounts) and return its directory.

    Call with ``rebuild=True`` once in the main process (picks up mounts/config.json edits);
    worker processes then use ``rebuild=False`` so parallel envs never rewrite the files at once.
    """
    scene = MOUNTED_DIR / "scene.xml"
    if rebuild or not scene.is_file():
        import sys

        if str(TABLETOP) not in sys.path:
            sys.path.insert(0, str(TABLETOP))
        from mounts.environment import build_variant

        build_variant()
    return MOUNTED_DIR


def build_model(variant: str = "mounted") -> mujoco.MjModel:
    """Compile the task scene with the SO-101 attached at its mount site.

    ``mounted``: Tony's stock-mount assembly (overhead stand + camera, hex-nut wrist mount, arm on
    the base adapter), generated from ../lerobot-tabletop/mounts. A reach-target marker is added.
    ``legacy``: ``scene_pick_place.xml`` with Menagerie's unmodified arm at the world origin.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "mounted":
        root = prepare_mounted(rebuild=False)
        scene, arm_xml = root / "scene.xml", root / "robotstudio_so101" / "so101.xml"
    else:
        scene, arm_xml = SCENE, ARM_XML
    spec = mujoco.MjSpec.from_file(str(scene))
    arm = mujoco.MjSpec.from_file(str(arm_xml))
    if variant == "legacy":
        for material in arm.materials:
            if not material.name.startswith("sts3215"):
                material.rgba = [*ARM_COLOR, 1.0]
    if ET.parse(scene).getroot().find("option") is None:  # nosec B314
        for name in dir(arm.option):
            if not name.startswith("_"):
                setattr(spec.option, name, getattr(arm.option, name))
    spec.attach(arm, site=MOUNT_SITE, prefix="")
    randomization.add_textures(spec)  # unused unless an env randomizes textures; rendering is unchanged
    if spec.body(TARGET) is None:
        target = spec.worldbody.add_body(name=TARGET, mocap=True, pos=[0.25, 0, 0.1])
        target.add_geom(
            name=TARGET, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0],
            rgba=[1.0, 0.55, 0.0, 0.6], contype=0, conaffinity=0, group=2,
        )
    return spec.compile()


class So101World:
    """Compiled model and state, unit conversions, kinematics helpers and physical knobs."""

    def __init__(self, fps: int = 30, variant: str = "mounted"):
        self.variant = variant
        self.model = build_model(variant)
        m = self.model
        self.substeps = max(1, round(1 / (fps * m.opt.timestep)))
        m.opt.timestep = 1 / (fps * self.substeps)
        self.fps = fps
        self.data = mujoco.MjData(m)
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        self.lo, self.hi = m.actuator_ctrlrange[self.act].T.copy()
        self.ee_site = m.site(EE_SITE).id
        self.cube_body = m.body(CUBE).id
        self.cube_geom = int(np.flatnonzero(m.geom_bodyid == self.cube_body)[0])
        self.cube_qadr = int(m.jnt_qposadr[m.body_jntadr[self.cube_body]])
        # Scene-defined start position of the cube and the arm base (moves with the mount).
        self.cube_start = m.qpos0[self.cube_qadr : self.cube_qadr + 3].copy()
        self.base_body = m.body("base").id
        self.box_body = m.body(BOX).id
        self.target_mocap = int(m.body_mocapid[m.body(TARGET).id])
        # Finger/jaw collision geoms carry priority 1, so their friction governs finger-object contact.
        self.finger_geoms = np.flatnonzero(m.geom_priority == 1)
        self.nominal_friction = float(m.geom_friction[self.finger_geoms[0], 0])
        self._ik_data = mujoco.MjData(m)
        # One renderer per image size: recreating a renderer (a new GL context) costs ~100 ms.
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        mujoco.mj_forward(m, self.data)
        self._pointing = self._pointing_axis()
        self.base_pos = self.data.xpos[self.base_body].copy()
        # The fixed task camera the policy sees (proposal: one external RGB camera).
        self.task_camera = "overhead" if variant == "mounted" else "front"

    # --- units -------------------------------------------------------------------------------
    def deg_to_ctrl(self, values) -> np.ndarray:
        """LeRobot units (degrees, gripper 0-100) to actuator radians."""
        values = np.asarray(values, dtype=np.float64)
        ctrl = np.deg2rad(values)
        ctrl[GRIPPER] = self.lo[GRIPPER] + values[GRIPPER] / 100 * (self.hi[GRIPPER] - self.lo[GRIPPER])
        return np.clip(ctrl, self.lo, self.hi)

    def ctrl_to_deg(self, ctrl) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float64)
        out = np.rad2deg(ctrl)
        out[GRIPPER] = (ctrl[GRIPPER] - self.lo[GRIPPER]) / (self.hi[GRIPPER] - self.lo[GRIPPER]) * 100
        return out

    # --- state -------------------------------------------------------------------------------
    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.dadr].copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self.data.ctrl[self.act].copy()

    def ee_pos(self, data: mujoco.MjData | None = None) -> np.ndarray:
        return (data or self.data).site_xpos[self.ee_site].copy()

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_in_box(self) -> bool:
        center = self.data.xpos[self.box_body]
        rot = self.data.xmat[self.box_body].reshape(3, 3)
        local = (self.data.xpos[self.cube_body] - center) @ rot
        return bool(np.all(np.abs(local[:2]) < BOX_INNER_HALF) and local[2] < IN_BOX_MAX_HEIGHT)

    def set_target(self, pos) -> None:
        self.data.mocap_pos[self.target_mocap] = pos

    # --- control -----------------------------------------------------------------------------
    def reset(self, joints_ctrl: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = joints_ctrl
        self.data.ctrl[self.act] = joints_ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray) -> None:
        """Hold actuator targets (radians) for one control period."""
        self.data.ctrl[self.act] = np.clip(ctrl, self.lo, self.hi)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    # --- physical knobs ----------------------------------------------------------------------
    def set_object_friction(self, mu: float) -> None:
        """Sliding friction of the finger geoms and the cube (the finger value governs their contact)."""
        self.model.geom_friction[self.finger_geoms, 0] = mu
        self.model.geom_friction[self.cube_geom, 0] = mu

    def physics_settings(self) -> dict[str, Any]:
        """Settings the proposal keeps fixed, recorded for reproducibility."""
        m = self.model
        return {
            "variant": self.variant,
            "arm_base_pos": self.base_pos.tolist(),
            "arm_mass": float(sum(m.body_mass[b] for b in range(self.base_body, m.nbody))),
            "timestep": m.opt.timestep,
            "substeps_per_control": self.substeps,
            "integrator": int(m.opt.integrator),
            "cone": int(m.opt.cone),
            "impratio": m.opt.impratio,
            "iterations": m.opt.iterations,
            "servo_kp": m.actuator_gainprm[self.act, 0].tolist(),
            "servo_kv": (-m.actuator_biasprm[self.act, 2]).tolist(),
            "servo_forcerange": m.actuator_forcerange[self.act].tolist(),
            "finger_friction": m.geom_friction[self.finger_geoms[0]].tolist(),
            "finger_condim": int(m.geom_condim[self.finger_geoms[0]]),
            "finger_solref": m.geom_solref[self.finger_geoms[0]].tolist(),
            "cube_friction": m.geom_friction[self.cube_geom].tolist(),
            "cube_condim": int(m.geom_condim[self.cube_geom]),
            "cube_solref": m.geom_solref[self.cube_geom].tolist(),
            "cube_mass": float(m.body_mass[self.cube_body]),
            "friction_rule": "finger geoms have priority 1, so finger friction applies to finger-cube contacts",
        }

    # --- kinematics --------------------------------------------------------------------------
    def _pointing_axis(self) -> np.ndarray:
        """Unit vector, in the EE site frame, from the wrist toward the fingertips."""
        body = self.model.site_bodyid[self.ee_site]
        rot = self.data.site_xmat[self.ee_site].reshape(3, 3)
        v = rot.T @ (self.data.site_xpos[self.ee_site] - self.data.xpos[body])
        return v / np.linalg.norm(v)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """EE site position for joint vector ``q`` (radians), without touching the live state."""
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.ee_site].copy()

    def ik(
        self,
        target: np.ndarray,
        q_init: np.ndarray | None = None,
        point_down: bool = True,
        iters: int = 200,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """Damped least-squares IK on the first four arm joints; returns (q, position error)."""
        m, d = self.model, self._ik_data
        q = (self.qpos if q_init is None else np.asarray(q_init, dtype=np.float64)).copy()
        cols = self.dadr[:4]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        down = np.array([0.0, 0.0, -1.0])
        err = np.inf
        for _ in range(iters):
            d.qpos[:] = self.data.qpos
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[self.ee_site]
            e_pos = target - pos
            err = float(np.linalg.norm(e_pos))
            mujoco.mj_jacSite(m, d, jacp, jacr, self.ee_site)
            rows, errs = [jacp[:, cols]], [e_pos]
            if point_down:
                axis = d.site_xmat[self.ee_site].reshape(3, 3) @ self._pointing
                skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                rows.append(0.05 * (-skew @ jacr[:, cols]))
                errs.append(0.05 * (down - axis))
            J, e = np.vstack(rows), np.concatenate(errs)
            if err < tol and (not point_down or np.linalg.norm(errs[-1]) < 1e-3):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            q[:4] = np.clip(q[:4] + np.clip(dq, -0.2, 0.2), self.lo[:4], self.hi[:4])
        return q, err

    # --- rendering ---------------------------------------------------------------------------
    def render(self, camera: str, height: int, width: int) -> np.ndarray:
        renderer = self._renderers.get((height, width))
        if renderer is None:
            renderer = self._renderers[(height, width)] = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


class _So101TaskEnv(gym.Env):
    """Shared action interface, observation history and optional camera for the SO-101 tasks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    QVEL_SCALE = 0.1  # rad/s -> roughly unit range

    def __init__(
        self,
        episode_steps: int,
        action_mode: str = "delta",
        delta_scale: float = 0.05,
        history: int = 3,
        image: bool = False,
        image_size: tuple[int, int] = (96, 128),
        camera: str | None = None,
        object_friction: float | None = None,
        joint_noise_deg: float = 2.0,
        render_mode: str | None = "rgb_array",
        variant: str = "mounted",
        include_targets: bool = True,
        wrist: bool = False,
        include_velocities: bool = True,
        relative_scale=None,
    ):
        super().__init__()
        # "relative" actions: target = measured joints + action * relative_scale (radians per joint).
        self.relative_scale = None if relative_scale is None else np.asarray(relative_scale, dtype=np.float64)
        if action_mode == "relative" and (self.relative_scale is None or self.relative_scale.shape != (6,)):
            raise ValueError("action_mode='relative' needs relative_scale: six per-joint scales in radians")
        # Joint velocities (and a multi-step history) also let BC extrapolate its own recent motion.
        self.include_velocities = include_velocities
        # Gripper-mounted camera as a second image ("wrist" key); it sees the cube between the
        # fingers when the arm blocks the overhead view during the grasp.
        self.wrist = wrist and image
        # Commanded targets (= the previous action) in the observation. BC must leave them out:
        # with them it learns to copy its last command ("copycat") instead of reading the image.
        self.include_targets = include_targets
        if action_mode not in ("delta", "absolute", "relative"):
            raise ValueError("action_mode must be 'delta', 'absolute' or 'relative'")
        self.episode_steps = episode_steps
        self.action_mode = action_mode
        self.delta_scale = delta_scale
        self.history = history
        self.image = image
        self.image_size = image_size
        self.object_friction = object_friction
        self.joint_noise = np.deg2rad(joint_noise_deg)
        self.render_mode = render_mode
        self.world = So101World(variant=variant)
        self.camera = camera or self.world.task_camera
        if object_friction is not None:
            self.world.set_object_friction(object_friction)
        self.home = self.world.deg_to_ctrl(HOME_DEG)
        self._steps = 0
        self._target_ctrl = self.home.copy()
        # Control latency (evaluation factor "actuator"): commands take effect this many steps late.
        self.action_latency = 0
        # Base of "relative" actions: "measured" joints (as trained) or the previous "command".
        # Under latency the measured joints are stale; the previous command is not.
        self.relative_base = "measured"
        # "resync": build on the previous command, but per joint compare the command sent
        # ``resync_lag`` steps ago with the reading now; where they differ by more than
        # ``resync_threshold`` (radians, per joint) the joint is blocked (e.g. the gripper on the
        # cube), so that joint's base snaps back to its reading. Stops command wind-up.
        self.resync_lag = 3
        # Normal tracking gap |q(t) - cmd(t-3)| in successful episodes stays under ~7.5 deg on the
        # arm joints (any latency 0-3); the gripper's threshold is tighter to catch the squeeze.
        self.resync_threshold = np.deg2rad([10.0] * 5 + [5.0])
        self._sent: deque = deque()
        self.resync_count = np.zeros(6, int)
        # Latch: a joint flagged as blocked stays on its reading until the reading moves again
        # (|q(t) - q(t - lag)| > resync_moving radians); otherwise the gap closes after a resync
        # and the command winds up again for ``lag`` steps before the next detection.
        self.resync_latch = True
        self.resync_moving = np.deg2rad(1.0)
        self._readings: deque = deque()
        self._blocked = np.zeros(6, bool)
        self._pending: deque = deque()
        self._held = self.home.copy()
        self._frames: deque = deque(maxlen=history)
        self._wrist_frames: deque = deque(maxlen=history)
        self._states: deque = deque(maxlen=history)

        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        n = len(self._state_vector())
        spaces_dict = {"state": spaces.Box(-np.inf, np.inf, (n * history,), np.float32)}
        if image:
            h, w = image_size
            spaces_dict["front"] = spaces.Box(0, 255, (h, w, 3 * history), np.uint8)
            if self.wrist:
                spaces_dict["wrist"] = spaces.Box(0, 255, (h, w, 3 * history), np.uint8)
        self.observation_space = spaces.Dict(spaces_dict) if image else spaces_dict["state"]

    # Subclasses: task-specific reset, observation extras, reward.
    def _reset_task(self) -> None: ...
    def _task_state(self) -> np.ndarray:
        return np.zeros(0)
    def _reward(self, action: np.ndarray) -> tuple[float, bool, dict]:
        raise NotImplementedError

    def _state_vector(self) -> np.ndarray:
        w = self.world
        span = w.hi - w.lo
        qpos_n = 2 * (w.qpos - w.lo) / span - 1
        parts = [qpos_n]
        if getattr(self, "include_velocities", True):
            parts.append(w.qvel * self.QVEL_SCALE)
        if self.include_targets:
            parts.append(2 * (self._target_ctrl - w.lo) / span - 1)
        parts.append(self._task_state())
        return np.concatenate(parts).astype(np.float32)

    WRIST_CAMERA = "wrist_cam"

    def _observe(self, fresh: bool = False):
        state = self._state_vector()
        frame = wrist = None
        if self.image:
            frame = self.world.render(self.camera, *self.image_size)
        if self.wrist:
            wrist = self.world.render(self.WRIST_CAMERA, *self.image_size)
        streams = [(self._states, state), (self._frames, frame), (self._wrist_frames, wrist)]
        for buffer, item in streams:
            if item is None:
                continue
            if fresh:
                buffer.clear()
                buffer.extend([item] * self.history)  # pad the history with the first frame
            else:
                buffer.append(item)
        flat = np.concatenate(self._states)
        if not self.image:
            return flat
        obs = {"state": flat, "front": np.concatenate(self._frames, axis=-1)}
        if self.wrist:
            obs["wrist"] = np.concatenate(self._wrist_frames, axis=-1)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        """``options={"joints": q}`` starts from actuator-space joints ``q`` (radians) instead of HOME."""
        super().reset(seed=seed)
        if options and options.get("joints") is not None:
            q = np.asarray(options["joints"], dtype=np.float64).copy()
        else:
            q = self.home.copy()
            q[ARM] += self.np_random.uniform(-self.joint_noise, self.joint_noise, 5)
        q = np.clip(q, self.world.lo, self.world.hi)
        self.world.reset(q)
        self._target_ctrl = q.copy()
        self._pending.clear()
        self._held = q.copy()
        self._sent.clear()
        self._readings.clear()
        self._blocked = np.zeros(6, bool)
        self.resync_count = np.zeros(6, int)
        self._steps = 0
        self._reset_task()
        mujoco.mj_forward(self.world.model, self.world.data)
        return self._observe(fresh=True), {"is_success": False}

    def _apply_action(self, action: np.ndarray) -> None:
        w = self.world
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_mode == "delta":
            self._target_ctrl = np.clip(self._target_ctrl + a * self.delta_scale, w.lo, w.hi)
        elif self.action_mode == "relative":
            # Motion from where the arm is now (measured), not from the last command.
            base = w.qpos if self.relative_base == "measured" else self._target_ctrl.copy()
            if self.relative_base == "resync" and len(self._sent) >= self.resync_lag:
                off = np.abs(w.qpos - self._sent[-self.resync_lag]) > self.resync_threshold
                if self.resync_latch:
                    moving = (np.abs(w.qpos - self._readings[-self.resync_lag]) > self.resync_moving
                              if len(self._readings) >= self.resync_lag else np.ones(6, bool))
                    self._blocked = (self._blocked & ~moving) | off
                    off = self._blocked
                base[off] = w.qpos[off]
                self.resync_count += off
            self._target_ctrl = np.clip(base + a * self.relative_scale, w.lo, w.hi)
        else:
            self._target_ctrl = w.lo + (a + 1) / 2 * (w.hi - w.lo)
        self._sent.append(self._target_ctrl.copy())
        self._readings.append(w.qpos.copy())
        if len(self._sent) > 8:
            self._sent.popleft()
            self._readings.popleft()
        applied = self._target_ctrl
        if self.action_latency:
            self._pending.append(self._target_ctrl.copy())
            applied = self._pending.popleft() if len(self._pending) > self.action_latency else self._held
        self._held = applied
        w.step(applied)

    def step(self, action):
        self._apply_action(action)
        self._steps += 1
        reward, terminated, info = self._reward(np.asarray(action))
        truncated = self._steps >= self.episode_steps
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        return self.world.render(self.camera, 480, 640)

    def close(self):
        self.world.close()


class So101ReachEnv(_So101TaskEnv):
    """Stage 1: move the gripper site to a sampled, reachable target position."""

    # Workspace in front of the arm: x, y relative to the arm base, z above the table (metres).
    WORKSPACE_LO = np.array([0.12, -0.20, 0.03])
    WORKSPACE_HI = np.array([0.32, 0.20, 0.20])

    def __init__(self, episode_steps: int = 90, success_tol: float = 0.02, **kwargs):
        self.success_tol = success_tol
        self.target = np.array([0.25, 0.0, 0.1])
        super().__init__(episode_steps=episode_steps, **kwargs)

    def _sample_target(self) -> np.ndarray:
        """Sample a joint configuration, keep its EE position if it lies in the workspace: reachable by construction."""
        w = self.world
        offset = np.array([w.base_pos[0], w.base_pos[1], 0.0])
        lo, hi = self.WORKSPACE_LO + offset, self.WORKSPACE_HI + offset
        for _ in range(1000):
            q = self.home.copy()
            q[:4] = self.np_random.uniform(w.lo[:4], w.hi[:4])
            p = w.fk(q)
            if np.all(p >= lo) and np.all(p <= hi):
                return p
        raise RuntimeError("Could not sample a reachable target")

    def _reset_task(self) -> None:
        self.target = self._sample_target()
        self.world.set_target(self.target)
        # Park the cube out of the way so it cannot obstruct reaching.
        a = self.world.cube_qadr
        self.world.data.qpos[a : a + 3] = (0.45, 0.30, CUBE_HALF)

    def _task_state(self) -> np.ndarray:
        if not hasattr(self, "world"):
            return np.zeros(6)
        ee = self.world.ee_pos()
        return np.concatenate([self.target, (self.target - ee) * 5])

    def distance(self) -> float:
        return float(np.linalg.norm(self.target - self.world.ee_pos()))

    def _reward(self, action):
        d = self.distance()
        reward = 1.0 - np.tanh(10 * d) + (0.5 if d < self.success_tol else 0.0)
        reward -= 0.01 * float(np.square(action[ARM]).sum())
        final = self._steps >= self.episode_steps
        info = {"distance": d, "is_success": bool(final and d < self.success_tol)}
        return float(reward), False, info


class So101PickPlaceEnv(_So101TaskEnv):
    """Stage 2: pick the red cube and leave it in the green box for ``settle_steps`` steps."""

    # Potential-based shaping (Ng et al. 1999): reward = gamma * phi(s') - phi(s) + success bonus.
    # Each stage of progress pays once when reached (and is taken back if lost, e.g. a dropped
    # cube), so stalling earns nothing and the optimal policy is unchanged by the shaping.
    PHI_LIFT, PHI_CARRY, PHI_IN_BOX, SUCCESS_BONUS = 1.0, 2.0, 3.0, 10.0

    def __init__(
        self,
        episode_steps: int = 900,  # demonstrations average 513 steps; Tony's evaluation deadline is 900
        cube_xy_range: tuple[float, float] = (0.04, 0.06),
        cube_yaw_range: float = 0.0,
        settle_steps: int = 15,
        image: bool = True,
        shaping_gamma: float = 0.99,
        start_joints=None,
        start_jitter_deg: float = 0.0,
        randomize: dict | None = None,
        fixed_factors: dict | None = None,
        **kwargs,
    ):
        # Domain randomization (randomization.py), applied at every reset:
        #   randomize={"factors": [...], "ranges": {...Ranges fields...}} samples a new setting;
        #   fixed_factors={...} applies one fixed setting (evaluation sweeps). Default: nominal.
        self.randomize = randomize
        self.fixed_factors = fixed_factors
        self.randomize_strength = 1.0  # curriculum: 0 = nominal .. 1 = full ranges (set via VecEnv.set_attr)
        self._randomizer = None
        self.last_factors: dict = {}
        self.cube_xy_range = np.asarray(cube_xy_range)
        self.cube_yaw_range = cube_yaw_range
        self.settle_steps = settle_steps
        self.shaping_gamma = shaping_gamma
        # Reset distribution for RL: random rows of start_joints (actuator radians, e.g. the
        # demonstrations' first poses) with +-start_jitter_deg on the arm joints. reset(options=
        # {"joints": q}) still overrides it.
        self.start_joints = None if start_joints is None else np.asarray(start_joints, dtype=np.float64)
        self.start_jitter = np.deg2rad(start_jitter_deg)
        self._in_box_steps = 0
        self._phi = 0.0
        super().__init__(episode_steps=episode_steps, image=image, **kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self.start_joints is not None and not (options and options.get("joints") is not None):
            gym.Env.reset(self, seed=seed)  # seed np_random before sampling the start
            q = self.start_joints[self.np_random.integers(len(self.start_joints))].copy()
            q[ARM] += self.np_random.uniform(-self.start_jitter, self.start_jitter, 5)
            options = {**(options or {}), "joints": q}
            seed = None  # already seeded
        return super().reset(seed=seed, options=options)

    def _potential(self) -> tuple[float, dict]:
        w = self.world
        ee, cube = w.ee_pos(), w.cube_pos()
        box = w.data.xpos[w.box_body].copy()
        d_reach = float(np.linalg.norm(ee - cube))
        lifted = bool(cube[2] > CUBE_HALF + 0.02)
        d_box = float(np.linalg.norm((box + (0, 0, 0.08)) - cube))
        in_box = w.cube_in_box()
        phi = (1 - np.tanh(10 * d_reach)) if not lifted else 1.0
        phi += lifted * (self.PHI_LIFT + self.PHI_CARRY * (1 - np.tanh(5 * d_box)))
        phi += self.PHI_IN_BOX * in_box
        return float(phi), {"d_reach": d_reach, "lifted": lifted, "in_box": in_box}

    def _apply_randomization(self) -> None:
        if self.randomize is None and self.fixed_factors is None:
            return
        if self._randomizer is None:
            self._randomizer = randomization.Randomizer(self.world, camera=self.camera)
        if self.fixed_factors is not None:
            self.last_factors = self.fixed_factors
        else:
            ranges = randomization.Ranges(**self.randomize.get("ranges", {}))
            factors = self.randomize.get("factors", ("texture", "lighting", "camera", "friction"))
            self.last_factors = self._randomizer.sample(self.np_random, ranges, factors,
                                                        strength=getattr(self, "randomize_strength", 1.0))
        self._randomizer.apply(self.last_factors)
        self.action_latency = int(self.last_factors.get("actuator", {}).get("latency_steps", 0))

    def _reset_task(self) -> None:
        self._apply_randomization()
        w = self.world
        a = w.cube_qadr
        xy = w.cube_start[:2] + self.np_random.uniform(-self.cube_xy_range, self.cube_xy_range)
        yaw = self.np_random.uniform(-self.cube_yaw_range, self.cube_yaw_range)
        w.data.qpos[a : a + 3] = (*xy, CUBE_HALF)
        w.data.qpos[a + 3 : a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        w.set_target((0.45, 0.30, -0.2))  # hide the reach marker under the table
        self._in_box_steps = 0
        mujoco.mj_forward(w.model, w.data)
        self._phi, _ = self._potential()

    def _task_state(self) -> np.ndarray:
        # Gripper state: commanded and measured opening, normalized (also inside qpos/targets).
        return np.zeros(0)

    def _reward(self, action):
        phi, s = self._potential()
        d_reach, lifted, in_box = s["d_reach"], s["lifted"], s["in_box"]
        # Released: cube in box and not between the fingers.
        released = in_box and d_reach > 0.03
        self._in_box_steps = self._in_box_steps + 1 if released else 0
        success = self._in_box_steps >= self.settle_steps
        reward = self.shaping_gamma * phi - self._phi + self.SUCCESS_BONUS * success
        reward -= 0.001 * float(np.square(action[ARM]).sum())
        self._phi = phi
        info = {
            "is_success": success,
            "cube_lifted": bool(lifted),
            "cube_in_box": in_box,
            "reach_distance": d_reach,
            "grasp_depth_cm": self.grasp_depth_cm() if lifted else float("nan"),
        }
        return float(reward), success, info

    def grasp_depth_cm(self) -> float:
        """Cube centre along the fingers, from the gripper site (cm): + toward the fingertips.

        Demonstrated carries hold it at +0.15 cm (median); BC grasps at +1.2-1.4 cm slip out.
        """
        w = self.world
        rot = w.data.site_xmat[w.ee_site].reshape(3, 3)
        return float(100 * (rot.T @ (w.cube_pos() - w.ee_pos())) @ w._pointing)


gym.register("So101Reach-v0", entry_point=So101ReachEnv)
gym.register("So101PickPlace-v0", entry_point=So101PickPlaceEnv)
