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
``delta_scale`` radians per step; ``"absolute"`` maps [-1, 1] onto each actuator's control range.

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
        self._renderer: mujoco.Renderer | None = None
        self._renderer_size: tuple[int, int] | None = None
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
        if self._renderer is None or self._renderer_size != (height, width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._renderer_size = (height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


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
    ):
        super().__init__()
        if action_mode not in ("delta", "absolute"):
            raise ValueError("action_mode must be 'delta' or 'absolute'")
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
        self._frames: deque = deque(maxlen=history)
        self._states: deque = deque(maxlen=history)

        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        n = len(self._state_vector())
        spaces_dict = {"state": spaces.Box(-np.inf, np.inf, (n * history,), np.float32)}
        if image:
            h, w = image_size
            spaces_dict["front"] = spaces.Box(0, 255, (h, w, 3 * history), np.uint8)
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
        target_n = 2 * (self._target_ctrl - w.lo) / span - 1
        return np.concatenate(
            [qpos_n, w.qvel * self.QVEL_SCALE, target_n, self._task_state()]
        ).astype(np.float32)

    def _observe(self, fresh: bool = False):
        state = self._state_vector()
        frame = None
        if self.image:
            frame = self.world.render(self.camera, *self.image_size)
        if fresh:
            self._states.clear()
            self._frames.clear()
            for _ in range(self.history):
                self._states.append(state)
                if frame is not None:
                    self._frames.append(frame)
        else:
            self._states.append(state)
            if frame is not None:
                self._frames.append(frame)
        flat = np.concatenate(self._states)
        if not self.image:
            return flat
        return {"state": flat, "front": np.concatenate(self._frames, axis=-1)}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        q = self.home.copy()
        q[ARM] += self.np_random.uniform(-self.joint_noise, self.joint_noise, 5)
        q = np.clip(q, self.world.lo, self.world.hi)
        self.world.reset(q)
        self._target_ctrl = q.copy()
        self._steps = 0
        self._reset_task()
        mujoco.mj_forward(self.world.model, self.world.data)
        return self._observe(fresh=True), {"is_success": False}

    def _apply_action(self, action: np.ndarray) -> None:
        w = self.world
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.action_mode == "delta":
            self._target_ctrl = np.clip(self._target_ctrl + a * self.delta_scale, w.lo, w.hi)
        else:
            self._target_ctrl = w.lo + (a + 1) / 2 * (w.hi - w.lo)
        w.step(self._target_ctrl)

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

    def __init__(
        self,
        episode_steps: int = 300,
        cube_xy_range: tuple[float, float] = (0.04, 0.06),
        cube_yaw_range: float = 0.0,
        settle_steps: int = 15,
        image: bool = True,
        **kwargs,
    ):
        self.cube_xy_range = np.asarray(cube_xy_range)
        self.cube_yaw_range = cube_yaw_range
        self.settle_steps = settle_steps
        self._in_box_steps = 0
        super().__init__(episode_steps=episode_steps, image=image, **kwargs)

    def _reset_task(self) -> None:
        w = self.world
        a = w.cube_qadr
        xy = w.cube_start[:2] + self.np_random.uniform(-self.cube_xy_range, self.cube_xy_range)
        yaw = self.np_random.uniform(-self.cube_yaw_range, self.cube_yaw_range)
        w.data.qpos[a : a + 3] = (*xy, CUBE_HALF)
        w.data.qpos[a + 3 : a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        w.set_target((0.45, 0.30, -0.2))  # hide the reach marker under the table
        self._in_box_steps = 0

    def _task_state(self) -> np.ndarray:
        # Gripper state: commanded and measured opening, normalized (also inside qpos/targets).
        return np.zeros(0)

    def _reward(self, action):
        w = self.world
        ee, cube = w.ee_pos(), w.cube_pos()
        box = w.data.xpos[w.box_body].copy()
        d_reach = float(np.linalg.norm(ee - cube))
        lifted = cube[2] > CUBE_HALF + 0.02
        d_box = float(np.linalg.norm((box + (0, 0, 0.08)) - cube))
        in_box = w.cube_in_box()
        # Released: cube in box and not between the fingers.
        released = in_box and d_reach > 0.03
        self._in_box_steps = self._in_box_steps + 1 if released else 0
        success = self._in_box_steps >= self.settle_steps
        reward = (1 - np.tanh(10 * d_reach))
        reward += 1.0 * lifted
        reward += 2.0 * (1 - np.tanh(5 * d_box)) * lifted
        reward += 5.0 * in_box
        reward += 10.0 * success
        reward -= 0.01 * float(np.square(action[ARM]).sum())
        info = {
            "is_success": success,
            "cube_lifted": bool(lifted),
            "cube_in_box": in_box,
            "reach_distance": d_reach,
        }
        return float(reward), success, info


gym.register("So101Reach-v0", entry_point=So101ReachEnv)
gym.register("So101PickPlace-v0", entry_point=So101PickPlaceEnv)
