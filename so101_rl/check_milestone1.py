"""Milestone 1 checks (proposal sec. 3, "Contact"; timeline Oct. 18).

1. Model: compiles, one cube, box, front camera renders.
2. Servos: step response of each arm joint (rise time, overshoot, steady-state error).
3. Scripted grasp at nominal friction: lift, hold, carry, release into the box.
4. Friction sweep: the same grasp with reduced object-finger friction, to find where it slips.

Writes ``outputs/milestone1/report.json`` (including the fixed physics settings) and key-frame PNGs.

    python check_milestone1.py              # full check
    python check_milestone1.py --frictions 1.0 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import mujoco
import numpy as np
from PIL import Image

from envs import CUBE_HALF, GRIPPER, HOME_DEG, JOINTS, VARIANTS, So101World, prepare_mounted

OUT = Path(__file__).parent / "outputs" / "milestone1"  # results go in OUT / <variant>
VARIANT = "mounted"
GRASP_Z = 0.012       # gripper-site height when closing (fingertips just above the table)
HOVER_Z = 0.08
CARRY_Z = 0.11
RELEASE_Z = 0.075
HOLD_STEPS = 60       # 2 s at 30 fps
SLIP_TOL = 0.005      # cube motion relative to the gripper during the hold
GRIPPER_OPEN, GRIPPER_CLOSED = 100.0, 0.0


# --- servo step response -----------------------------------------------------------------------
def servo_check(world: So101World, step_deg: float = 15.0, steps: int = 60) -> dict:
    results = {}
    base = world.deg_to_ctrl((0.0, -30.0, 30.0, 30.0, 0.0, 30.0))
    for j, name in enumerate(JOINTS[:5]):
        world.reset(base)
        for _ in range(30):  # settle under gravity
            world.step(base)
        start = world.qpos[j]
        target = base.copy()
        target[j] += np.deg2rad(step_deg)
        trace = []
        for _ in range(steps):
            world.step(target)
            trace.append(world.qpos[j])
        trace = np.rad2deg(np.array(trace) - start)
        rise = next((i for i, v in enumerate(trace) if v >= 0.9 * step_deg), None)
        results[name] = {
            "step_deg": step_deg,
            "rise_time_90_s": None if rise is None else (rise + 1) / world.fps,
            "overshoot_deg": float(max(0.0, trace.max() - step_deg)),
            "steady_state_error_deg": float(step_deg - trace[-10:].mean()),
            "ok": rise is not None and abs(step_deg - trace[-10:].mean()) < 1.0,
        }
    return results


# --- scripted grasp -------------------------------------------------------------------------------
class Script:
    """Moves joint targets along straight joint-space segments and records what the cube does."""

    def __init__(self, world: So101World, frames: dict | None):
        self.w = world
        self.target = world.ctrl
        self.frames = frames

    def move(self, q_goal: np.ndarray, steps: int) -> None:
        start = self.target.copy()
        for i in range(1, steps + 1):
            self.target = start + (q_goal - start) * i / steps
            self.w.step(self.target)

    def hold(self, steps: int) -> None:
        for _ in range(steps):
            self.w.step(self.target)

    def gripper(self, value: float, steps: int = 30) -> None:
        goal = self.target.copy()
        goal[GRIPPER] = self.w.deg_to_ctrl([0, 0, 0, 0, 0, value])[GRIPPER]
        self.move(goal, steps)

    def snap(self, name: str) -> None:
        if self.frames is not None:
            self.frames[name] = self.w.render("front", 480, 640)
            self.frames[f"{name}_{self.w.task_camera}"] = self.w.render(self.w.task_camera, 480, 640)

    def arm_at(self, pos, point_down: bool = True, roll: float | None = None) -> np.ndarray:
        q, err = self.w.ik(np.asarray(pos, dtype=np.float64), q_init=self.target, point_down=point_down)
        if err > 0.005 and point_down:
            q, err = self.w.ik(np.asarray(pos, dtype=np.float64), q_init=self.target, point_down=False)
        if err > 0.005:
            raise RuntimeError(f"IK failed for {pos}: error {err:.4f} m")
        q[GRIPPER] = self.target[GRIPPER]
        if roll is not None:
            q[4] = roll
        return q


def jaw_roll(world: So101World, q_grasp: np.ndarray, cube_yaw: float = 0.0) -> float:
    """Wrist roll that lines the jaw closing direction up with a cube face (scan, kinematics only)."""
    m, d = world.model, world._ik_data
    fixed = m.geom("fixed_jaw_box1").id
    moving = m.geom("moving_jaw_box1").id
    best, best_cost = 0.0, np.inf
    for roll in np.linspace(world.lo[4], world.hi[4], 361):
        d.qpos[:] = world.data.qpos
        d.qpos[world.qadr] = q_grasp
        d.qpos[world.qadr[4]] = roll
        mujoco.mj_kinematics(m, d)
        v = d.geom_xpos[moving] - d.geom_xpos[fixed]
        angle = np.arctan2(v[1], v[0]) - cube_yaw
        # Closing along a face normal: angle is a multiple of 90 deg. Also keep the jaws level.
        cost = abs(np.sin(2 * angle)) + 2 * abs(v[2]) / (np.linalg.norm(v) + 1e-9)
        if cost < best_cost:
            best, best_cost = roll, cost
    return float(best)


def grasp_trial(friction: float | None, frames: dict | None = None) -> dict:
    world = So101World(variant=VARIANT)
    if friction is not None:
        world.set_object_friction(friction)
    home = world.deg_to_ctrl(HOME_DEG)
    world.reset(home)
    a = world.cube_qadr
    CUBE_START = world.cube_start.copy()
    CUBE_START[2] = CUBE_HALF
    world.data.qpos[a : a + 3] = CUBE_START
    world.data.qpos[a + 3 : a + 7] = (1, 0, 0, 0)
    world.set_target((0.45, 0.30, -0.2))
    mujoco.mj_forward(world.model, world.data)
    s = Script(world, frames)
    s.hold(15)

    box = world.data.xpos[world.box_body].copy()
    s.gripper(GRIPPER_OPEN)
    q_hover = s.arm_at((*CUBE_START[:2], HOVER_Z))
    roll = jaw_roll(world, q_hover)
    q_hover[4] = roll
    s.move(q_hover, 60)
    s.snap("1_hover")
    s.move(s.arm_at((*CUBE_START[:2], GRASP_Z), roll=roll), 45)
    s.hold(10)
    s.gripper(GRIPPER_CLOSED, 30)
    s.hold(15)
    s.snap("2_grasp")
    s.move(s.arm_at((*CUBE_START[:2], CARRY_Z), roll=roll), 60)
    cube_lift = world.cube_pos()
    lifted = bool(cube_lift[2] > CUBE_HALF + 0.04)

    # Hold: does the cube move relative to the gripper?
    rel0 = world.cube_pos() - world.ee_pos()
    s.hold(HOLD_STEPS)
    rel1 = world.cube_pos() - world.ee_pos()
    slip = float(np.linalg.norm(rel1 - rel0))
    held = bool(lifted and world.cube_pos()[2] > CUBE_HALF + 0.04 and slip < SLIP_TOL)
    s.snap("3_hold")

    s.move(s.arm_at((box[0], box[1], CARRY_Z), point_down=False), 75)
    carried = bool(world.cube_pos()[2] > CUBE_HALF + 0.04)
    s.move(s.arm_at((box[0], box[1], RELEASE_Z), point_down=False), 30)
    s.snap("4_over_box")
    s.gripper(GRIPPER_OPEN, 20)
    s.move(s.arm_at((box[0], box[1], CARRY_Z), point_down=False), 30)
    s.hold(45)
    s.snap("5_released")
    cube_vel = float(np.linalg.norm(world.data.qvel[world.model.jnt_dofadr[world.model.body_jntadr[world.cube_body]]:][:3]))
    placed = world.cube_in_box() and cube_vel < 0.01
    return {
        "friction": world.model.geom_friction[world.finger_geoms[0], 0].item(),
        "wrist_roll_deg": float(np.rad2deg(roll)),
        "lifted": lifted,
        "lift_height_m": float(cube_lift[2]),
        "held_2s": held,
        "slip_during_hold_m": slip,
        "carried_to_box": carried,
        "released_in_box": bool(placed),
        "final_cube_pos": world.cube_pos().round(4).tolist(),
        "physics": world.physics_settings() if friction is None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.5, 0.2, 0.15, 0.12, 0.1, 0.05])
    parser.add_argument("--variant", choices=VARIANTS, default="mounted")
    args = parser.parse_args()
    global OUT, VARIANT
    VARIANT = args.variant
    OUT = OUT / VARIANT
    OUT.mkdir(parents=True, exist_ok=True)
    if VARIANT == "mounted":
        prepare_mounted(rebuild=True)

    world = So101World(variant=VARIANT)
    report: dict = {"model": {
        "bodies": world.model.nbody, "dofs": world.model.nv, "actuators": world.model.nu,
        "fps": world.fps, "joints": list(JOINTS),
    }}
    img = world.render("front", 480, 640)
    Image.fromarray(img).save(OUT / "0_scene.png")
    report["model"]["front_camera_ok"] = bool(img.std() > 1)

    report["servos"] = servo_check(world)
    for name, r in report["servos"].items():
        print(f"servo {name:13s} rise {r['rise_time_90_s']}s  overshoot {r['overshoot_deg']:.2f}  "
              f"sse {r['steady_state_error_deg']:.2f} deg  {'OK' if r['ok'] else 'FAIL'}")

    frames: dict = {}
    nominal = grasp_trial(None, frames)
    for name, frame in frames.items():
        Image.fromarray(frame).save(OUT / f"{name}.png")
    report["physics_settings"] = nominal.pop("physics")
    report["grasp_nominal"] = nominal
    print("nominal:", {k: v for k, v in nominal.items() if k != "final_cube_pos"})

    report["friction_sweep"] = []
    for mu in args.frictions:
        r = grasp_trial(mu)
        r.pop("physics")
        report["friction_sweep"].append(r)
        print(f"mu={mu:<5} lifted={r['lifted']!s:5} held={r['held_2s']!s:5} slip={r['slip_during_hold_m']*1000:6.2f} mm "
              f"carried={r['carried_to_box']!s:5} placed={r['released_in_box']}")

    ok = (report["model"]["front_camera_ok"] and all(r["ok"] for r in report["servos"].values())
          and nominal["held_2s"] and nominal["released_in_box"])
    report["milestone1_pass"] = bool(ok)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: o.item()))
    print(f"\nMILESTONE 1 {'PASS' if ok else 'FAIL'}; report and frames in {OUT}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
