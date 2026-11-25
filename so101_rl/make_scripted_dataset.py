"""Scripted pick-and-place demonstrations in the current (measured, real-matched) scene.

Tony's teleoperated demos were recorded in the old scene (cube centred in front of the arm, bin on
its left). The measured scene (lerobot-tabletop/mounts/config.json, Nov 23) moved the cube and
the bin (now on the arm's right) and changed both cameras, so those demos no longer match. Until
new teleoperated demos exist, this script generates demonstrations with an IK expert that has
privileged access to the cube pose:

    hover over the cube (jaws open, aligned to a cube face) -> descend -> close -> lift
    -> carry over the bin -> lower -> open -> retreat

Motion is human-paced: each segment uses a cosine (smooth start/stop) profile with a peak joint
speed of about 2 deg per 33 ms step, like the teleoperated demos. Starts are the teleoperated
demos' first arm poses (+-jitter); the cube is placed within +-cube_xy of its measured spot with
+-cube_yaw.

Noise injection (DART, Laskey et al. 2017): with ``--noise-deg`` the executed arm targets get a
slowly varying random offset, while the recorded label is the expert's clean target. The arm
therefore drifts off the expert path and every label says how to get back, the recovery data plain
behavior cloning lacks. The expert re-plans each segment from the arm's actual pose and the cube's
actual position, so the labels stay valid after drift.

Output (same arrays as make_bc_dataset.py, so train_bc.py reads it unchanged):
``outputs/bc_data/<name>.npz`` and ``<name>.json`` (layout, split, success counts).

    python make_scripted_dataset.py --episodes 5 --name scripted-smoke --video
    python make_scripted_dataset.py --episodes 200 --noise-deg 2.0 --name scripted-v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import numpy as np

from check_milestone1 import jaw_roll
from envs import ARM, CUBE_HALF, GRIPPER, So101PickPlaceEnv, prepare_mounted

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "bc_data"

HOVER_DZ = 0.07        # gripper site above the cube centre while hovering
GRASP_Z = 0.012        # gripper-site height when closing (fingertips just above the table)
CARRY_Z = 0.11
RELEASE_Z = 0.075
GRIPPER_OPEN, GRIPPER_CLOSED = 70.0, 0.0   # LeRobot units (0-100)
CLOSE_OVERLAP = 6                          # descent steps left when the jaws start closing
PEAK_DEG_PER_STEP = 2.0                    # teleop 99th percentile is 0.9-2.1 deg/step on the arm


def smooth(n: int) -> np.ndarray:
    """Cosine profile 0 -> 1 over n steps (zero velocity at both ends)."""
    s = np.arange(1, n + 1) / n
    return 0.5 - 0.5 * np.cos(np.pi * s)


class Expert:
    """Generates the next clean joint target (actuator radians) for one episode."""

    def __init__(self, env: So101PickPlaceEnv, rng: np.random.Generator, noise_deg: float):
        self.env, self.w, self.rng = env, env.world, rng
        self.noise = np.deg2rad(noise_deg)
        self.offset = np.zeros(6)
        self.plan: list[np.ndarray] = []
        self.target = self.w.qpos.copy()
        self.phase = 0
        self.roll = None

    def grip(self, value: float) -> float:
        return float(self.w.deg_to_ctrl([0, 0, 0, 0, 0, value])[GRIPPER])

    def ik(self, pos, base: np.ndarray, point_down: bool = True) -> np.ndarray:
        q, err = self.w.ik(np.asarray(pos, dtype=np.float64), q_init=base, point_down=point_down)
        if err > 0.005 and point_down:
            q, err = self.w.ik(np.asarray(pos, dtype=np.float64), q_init=base, point_down=False)
        if err > 0.008:
            raise RuntimeError(f"IK failed for {np.round(pos, 3)}: {err:.4f} m")
        q[GRIPPER] = base[GRIPPER]
        return q

    def segment(self, goal: np.ndarray, min_steps: int = 8, speed: float = 1.0) -> None:
        start = self.target.copy()
        delta = np.rad2deg(np.abs(goal[ARM] - start[ARM])).max()
        # cosine profile peaks at pi/2 x the mean speed
        n = max(min_steps, int(np.ceil(delta * np.pi / 2 / (PEAK_DEG_PER_STEP * speed))))
        self.plan = [start + (goal - start) * s for s in smooth(n)]

    def hold(self, steps: int) -> None:
        self.plan = [self.target.copy()] * steps

    def next_phase(self) -> bool:
        """Plan the next segment from the current state. Returns False when the script is done."""
        w, cube = self.w, self.w.cube_pos()
        box = w.data.xpos[w.box_body].copy()
        q = self.target.copy()
        self.phase += 1
        if self.phase == 1:      # open the jaws while moving over the cube
            # Solve the grasp pose first, then the hover pose from it: IK started from a start pose
            # can settle on a tilted gripper over the cube, and tilted jaws sweep into the cube.
            q_grasp = self.ik((cube[0], cube[1], GRASP_Z), q)
            goal = self.ik((cube[0], cube[1], cube[2] + HOVER_DZ), q_grasp)
            yaw = 2 * np.arctan2(w.data.qpos[w.cube_qadr + 6], w.data.qpos[w.cube_qadr + 3])
            self.roll = jaw_roll(w, goal, cube_yaw=float(yaw))
            goal[4] = self.roll
            goal[GRIPPER] = self.grip(GRIPPER_OPEN)
            self.segment(goal, min_steps=30, speed=1.0)
        elif self.phase == 2:    # descend, closing over the last steps (no open-jaw pause)
            # A pause at the bottom with the jaws open, then closing, gives the same observation two
            # labels ("stay", "close"); BC averaged them and stalled there. People start squeezing
            # while still arriving, so the jaws start closing CLOSE_OVERLAP steps before the bottom.
            goal = self.ik((cube[0], cube[1], GRASP_Z), q)
            goal[4], goal[GRIPPER] = self.roll, q[GRIPPER]
            self.segment(goal, min_steps=25, speed=0.6)
            arm = self.plan + [goal] * 20
            k = len(self.plan) - CLOSE_OVERLAP
            g0, g1 = q[GRIPPER], self.grip(GRIPPER_CLOSED)
            grip = [g0] * k + [g0 + (g1 - g0) * s for s in smooth(20)]
            grip += [g1] * (len(arm) - len(grip))
            self.plan = [np.concatenate([a[:GRIPPER], [g]]) for a, g in zip(arm, grip)][: k + 20 + 6]
        elif self.phase == 3:    # (closing happens at the end of the descent)
            self.plan = []
            return True
        elif self.phase == 4:    # lift
            goal = self.ik((cube[0], cube[1], CARRY_Z), q)
            goal[4], goal[GRIPPER] = self.roll, q[GRIPPER]
            self.segment(goal, min_steps=25, speed=0.7)
        elif self.phase == 5:    # carry over the bin
            goal = self.ik((box[0], box[1], CARRY_Z), q, point_down=False)
            goal[GRIPPER] = q[GRIPPER]
            self.segment(goal, min_steps=30, speed=0.8)
        elif self.phase == 6:    # lower, opening over the last steps (same reason as the grasp)
            goal = self.ik((box[0], box[1], RELEASE_Z), q, point_down=False)
            goal[GRIPPER] = q[GRIPPER]
            self.segment(goal, min_steps=15, speed=0.6)
            arm = self.plan + [goal] * 12
            k = len(self.plan) - CLOSE_OVERLAP
            g0, g1 = q[GRIPPER], self.grip(GRIPPER_OPEN)
            grip = [g0] * k + [g0 + (g1 - g0) * s for s in smooth(12)]
            grip += [g1] * (len(arm) - len(grip))
            self.plan = [np.concatenate([a[:GRIPPER], [g]]) for a, g in zip(arm, grip)][: k + 12]
        elif self.phase == 7:    # (opening happens at the end of the lowering)
            self.plan = []
            return True
        elif self.phase == 8:    # retreat upward and wait for the env's success check
            goal = self.ik((box[0], box[1], CARRY_Z + 0.03), q, point_down=False)
            goal[GRIPPER] = q[GRIPPER]
            self.segment(goal, min_steps=15, speed=0.8)
            self.plan += [goal] * 40
        else:
            return False
        return True

    def step(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(clean target = label, executed target) for this step, or None when finished."""
        while not self.plan:
            if not self.next_phase():
                return None
        self.target = self.plan.pop(0)
        executed = self.target.copy()
        if self.noise:
            # Ornstein-Uhlenbeck-like drift: correlated over ~10 steps, std about noise_deg. Quieter
            # while descending and closing (a person slows down for the grasp too): full noise there
            # makes most grasps miss, and only successful episodes are kept.
            sigma = self.noise * (0.3 if self.phase in (2, 3) else 1.0)
            self.offset[:5] = 0.9 * self.offset[:5] + np.sqrt(1 - 0.81) * self.rng.normal(0, sigma, 5)
            executed[:5] += self.offset[:5]
        return self.target, np.clip(executed, self.w.lo, self.w.hi)


def to_action(world, q: np.ndarray) -> np.ndarray:
    return (2 * (q - world.lo) / (world.hi - world.lo) - 1).astype(np.float32)


def run_episode(env, starts, rng, args, record: bool, frames: list | None = None):
    q0 = starts[rng.integers(len(starts))].copy()
    q0[ARM] += np.deg2rad(rng.uniform(-args.start_jitter, args.start_jitter, 5))
    obs, _ = env.reset(seed=int(rng.integers(2**31)), options={"joints": q0})
    expert = Expert(env, rng, args.noise_deg)
    rec = {k: [] for k in ("states", "images", "wrist_images", "actions")}
    info = {"is_success": False, "cube_lifted": False}
    lifted = False
    for t in range(args.max_steps):
        out = expert.step()
        if out is None:
            break
        label, executed = out
        if record:
            rec["states"].append(obs["state"])
            rec["images"].append(obs["front"])
            rec["wrist_images"].append(obs["wrist"])
            rec["actions"].append(to_action(env.world, label))
        if frames is not None:
            frames.append(np.concatenate([env.world.render(env.camera, 240, 320),
                                          env.world.render(env.WRIST_CAMERA, 240, 320)], 1))
        obs, _, term, trunc, info = env.step(to_action(env.world, executed))
        lifted |= info["cube_lifted"]
        if term:
            break
    return rec, bool(info["is_success"]), lifted, t + 1


def main() -> None:
    import train_bc
    import train_ppo

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="dataset name: outputs/bc_data/<name>.npz")
    parser.add_argument("--episodes", type=int, default=200, help="successful episodes to keep")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--noise-deg", type=float, default=0.0, help="DART arm-target noise std (deg)")
    parser.add_argument("--start-jitter", type=float, default=3.0, help="+- deg on the start poses")
    parser.add_argument("--cube-xy", type=float, default=0.015, help="+- m around the measured cube spot")
    parser.add_argument("--cube-yaw", type=float, default=15.0, help="+- deg cube yaw")
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="save the first episode as an mp4")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    env = So101PickPlaceEnv(action_mode="absolute", image=True, image_size=(96, 128), history=1, wrist=True,
                            include_targets=False, include_velocities=True, episode_steps=args.max_steps,
                            cube_xy_range=(args.cube_xy, args.cube_xy), cube_yaw_range=np.deg2rad(args.cube_yaw))
    # Start poses: the teleoperated demos' first arm poses (old dataset; arm poses only).
    starts = np.asarray(train_ppo.start_joints(train_ppo.demo_starts(), env.world, set(range(50))))
    rng = np.random.default_rng(args.seed)
    data = {k: [] for k in ("states", "images", "wrist_images", "actions", "episode", "frame")}
    kept, tried, failures, lengths = 0, 0, 0, []
    t0 = time.time()
    while kept < args.episodes:
        tried += 1
        frames = [] if (args.video and kept == 0) else None
        try:
            rec, success, lifted, n = run_episode(env, starts, rng, args, record=True, frames=frames)
        except RuntimeError as e:  # IK failure: skip the layout
            print(f"  attempt {tried}: {e}", flush=True)
            failures += 1
            continue
        if frames:
            import imageio.v2 as imageio
            path = OUT / f"{args.name}_episode0.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(path, frames, fps=30)
            print(f"video -> {path}")
        if not success:
            failures += 1
            print(f"  attempt {tried}: failed (lifted {lifted}, {n} steps)", flush=True)
            continue
        for k in ("states", "images", "wrist_images", "actions"):
            data[k].extend(rec[k])
        data["episode"].extend([kept] * len(rec["actions"]))
        data["frame"].extend(range(len(rec["actions"])))
        lengths.append(len(rec["actions"]))
        kept += 1
        if kept % 10 == 0 or kept == args.episodes:
            print(f"{kept}/{args.episodes} episodes ({tried} tried, {failures} failed), "
                  f"{time.time() - t0:.0f} s", flush=True)
    env.close()

    n_val = max(1, int(round(args.val_fraction * kept)))
    val = list(range(kept - n_val, kept))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.name}.npz",
             states=np.asarray(data["states"], np.float32), images=np.asarray(data["images"], np.uint8),
             wrist_images=np.asarray(data["wrist_images"], np.uint8),
             actions=np.asarray(data["actions"], np.float32),
             episode=np.asarray(data["episode"], np.int16), frame=np.asarray(data["frame"], np.int32))
    info = {"source": "make_scripted_dataset.py", "episodes": kept, "attempts": tried, "failed": failures,
            "frames": len(data["actions"]), "mean_length": float(np.mean(lengths)),
            "val_episodes": val, "train_episodes": list(range(kept - n_val)),
            "layout": {"cube_xy_m": args.cube_xy, "cube_yaw_deg": args.cube_yaw},
            "noise_deg": args.noise_deg, "start_jitter_deg": args.start_jitter, "seed": args.seed,
            "action_mode": "absolute", "image_size": [96, 128], "scene": "mounts/config.json (measured)",
            "seconds": round(time.time() - t0, 1)}
    (OUT / f"{args.name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k not in ("val_episodes", "train_episodes")}, indent=2))
    _ = train_bc  # imported so the module's dataset paths stay importable from here


if __name__ == "__main__":
    main()
