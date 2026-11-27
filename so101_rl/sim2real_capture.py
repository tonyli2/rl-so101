"""Sim-to-real test battery capture on the real follower (slow moves, torque on).

Phases (all commands in the sim joint convention, converted with run_real.SIM_MINUS_REAL_DEG):
  A  varied static poses: overhead + wrist frames and joint readings (camera/joint/layout fits)
  B  gripper sweep at a fixed pose: wrist frames per gripper value (gripper mapping)
  C  per-joint step responses at 30 Hz: commanded vs measured (latency, servo response)

    python sim2real_capture.py --overhead 0 --wrist 1
    python sim2real_capture.py --overhead 0 --wrist 1 --phases A --pose-file wrist_poses.json  # wrist-on-cube views

Every pose is checked in sim first: the wrist/gripper geoms must stay >= 4 cm above the table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from envs import HOME_DEG, So101World, prepare_mounted
from run_real import FPS, SIM_MINUS_REAL_DEG, Camera, make_robot, move_to, read_deg, send_deg

POSES_A = [
    (0, -30, 30, 60, -85, 30), (15, -20, 30, 60, -85, 30), (30, -20, 30, 60, -85, 30),
    (-15, -20, 30, 60, -85, 30), (0, -10, 20, 50, -85, 30), (25, -10, 20, 50, -85, 30),
    (0, -40, 60, 20, -85, 30), (20, -45, 60, 20, 0, 30), (-20, 0, 0, 30, 0, 30),
    (40, 10, 0, 40, -45, 30), (0, 10, -10, 60, -85, 30), (35, 20, -10, 45, -85, 30),
    (10, -10, 10, 70, -120, 30), (-30, -30, 40, 40, -60, 30),
]
POSE_B = (-20, 0, 0, 30, 0, 30)
GRIPPER_B = (0, 10, 25, 40, 60, 80, 100, 30)
POSE_C = (10, -20, 30, 50, -85, 30)
STEP_C = (8, 8, 8, 8, 8, 15)  # deg per arm joint; gripper in 0-100 units
MIN_CLEARANCE = 0.04


def clearance(world: So101World, deg) -> float:
    m = world.model
    world.reset(world.deg_to_ctrl(deg))
    lo, hi = m.body("wrist").id, m.body("camera_mount").id
    geoms = [g for g in range(m.ngeom) if lo <= m.geom_bodyid[g] <= hi]
    return float(min(world.data.geom_xpos[g][2] - m.geom_rbound[g] for g in geoms))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--overhead", type=int, required=True)
    parser.add_argument("--wrist", type=int, required=True)
    parser.add_argument("--move-seconds", type=float, default=5.0)
    parser.add_argument("--phases", default="ABC")
    parser.add_argument("--out", type=Path, default=Path("outputs/sim2real") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--pose-file", type=Path, help="JSON list of sim-convention poses replacing phase A's")
    args = parser.parse_args()

    global POSES_A
    if args.pose_file:
        POSES_A = [tuple(p) for p in json.loads(args.pose_file.read_text())]
    prepare_mounted(rebuild=True)
    world = So101World()
    off = np.asarray(SIM_MINUS_REAL_DEG, dtype=np.float64)
    to_real = lambda d: np.asarray(d, dtype=np.float64) - off  # noqa: E731
    for p in POSES_A + [POSE_B, POSE_C]:
        c = clearance(world, p)
        if c < MIN_CLEARANCE:
            raise SystemExit(f"pose {p} clearance {c:.3f} m < {MIN_CLEARANCE}; refusing")
    args.out.mkdir(parents=True, exist_ok=True)
    cams = {"overhead": Camera(args.overhead, "overhead"), "wrist": Camera(args.wrist, "wrist")}
    robot = make_robot(args.port, 10.0)
    robot.connect(calibrate=False)
    if not robot.bus.is_calibrated:
        raise RuntimeError("motor calibration does not match follower.json")

    def capture(tag: str) -> dict:
        time.sleep(1.2)
        real = read_deg(robot)
        for name, cam in cams.items():
            raw, _ = cam.read()
            cv2.imwrite(str(args.out / f"{tag}_{name}.png"), cv2.undistort(raw, cam.K, cam.D, None, cam.K))
        return {"tag": tag, "measured_real": real.tolist(), "measured_sim": (real + off).tolist()}

    rest = read_deg(robot)
    log: dict = {"sim_minus_real": off.tolist(), "rest_real": rest.tolist(), "A": [], "B": [], "C": []}
    try:
        move_to(robot, to_real(HOME_DEG), args.move_seconds)
        if "A" in args.phases:
            for i, p in enumerate(POSES_A):
                print(f"A{i:02d} -> {p}", flush=True)
                move_to(robot, to_real(p), args.move_seconds)
                log["A"].append({**capture(f"A{i:02d}"), "commanded_sim": list(p)})
            move_to(robot, to_real(HOME_DEG), args.move_seconds)
        if "B" in args.phases:
            move_to(robot, to_real(POSE_B), args.move_seconds)
            for g in GRIPPER_B:
                print(f"B gripper -> {g}", flush=True)
                cmd = np.r_[POSE_B[:5], g]
                move_to(robot, to_real(cmd), 1.5)
                log["B"].append({**capture(f"B_g{g:03d}"), "commanded_sim": cmd.tolist()})
            move_to(robot, to_real(HOME_DEG), args.move_seconds)
        if "C" in args.phases:
            move_to(robot, to_real(POSE_C), args.move_seconds)
            time.sleep(1.0)
            for j in range(6):
                print(f"C step joint {j}", flush=True)
                base = np.asarray(POSE_C, dtype=np.float64)
                seq = [base] * 15 + [base + np.eye(6)[j] * STEP_C[j]] * 45 + [base] * 45
                rec = {"joint": j, "t": [], "cmd_sim": [], "meas_sim": []}
                t0 = time.perf_counter()
                for k, cmd in enumerate(seq):
                    meas = read_deg(robot) + off
                    send_deg(robot, to_real(cmd))
                    rec["t"].append(time.perf_counter() - t0)
                    rec["cmd_sim"].append(cmd.tolist())
                    rec["meas_sim"].append(meas.tolist())
                    time.sleep(max(0.0, t0 + (k + 1) / FPS - time.perf_counter()))
                log["C"].append(rec)
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        try:
            print("returning to rest", flush=True)
            move_to(robot, to_real(HOME_DEG), args.move_seconds)
            move_to(robot, rest, args.move_seconds)
        finally:
            robot.disconnect()
            for cam in cams.values():
                cam.close()
            (args.out / "capture.json").write_text(json.dumps(log))
            print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
