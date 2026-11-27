"""Joint-zero check: drive the real follower through reference poses and compare with the sim.

LeRobot's degree mode puts 0 deg at the midpoint of each joint's recorded calibration range, while
the sim assumes 0 deg is the MJCF zero. This visits a few poses slowly, holds each, and saves
real overhead/wrist frames beside sim renders at the *measured* joint readings, so any per-joint
offset shows as a mismatch. Commanded vs measured readings (servo droop) are printed too.

    python calib_poses.py --overhead 0 --wrist 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from envs import HOME_DEG, So101World, prepare_mounted
from run_real import Camera, make_robot, move_to, read_deg

POSES = [  # name, degrees (pan, lift, elbow, wrist_flex, wrist_roll, gripper 0-100)
    ("home", HOME_DEG),
    ("zero", (0, 0, 0, 0, 0, 50)),
    ("pan+30", (30, 0, 0, 0, 0, 50)),
    ("zero", (0, 0, 0, 0, 0, 50)),
    ("pan-30", (-30, 0, 0, 0, 0, 50)),
    ("zero", (0, 0, 0, 0, 0, 50)),
    ("flex+45", (0, 0, 0, 45, 0, 50)),
    ("home", HOME_DEG),
]
ROLL_POSES = [("home", HOME_DEG)] + [
    (f"roll{r:+d}", (0, 0, 0, 0, r, 50)) for r in (0, -45, -90, -45, 0, 45, 90, 45, 0)
] + [("home", HOME_DEG)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--overhead", type=int, required=True)
    parser.add_argument("--wrist", type=int, required=True)
    parser.add_argument("--move-seconds", type=float, default=6.0)
    parser.add_argument("--max-step", type=float, default=5.0)
    parser.add_argument("--set", choices=("joints", "roll"), default="joints",
                        help="joints: zero/pan/flex reference poses; roll: wrist-roll sweep at the zero pose")
    parser.add_argument("--out", type=Path, default=Path("outputs/calib_poses") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    world = So101World()
    args.out.mkdir(parents=True, exist_ok=True)
    cams = {"overhead": Camera(args.overhead, "overhead"), "wrist": Camera(args.wrist, "wrist")}
    robot = make_robot(args.port, args.max_step)
    robot.connect(calibrate=False)
    if not robot.bus.is_calibrated:
        raise RuntimeError("motor calibration does not match follower.json")
    rest = read_deg(robot)
    print("rest pose (deg):", np.round(rest, 1), flush=True)
    records = []
    try:
        for k, (name, cmd) in enumerate(POSES if args.set == "joints" else ROLL_POSES):
            print(f"-> {name} {cmd}", flush=True)
            move_to(robot, cmd, args.move_seconds)
            time.sleep(1.5)  # settle, and let the camera buffers catch up
            meas = read_deg(robot)
            world.reset(world.deg_to_ctrl(meas))
            tiles = []
            for cam_name, sim_cam in (("overhead", "overhead"), ("wrist", "wrist_cam")):
                raw, _ = cams[cam_name].read()
                real = cv2.undistort(raw, cams[cam_name].K, cams[cam_name].D, None, cams[cam_name].K)
                sim = cv2.cvtColor(world.render(sim_cam, 480, 640), cv2.COLOR_RGB2BGR)
                blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
                cv2.imwrite(str(args.out / f"{k}_{name}_{cam_name}_real.jpg"), real)
                cv2.imwrite(str(args.out / f"{k}_{name}_{cam_name}_sim.jpg"), sim)
                tiles.append(np.hstack([real, sim, blend]))
            cv2.imwrite(str(args.out / f"{k}_{name}_compare.jpg"), np.vstack(tiles))
            err = meas - np.asarray(cmd, dtype=float)
            print(f"   measured {np.round(meas, 1)}  measured-commanded {np.round(err, 1)}", flush=True)
            records.append({"pose": name, "commanded": list(map(float, cmd)), "measured": meas.tolist()})
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        try:
            print("returning to rest", flush=True)
            move_to(robot, rest, args.move_seconds)
        finally:
            robot.disconnect()
            for cam in cams.values():
                cam.close()
            (args.out / "poses.json").write_text(json.dumps(records, indent=1))
            print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
