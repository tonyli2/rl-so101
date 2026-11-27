"""Run a policy bundle on the real SO-101 follower with the overhead and wrist cameras.

Observation and action handling follow the bundle's bc_config.json "hardware" block:
30 Hz; RGB images undistorted with each camera's measured K/D (same output K) and resized to
128x96 with INTER_AREA; state = 6 joint positions normalized to [-1, 1] over the actuator
ranges; actions mapped by deploy.Policy.mapper (resync).

    # read-only check: no torque, prints joints/actions/targets, saves the policy's images
    python run_real.py outputs/deploy/pilot2_randomized_s1_1300k --overhead 0 --wrist 1 --dry-run

    # live: moves slowly to HOME, runs the policy, moves slowly back to the start pose
    python run_real.py outputs/deploy/pilot2_randomized_s1_1300k --overhead 0 --wrist 1 --seconds 30 --show

Ctrl-C during a live run stops the policy and returns the arm to where it started before torque
is released. Keep a hand on the power switch for the first runs.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from deploy import load_policy
from envs import HOME_DEG, JOINTS, So101World, prepare_mounted

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / "lerobot-tabletop" / "calibration"
INTRINSICS_DIR = ROOT / "lerobot-tabletop" / "mounts" / "calibration"
NATIVE = (640, 480)  # width, height the intrinsics were measured at
POLICY_SIZE = (128, 96)  # width, height
FPS = 30
# Sim joint angle minus real LeRobot reading, per joint (deg; gripper unused). Measured
# 2026-11-26 with calib_poses.py --set roll: over 9 wrist-roll poses, the sim matches the real
# overhead silhouette at sim_roll = real_roll - 73 deg (4.8 deg RMS; the opposite sign fails).
# Refined 2026-11-26 from two capture sessions (sim2real_capture.py: 14 varied poses and 14
# wrist-on-cube poses), fitting the overhead arm silhouette and the wrist camera's view of the
# cube and bin (placed where the corrected overhead camera located them). Fitting either session
# and testing on the other raised the wrist cube IoU from 0.13-0.23 to 0.72-0.84; pan (+3),
# wrist flex (0) and roll (-80.5) agreed across sessions; lift and elbow trade off (their sum is
# what is well determined). Observations are converted to the sim convention and commands back,
# so the policy sees the joint angles it was trained with.
SIM_MINUS_REAL_DEG = (3.0, -1.5, -9.5, 0.0, -80.5, 0.0)
REAL_ROLL_LIMIT = 175.0  # wrist_roll is calibrated over one full turn (+-180 deg)


class Camera:
    """Background reader that keeps only the newest frame, so the policy never sees a stale queue."""

    def __init__(self, index: int, name: str):
        self.name = name
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"{name} camera (index {index}) did not open; check macOS camera permission")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE[1])
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        cal = json.loads((INTRINSICS_DIR / f"{name}_intrinsics.json").read_text())
        if (cal["width"], cal["height"]) != NATIVE:
            raise ValueError(f"{name} intrinsics measured at {cal['width']}x{cal['height']}, expected {NATIVE}")
        self.K = np.asarray(cal["camera_matrix"], dtype=np.float64)
        self.D = np.asarray(cal["distortion"], dtype=np.float64)
        self.frame = None
        self.stamp = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self.frame is None:
            if time.time() - t0 > 5:
                raise RuntimeError(f"{name} camera gave no frames in 5 s")
            time.sleep(0.01)
        h, w = self.frame.shape[:2]
        if (w, h) != NATIVE:
            raise RuntimeError(f"{name} camera delivers {w}x{h}, intrinsics need {NATIVE[0]}x{NATIVE[1]}")

    def _run(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self.frame, self.stamp = f, time.time()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """(raw BGR frame, policy image: RGB, undistorted to the same K, 128x96 INTER_AREA)."""
        with self._lock:
            raw = self.frame.copy()
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        und = cv2.undistort(rgb, self.K, self.D, None, self.K)
        return raw, cv2.resize(und, POLICY_SIZE, interpolation=cv2.INTER_AREA)

    def close(self):
        self._stop = True
        self._thread.join(timeout=1)
        self.cap.release()


def make_robot(port: str, max_step: float):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(port=port, id="follower", calibration_dir=CALIBRATION_DIR,
                                max_relative_target=max_step)
    return SOFollower(cfg)


def read_deg(robot) -> np.ndarray:
    pos = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    return np.array([pos[j] for j in JOINTS], dtype=np.float64)


def send_deg(robot, deg) -> None:
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, deg)})


def move_to(robot, goal_deg, seconds: float) -> None:
    """Straight-line joint move at 30 Hz; max_relative_target still caps every step."""
    start = read_deg(robot)
    n = max(1, int(seconds * FPS))
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        send_deg(robot, start + (np.asarray(goal_deg) - start) * i / n)
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--overhead", type=int, required=True, help="OpenCV index of the overhead camera")
    parser.add_argument("--wrist", type=int, required=True, help="OpenCV index of the wrist camera")
    parser.add_argument("--seconds", type=float, default=30.0, help="policy run length")
    parser.add_argument("--max-step", type=float, default=10.0,
                        help="LeRobot max_relative_target: per-step cap (deg; gripper 0-100 units)")
    parser.add_argument("--move-seconds", type=float, default=4.0, help="duration of the home/return moves")
    parser.add_argument("--dry-run", action="store_true", help="never enable torque; compute and print only")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/real_runs"))
    parser.add_argument("--sim-offset", default=",".join(f"{v:g}" for v in SIM_MINUS_REAL_DEG),
                        help="sim angle minus real reading per joint, deg (pan,lift,elbow,flex,roll,gripper)")
    parser.add_argument("--gripper-torque", type=int, choices=range(51, 101), metavar="PCT",
                        help="raise the gripper torque limit from LeRobot's 50%% to PCT%% for this run (protection "
                             "current 310 like the arm joints, overload torque 50%%); restored to LeRobot's values after")
    parser.add_argument("--show", action="store_true",
                        help="live window: raw feeds (top) and the policy's 128x96 inputs (bottom); q stops")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    world = So101World()
    lo, hi = world.lo, world.hi
    policy = load_policy(args.bundle, device="cpu")
    if policy.action_history:
        raise NotImplementedError("bundle uses action history; this runner does not keep that buffer")
    if int(policy.obs.get("history", 1)) != 1 or policy.obs.get("velocities"):
        raise NotImplementedError("runner supports history=1 and no velocities only")
    mapper = policy.mapper(lo, hi)

    run_dir = args.log_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cams = {"front": Camera(args.overhead, "overhead"), "wrist": Camera(args.wrist, "wrist")}
    for key, cam in cams.items():
        raw, img = cam.read()
        cv2.imwrite(str(run_dir / f"{cam.name}_raw.jpg"), raw)
        cv2.imwrite(str(run_dir / f"{cam.name}_policy.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"camera snapshots in {run_dir}", flush=True)

    robot = make_robot(args.port, args.max_step)
    if args.dry_run:
        robot.bus.connect()  # no configure(): torque stays as it is (off)
    else:
        robot.connect(calibrate=False)
    if not robot.bus.is_calibrated:
        raise RuntimeError("motor calibration does not match lerobot-tabletop/calibration/follower.json")
    GRIPPER_LEROBOT = {"Max_Torque_Limit": 500, "Torque_Limit": 500, "Protection_Current": 250, "Overload_Torque": 25}

    def set_gripper(regs: dict) -> None:
        with robot.bus.torque_disabled(["gripper"]):
            for reg, val in regs.items():
                robot.bus.write(reg, "gripper", val)
        print("gripper limits:", {reg: robot.bus.read(reg, "gripper", normalize=False) for reg in regs}, flush=True)

    if args.gripper_torque and not args.dry_run:
        t = args.gripper_torque * 10
        set_gripper({"Max_Torque_Limit": t, "Torque_Limit": t, "Protection_Current": 310, "Overload_Torque": 50})

    sim_off = np.array([float(v) for v in args.sim_offset.split(",")], dtype=np.float64)
    if sim_off.shape != (6,):
        raise ValueError("--sim-offset needs 6 comma-separated values")
    print("sim minus real offsets (deg):", sim_off, flush=True)

    def to_real(sim_deg) -> np.ndarray:
        real = np.asarray(sim_deg, dtype=np.float64) - sim_off
        real[4] = np.clip(real[4], -REAL_ROLL_LIMIT, REAL_ROLL_LIMIT)
        return real

    def observe():
        """Joint readings in the sim convention (what the policy was trained on)."""
        deg = read_deg(robot) + sim_off
        q = world.deg_to_ctrl(deg)
        state = (2 * (q - lo) / (hi - lo) - 1).astype(np.float32)
        obs = {"state": state}
        for key, cam in cams.items():
            raws[key], obs[key] = cam.read()
        return deg, q, obs

    raws: dict[str, np.ndarray] = {}
    tile = (320, 240)

    def show(obs, t: float, deg) -> None:
        """Draw the live window; 'q' in the window stops the run like Ctrl-C."""
        top = np.hstack([cv2.resize(raws[k], tile) for k in ("front", "wrist")])
        bottom = np.hstack([cv2.resize(cv2.cvtColor(obs[k], cv2.COLOR_RGB2BGR), tile,
                                       interpolation=cv2.INTER_NEAREST) for k in ("front", "wrist")])
        frame = np.vstack([top, bottom])
        for (x, y), label in [((8, 20), "overhead raw"), ((328, 20), "wrist raw"),
                              ((8, 260), "overhead -> policy"), ((328, 260), "wrist -> policy")]:
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"t={t:4.1f}s  gripper {deg[5]:.0f}", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("so101 policy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    rest_deg = read_deg(robot)
    print("start pose (deg):", np.round(rest_deg, 1), flush=True)
    log = {k: [] for k in ("t", "deg", "action", "target_deg", "dt", "front", "wrist")}
    try:
        if not args.dry_run:
            print(f"moving to HOME {HOME_DEG} (sim convention) over {args.move_seconds} s", flush=True)
            move_to(robot, to_real(HOME_DEG), args.move_seconds)
        deg, q, obs = observe()
        mapper.reset(q)
        print(f"running policy for {args.seconds} s{' (dry run)' if args.dry_run else ''}; Ctrl-C to stop",
              flush=True)
        t_start = time.perf_counter()
        step = 0
        while time.perf_counter() - t_start < args.seconds:
            t0 = time.perf_counter()
            if step:
                deg, q, obs = observe()
            action, _ = policy.model.predict(obs, deterministic=True)
            target = mapper(action, q)
            target_deg = world.ctrl_to_deg(target)
            if not args.dry_run:
                send_deg(robot, to_real(target_deg))
            dt = time.perf_counter() - t0
            log["t"].append(t0 - t_start)
            log["deg"].append(deg)
            log["action"].append(np.asarray(action))
            log["target_deg"].append(target_deg)
            log["dt"].append(dt)
            log["front"].append(obs["front"])
            log["wrist"].append(obs["wrist"])
            if args.show:
                show(obs, t0 - t_start, deg)
            if step % FPS == 0:
                print(f"t={t0 - t_start:5.1f}s  joints {np.round(deg, 1)}  target {np.round(target_deg, 1)}  "
                      f"step {dt * 1e3:.0f} ms", flush=True)
            step += 1
            # fixed-rate schedule: the next step (sensing included) starts on the 30 Hz grid
            time.sleep(max(0.0, t_start + step / FPS - time.perf_counter()))
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        try:
            if not args.dry_run:
                print("returning to start pose", flush=True)
                move_to(robot, rest_deg, args.move_seconds)
        finally:
            if args.gripper_torque and not args.dry_run:
                try:
                    print("gripper temperature after the run:",
                          robot.bus.read("Present_Temperature", "gripper", normalize=False), "C", flush=True)
                    set_gripper(GRIPPER_LEROBOT)
                except Exception as e:  # never block the disconnect below
                    print("could not restore gripper limits:", e, flush=True)
            if args.dry_run:
                robot.bus.disconnect(disable_torque=False)
            else:
                robot.disconnect()
            for cam in cams.values():
                cam.close()
            if args.show:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            if log["t"]:
                np.savez(run_dir / "log.npz", **{k: np.asarray(v) for k, v in log.items()})
                dts = np.asarray(log["dt"]) * 1e3
                print(f"{len(dts)} steps; compute+send {np.median(dts):.0f} ms median, "
                      f"{dts.max():.0f} ms max; log {run_dir / 'log.npz'}", flush=True)


if __name__ == "__main__":
    main()
