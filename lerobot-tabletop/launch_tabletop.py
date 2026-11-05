"""Launch or check the unmodified upstream LeRobot tabletop environment."""

import argparse
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--leader-port",
    help="Read targets from a real SO-101 leader, e.g. /dev/ttyACM0",
)
parser.add_argument(
    "--calibration-file",
    type=Path,
    default=Path(__file__).parent / "calibration" / "leader.json",
    help="Existing leader calibration JSON (default: calibration/leader.json beside this script)",
)
parser.add_argument(
    "--check",
    action="store_true",
    help="Check physics and save camera images without a window",
)
parser.add_argument(
    "--seconds",
    type=float,
    default=0,
    help="Viewer duration; 0 keeps it open until closed",
)
args = parser.parse_args()
if args.seconds < 0:
    parser.error("--seconds must be nonnegative")
if args.check and args.leader_port:
    parser.error("--check and --leader-port are separate modes")
if args.leader_port:
    args.calibration_file = args.calibration_file.expanduser().resolve()
    if not args.calibration_file.is_file():
        parser.error(f"Existing calibration file not found: {args.calibration_file}")
if args.check:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

import env as tabletop

environments = tabletop.make_env()
vector_env = environments["so101_tabletop"][0]
leader = None
try:
    observation, _ = vector_env.reset(seed=0)
    simulation = vector_env.envs[0]
    # Upstream exposes model/data only through its internal world object.
    world = simulation._ensure_world()
    action = np.asarray([tabletop.HOME], dtype=np.float32)
    if args.leader_port:
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

        leader = SOLeader(SOLeaderTeleopConfig(
            port=args.leader_port,
            id=args.calibration_file.stem,
            calibration_dir=args.calibration_file.parent,
            use_degrees=True,
        ))
        if set(leader.calibration) != set(tabletop.JOINTS):
            raise ValueError("Calibration must contain all six SO-101 leader joints")
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("Leader calibration does not match the motors; stopping without recalibrating")
        print(f"Leader connected on {args.leader_port}. Move it to control the simulated follower.", flush=True)
    print(
        f"MuJoCo {mujoco.__version__}; task: {simulation.task_description}", flush=True
    )
    print(f"Joints: {tabletop.JOINTS}; cameras: {simulation.cameras}", flush=True)
    if args.check:
        for _ in range(60):
            observation, reward, terminated, truncated, info = vector_env.step(action)
            assert np.isfinite(world.data.qpos).all(), "Non-finite physics state"
        assert vector_env.single_observation_space.contains(
            {
                "agent_pos": observation["agent_pos"][0],
                "pixels": {
                    key: value[0] for key, value in observation["pixels"].items()
                },
            }
        )
        output = Path(__file__).parent / "verification"
        output.mkdir(exist_ok=True)
        for name, images in observation["pixels"].items():
            image = images[0]
            assert image.shape == (480, 640, 3) and image.std() > 1
            path = output / f"{name}.png"
            Image.fromarray(image).save(path)
            print(f"Camera OK: {name}, {image.shape}; saved {path}", flush=True)
        print(
            f"PASS: 60 control steps; simulation time {world.data.time:.3f}s; reward {reward}",
            flush=True,
        )
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
            start = time.monotonic()
            while viewer.is_running() and (
                args.seconds <= 0 or time.monotonic() - start < args.seconds
            ):
                tick = time.monotonic()
                if leader is not None:
                    readings = leader.get_action()
                    action = np.asarray(
                        [[readings[f"{joint}.pos"] for joint in tabletop.JOINTS]],
                        dtype=np.float32,
                    )
                    if not np.isfinite(action).all():
                        raise RuntimeError(f"Invalid leader readings: {action}")
                with viewer.lock():
                    _, _, terminated, truncated, _ = vector_env.step(action)
                    if terminated[0] or truncated[0]:
                        vector_env.reset()
                viewer.sync()
                time.sleep(max(0, 1 / simulation.fps - (time.monotonic() - tick)))
        print("Viewer closed normally.", flush=True)
except KeyboardInterrupt:
    print("\nStopping.", flush=True)
finally:
    try:
        vector_env.close()
    finally:
        if leader is not None and leader.is_connected:
            leader.disconnect()
