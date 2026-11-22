"""Launch the stock-mount scene, or the unchanged upstream tabletop."""

import argparse
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=("mounted", "upstream"),
        default="mounted",
        help="Stock black mounts and overhead/wrist views (default), or original upstream scene",
    )
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
            parser.error(
                f"Existing calibration file not found: {args.calibration_file}"
            )
    # Select per process before importing MuJoCo; inherited shell settings must not
    # make desktop viewing and headless checks interfere with one another.
    os.environ["MUJOCO_GL"] = "egl" if args.check else "glfw"

    import env as tabletop
    import mujoco
    import numpy as np
    from PIL import Image

    if args.scene == "mounted":
        from mounts.environment import make_env
    else:
        make_env = tabletop.make_env

    environments = make_env()
    vector_env = environments["so101_tabletop"][0]
    leader = None
    try:
        observation, _ = vector_env.reset(seed=0)
        simulation = vector_env.envs[0]
        # Upstream exposes model/data only through its internal world object.
        world = simulation._ensure_world()
        action = np.asarray([tabletop.HOME], dtype=np.float32)
        if args.leader_port:
            from recording.session import open_leader

            leader = open_leader(args.leader_port, args.calibration_file)
            print(
                f"Leader connected on {args.leader_port}. Move it to control the simulated follower.",
                flush=True,
            )
        print(
            f"MuJoCo {mujoco.__version__}; task: {simulation.task_description}",
            flush=True,
        )
        print(f"Joints: {tabletop.JOINTS}; cameras: {simulation.cameras}", flush=True)
        if args.check:
            for _ in range(60):
                observation, reward, terminated, truncated, info = vector_env.step(
                    action
                )
                assert np.isfinite(world.data.qpos).all(), "Non-finite physics state"
            assert vector_env.single_observation_space.contains(
                {
                    "agent_pos": observation["agent_pos"][0],
                    "pixels": {
                        key: value[0] for key, value in observation["pixels"].items()
                    },
                }
            )
            output = Path(__file__).parent / "verification" / args.scene
            output.mkdir(parents=True, exist_ok=True)
            for name, images in observation["pixels"].items():
                image = images[0]
                assert image.shape == (480, 640, 3) and image.std() > 1
                path = output / f"{name}.png"
                Image.fromarray(image).save(path)
                print(f"Camera OK: {name}, {image.shape}; saved {path}", flush=True)
            overview = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(overview)
            overview.lookat[:] = [0.08, 0.04, 0.24]
            overview.distance = 1.25
            overview.azimuth = 180
            overview.elevation = -20
            world.renderer.update_scene(world.data, camera=overview)
            Image.fromarray(world.renderer.render().copy()).save(
                output / "overview.png"
            )
            print(
                f"PASS: 60 control steps; simulation time {world.data.time:.3f}s; reward {reward}",
                flush=True,
            )
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(world.model, world.data) as viewer:
                if args.scene == "mounted":
                    viewer.cam.lookat[:] = [0.08, 0.04, 0.24]
                    viewer.cam.distance = 1.25
                    viewer.cam.azimuth = 180
                    viewer.cam.elevation = -20
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


if __name__ == "__main__":
    main()
