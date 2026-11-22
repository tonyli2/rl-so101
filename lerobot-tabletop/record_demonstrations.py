"""Record local, reusable leader-to-simulation demonstrations with LeRobot."""

import argparse
import contextlib
import logging
import os
import queue
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local dataset directory (new unless --resume)",
    )
    parser.add_argument(
        "--repo-id",
        default="local/red_cube_demonstrations",
        help="Local dataset identifier; nothing is uploaded",
    )
    parser.add_argument("--leader-port", help="Physical leader port, e.g. /dev/ttyACM0")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=Path(__file__).parent / "calibration/leader.json",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Accepted episodes before exit"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1800,
        help="Maximum recorded control steps before settling/review",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Record a held-target tail after finishing",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headless", action="store_true", help="Offscreen cameras, no MuJoCo window"
    )
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Four-view collection window (desktop default); recorded cameras unchanged",
    )
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Use the original orbitable viewer and terminal controls",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=30,
        help="Display target refresh rate (default: 30 Hz; recording FPS unchanged)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to a validated existing local dataset",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Automated test inputs; episodes explicitly NOT labeled demonstrations",
    )
    parser.add_argument(
        "--discard-first",
        action="store_true",
        help="Synthetic test: discard an initial attempt",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace synthetic tests at the requested FPS",
    )
    args = parser.parse_args()
    if not args.synthetic and not args.leader_port:
        parser.error("--leader-port is required unless --synthetic is selected")
    if args.synthetic and args.leader_port:
        parser.error("Synthetic mode never connects hardware; omit --leader-port")
    if args.multi_view and args.headless:
        parser.error("--multi-view requires a desktop window; omit --headless")
    if args.multi_view and args.single_view:
        parser.error("Choose either --multi-view or --single-view")
    args.multi_view = not args.headless and not args.single_view
    if not 0 < args.preview_fps <= 60:
        parser.error("--preview-fps must be greater than zero and no more than 60")
    if (
        min(args.episodes, args.steps, args.fps, args.width, args.height) <= 0
        or args.settle_seconds < 0
    ):
        parser.error(
            "Counts/dimensions must be positive and settling duration nonnegative"
        )
    if args.width % 2 or args.height % 2:
        parser.error("H264 capture requires even image dimensions")
    os.environ["MUJOCO_GL"] = "egl" if args.headless else "glfw"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import env as tabletop
    import numpy as np
    from lerobot.utils.cycle_timer import CycleTimer
    from lerobot.utils.keyboard_input import create_key_listener
    from mounts.environment import make_single_env
    from recording.schema import JOINTS
    from recording.session import RecordingSession, open_leader

    simulation = make_single_env(
        fps=args.fps,
        observation_height=args.height,
        observation_width=args.width,
        terminate_on_success=False,
        episode_length=args.steps + int(args.settle_seconds * args.fps) + 2,
    )
    session = leader = viewer = listener = preview = None
    commands = queue.SimpleQueue()
    saved = 0
    timer = CycleTimer(args.fps)
    home = np.asarray(tabletop.HOME, dtype=np.float32)
    last_action = home.copy()

    def show_view():
        if viewer is not None:
            # State-only sync avoids importing viewer-side model edits. Mouse
            # perturbation forces are explicitly cleared before the next step.
            viewer.sync(state_only=True)
            with viewer.lock():
                session.world.data.qfrc_applied[:] = 0
                session.world.data.xfrc_applied[:] = 0
        if preview is not None:
            preview.update(
                session.world,
                lock=viewer.lock() if viewer is not None else None,
                status=(
                    "REVIEW — k: keep / d: discard"
                    if session.review
                    else "RECORDING — f: finish"
                    if session.active
                    else "PREVIEW — s: start"
                ),
            )

    def reset_preview():
        with viewer.lock() if viewer is not None else contextlib.nullcontext():
            session.reset()

    def tick(action, *, settling=False):
        timer.tick()
        with timer.section("leader_read"):
            target = action() if callable(action) else action
            read_wall = time.monotonic()
        with timer.section("simulation_and_record"):
            with viewer.lock() if viewer is not None else contextlib.nullcontext():
                session.step(target, leader_read_wall=read_wall, settling=settling)
        with timer.section("viewer"):
            show_view()
        if not args.synthetic or args.realtime:
            timer.wait()
        return target

    def read_leader():
        readings = leader.get_action()
        return np.asarray(
            [readings[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32
        )

    def finish():
        for _ in range(round(args.settle_seconds * args.fps)):
            if viewer is not None and not viewer.is_running():
                break
            tick(last_action, settling=True)
        session.finish()
        print(
            "Review: k=keep successful demo, d=discard/reset, q=quit. Simulation paused.",
            flush=True,
        )

    try:
        if not args.synthetic:
            leader = open_leader(args.leader_port, args.calibration_file)
        session = RecordingSession(
            simulation,
            args.root,
            args.repo_id,
            seed=args.seed,
            resume=args.resume,
            source="synthetic" if args.synthetic else "real_leader",
        )
        if args.multi_view:
            from recording.preview import MultiViewPreview

            preview = MultiViewPreview(
                session.world.model, fps=args.preview_fps, on_command=commands.put
            )
        elif not args.headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                session.world.model,
                session.world.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            viewer.cam.lookat[:] = [0.08, 0.04, 0.24]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                1.25,
                180,
                -20,
            )
        if args.synthetic:
            attempts = args.episodes + int(args.discard_first)
            for attempt in range(attempts):
                reset_preview()
                with viewer.lock() if viewer is not None else contextlib.nullcontext():
                    session.begin()
                for index in range(args.steps):
                    action = home.copy()
                    if index == 0:
                        action[0] = (
                            999  # Intentionally exercise applied-action clipping.
                        )
                    tick(action)
                session.finish()
                if args.discard_first and attempt == 0:
                    session.discard()
                    print("Discarded synthetic attempt", flush=True)
                else:
                    identifier = session.save(
                        "synthetic", annotation="automated_test_not_a_demonstration"
                    )
                    saved += 1
                    print(f"Saved synthetic episode {saved}: {identifier}", flush=True)
            return

        if preview is None:
            listener = create_key_listener(
                commands.put,
                controls_help="s=start, f=finish, k=keep, d=discard, r=reset, q=quit",
            )
        if listener is None and preview is None:
            raise RuntimeError(
                "Recording requires an interactive terminal or usable keyboard listener"
            )
        print(
            "Preview: s=start, f=finish+settle, k=keep successful demo, d=discard, r=reset, q=quit.",
            flush=True,
        )
        print(
            "Use the collection window buttons or keys."
            if preview is not None
            else "Keep this terminal focused for controls.",
            flush=True,
        )
        while saved < args.episodes and (viewer is None or viewer.is_running()):
            try:
                command = commands.get_nowait().lower()
            except queue.Empty:
                command = None
            if command in ("q", "esc"):
                break
            if command == "s" and not session.active and not session.review:
                with viewer.lock() if viewer is not None else contextlib.nullcontext():
                    session.begin()
                timer.restart()
                print("Recording. f=finish; d=discard.", flush=True)
            elif command == "f" and session.active:
                finish()
            elif command == "k" and session.review:
                print("Saving and validating both camera streams...", flush=True)
                identifier = session.save()
                saved += 1
                print(f"Saved successful episode {saved}: {identifier}", flush=True)
                reset_preview()
                timer.restart()
                print("Preview: s=start next episode.", flush=True)
            elif command in ("d", "r"):
                session.discard()
                reset_preview()
                timer.restart()
                print("Reset. Preview: s=start.", flush=True)
            if session.review:
                show_view()
                time.sleep(0.02)
                continue
            last_action = tick(read_leader)
            if session.active and session.length >= args.steps:
                finish()
    except KeyboardInterrupt:
        print("\nStopping; unfinished attempt will be discarded.", flush=True)
    finally:
        # ExitStack runs every cleanup even if an earlier resource fails to close.
        try:
            with contextlib.ExitStack() as cleanup:
                if leader is not None and leader.is_connected:
                    cleanup.callback(leader.disconnect)
                cleanup.callback(simulation.close)
                for resource in (viewer, preview, session):
                    if resource is not None:
                        cleanup.callback(resource.close)
                if listener is not None:
                    cleanup.callback(listener.stop)
        finally:
            timer.log_run_summary()
            print(
                f"Collection closed; {saved} episode(s) saved this run. No data uploaded.",
                flush=True,
            )


if __name__ == "__main__":
    main()
