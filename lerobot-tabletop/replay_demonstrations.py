"""Validate, replay or re-render local demonstrations without connecting hardware."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Source local LeRobot dataset root"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Check archive alignment and decode every video frame",
    )
    mode.add_argument(
        "--replay",
        choices=("states", "commands"),
        help="Restore states or replay applied controls",
    )
    mode.add_argument(
        "--rerender",
        type=Path,
        metavar="NEW_ROOT",
        help="Write a new native dataset using archived states",
    )
    parser.add_argument(
        "--episode",
        help="One episode UUID or numeric dataset episode index; default: all",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use EGL and do not open the replay viewer",
    )
    parser.add_argument(
        "--camera-overrides",
        type=Path,
        help="JSON: named MuJoCo cameras -> local pos/quat/fovy overrides",
    )
    parser.add_argument(
        "--height", type=int, help="New render height, with --rerender and --width"
    )
    parser.add_argument(
        "--width", type=int, help="New render width, with --rerender and --height"
    )
    parser.add_argument(
        "--compare-images",
        action="store_true",
        help="During validation report original-render/video MAE",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed qpos error for command replay",
    )
    args = parser.parse_args()
    if args.camera_overrides and args.rerender is None:
        parser.error("--camera-overrides is only valid with --rerender")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be provided together")
    if args.height is not None and (
        args.rerender is None or args.height <= 0 or args.width <= 0
    ):
        parser.error("Positive --height/--width values are only valid with --rerender")
    if args.tolerance <= 0:
        parser.error("--tolerance must be positive")
    os.environ["MUJOCO_GL"] = "egl" if args.headless else "glfw"

    from recording.replay import replay, rerender, validate

    root = args.root.expanduser().resolve()
    try:
        if args.validate:
            validate(root, args.episode, compare_images=args.compare_images)
        elif args.replay:
            replay(
                root,
                args.episode,
                mode=args.replay,
                headless=args.headless,
                tolerance=args.tolerance,
            )
        else:
            rerender(
                root,
                args.rerender.expanduser().resolve(),
                args.episode,
                args.camera_overrides,
                height=args.height,
                width=args.width,
            )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
