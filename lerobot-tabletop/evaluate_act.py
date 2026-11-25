"""Native LeRobot checkpoint evaluation with the local tabletop task registered."""

import os

os.environ["MUJOCO_GL"] = "egl"

from learning.tabletop import TabletopConfig  # noqa: E402,F401
from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
