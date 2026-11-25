"""Native LeRobot training with local task registration and quarantine protection."""

import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

from learning.tabletop import TabletopConfig  # noqa: E402,F401
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.scripts.lerobot_train import train  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402


@parser.wrap()
def guarded_train(cfg: TrainPipelineConfig):
    """Check the resolved root, including configs and resumed configurations."""
    if cfg.dataset.root:
        root = Path(cfg.dataset.root).expanduser().resolve()
        if (root / "QUARANTINED.json").exists():
            raise SystemExit(
                "Refusing to train on quarantined collision demonstrations"
            )
        if root.parent.name == "dataset-backups":
            raise SystemExit(
                "Historical dataset backups are for reproduction, not training"
            )
    return train(cfg)


if __name__ == "__main__":
    register_third_party_plugins()
    guarded_train()
