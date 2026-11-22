"""Shared, policy-independent learning-data contract and native writer setup."""

from pathlib import Path

import numpy as np

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CAMERAS = {"overhead": "overhead", "wrist": "wrist_cam"}


def dataset_features(height=480, width=640):
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        **{
            f"observation.images.{name}": {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            }
            for name in CAMERAS
        },
    }


def learning_frame(observation, applied_action, task):
    return {
        "observation.state": np.asarray(
            observation["agent_pos"], dtype=np.float32
        ).copy(),
        "action": np.asarray(applied_action, dtype=np.float32).copy(),
        **{
            f"observation.images.{name}": np.asarray(
                observation["pixels"][name], dtype=np.uint8
            )
            for name in CAMERAS
        },
        "task": task,
    }


def create_dataset(root, repo_id, fps=30, height=480, width=640, resume=False):
    if min(fps, height, width) <= 0 or height % 2 or width % 2:
        raise ValueError(
            "FPS and image dimensions must be positive; H264 dimensions must be even"
        )
    import av
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    av.logging.set_level(av.logging.ERROR)
    options = dict(
        repo_id=repo_id,
        root=Path(root),
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18, g=2),
        streaming_encoding=True,
        encoder_queue_maxsize=8,
        encoder_threads=2,
    )
    if resume:
        return LeRobotDataset.resume(**options)
    return LeRobotDataset.create(
        **options,
        fps=fps,
        features=dataset_features(height, width),
        robot_type="so101_mujoco",
        use_videos=True,
    )


def check_encoder(dataset):
    """Fail closed if the installed native streaming encoder dropped frames."""
    encoder = dataset.writer._streaming_encoder
    if encoder is not None and any(encoder._dropped_frames.values()):
        raise RuntimeError(
            f"Video encoder dropped frames: {encoder._dropped_frames}; discard this attempt"
        )


def applied_action(world):
    controls = world.data.ctrl[world.actuators].copy()
    action = np.rad2deg(controls)
    action[5] = (controls[5] - world.lo[5]) / (world.hi[5] - world.lo[5]) * 100
    return action.astype(np.float32), controls
