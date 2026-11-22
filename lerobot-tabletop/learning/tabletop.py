"""Register the mounted task with LeRobot; keep collection's environment unchanged."""

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from mounts.environment import MountedSO101Env, build_variant


class EvaluationEnv(MountedSO101Env):
    """Success means a released, settled cube wholly inside the bin for 0.5 seconds."""

    def __init__(self, *, reset_dataset_root=None, reset_episode=45, **kwargs):
        super().__init__(**kwargs)
        self.reset_dataset_root = reset_dataset_root
        self.reset_episode = reset_episode
        self._initial_state = None

    def reset(self, **kwargs):
        self._settled_steps = 0
        observation, info = super().reset(**kwargs)
        if self.reset_dataset_root is not None:
            from recording.archive import load_episode_archive, restore_state

            world = self._ensure_world()
            if self._initial_state is None:
                manifest, arrays, saved = load_episode_archive(
                    Path(self.reset_dataset_root).expanduser(), self.reset_episode
                )
                if not manifest["metadata"].get("policy_eligible", False):
                    raise ValueError(
                        "Evaluation reset must be a successful real-leader episode"
                    )
                current = world.model
                if (
                    current.names != saved.names
                    or current.opt.timestep != saved.opt.timestep
                ):
                    raise ValueError("Current scene differs from archived reset scene")
                for name in (
                    "qpos0",
                    "body_pos",
                    "body_quat",
                    "body_mass",
                    "body_inertia",
                    "geom_pos",
                    "geom_quat",
                    "geom_size",
                    "geom_friction",
                    "geom_contype",
                    "geom_conaffinity",
                    "jnt_range",
                    "dof_damping",
                    "actuator_ctrlrange",
                    "cam_pos",
                    "cam_quat",
                    "cam_fovy",
                ):
                    if not np.array_equal(getattr(current, name), getattr(saved, name)):
                        raise ValueError(
                            f"Current scene {name} differs from archived reset scene"
                        )
                self._initial_state = arrays["states"][0].copy()
            restore_state(world.model, world.data, self._initial_state)
            observation = self._obs(world)
            info["reset_episode"] = self.reset_episode
        return observation, info

    def placement_valid(self):
        world = self._ensure_world()
        model, data = world.model, world.data
        cube = world.cube_bodies[0]
        geom = int(model.body_geomadr[cube])
        rotation = data.xmat[world.box_body].reshape(3, 3)
        local = (data.geom_xpos[geom] - data.xpos[world.box_body]) @ rotation
        extent = (
            np.abs(rotation.T @ data.geom_xmat[geom].reshape(3, 3))
            @ model.geom_size[geom]
        )
        # Bin's interior: +/-45 mm, floor top 4 mm, rim 35 mm.
        inside = bool(
            np.all(np.abs(local[:2]) + extent[:2] < 0.045)
            and local[2] - extent[2] >= 0.002
            and local[2] + extent[2] <= 0.037
        )
        settled = bool(
            np.linalg.norm(data.cvel[cube, 3:]) < 0.02
            and np.linalg.norm(data.cvel[cube, :3]) < 0.5
        )
        released = True
        for contact in data.contact:
            bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
            if cube in bodies and any(
                b not in (cube, world.box_body, 0) for b in bodies
            ):
                released = False
                break
        return inside and settled and released

    def step(self, action):
        obs, _, _, truncated, info = super().step(action)
        self._settled_steps = self._settled_steps + 1 if self.placement_valid() else 0
        success = self._settled_steps >= int(np.ceil(self.fps * 0.5))
        info.update(is_success=success, placement_settled_steps=self._settled_steps)
        return obs, float(success), success, truncated, info


@EnvConfig.register_subclass("so101_tabletop")
@dataclass
class TabletopConfig(EnvConfig):
    task: str | None = "red_cube_in_bin"
    fps: int = 30
    episode_length: int = 900
    reset_dataset_root: str | None = None
    reset_episode: int = 45
    observation_height: int = 480
    observation_width: int = 640
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.features = {
            "action": PolicyFeature(FeatureType.ACTION, (6,)),
            "agent_pos": PolicyFeature(FeatureType.STATE, (6,)),
            **{
                f"pixels/{cam}": PolicyFeature(
                    FeatureType.VISUAL,
                    (self.observation_height, self.observation_width, 3),
                )
                for cam in ("overhead", "wrist")
            },
        }
        self.features_map = {
            "action": "action",
            "agent_pos": "observation.state",
            **{
                f"pixels/{cam}": f"observation.images.{cam}"
                for cam in ("overhead", "wrist")
            },
        }

    @property
    def gym_kwargs(self):
        return {
            "reset_dataset_root": self.reset_dataset_root,
            "reset_episode": self.reset_episode,
            "fps": self.fps,
            "episode_length": self.episode_length,
            "observation_height": self.observation_height,
            "observation_width": self.observation_width,
            "cameras": {"overhead": "overhead", "wrist": "wrist_cam"},
            "randomize": {},
            "terminate_on_success": False,
        }

    def create_envs(self, n_envs, use_async_envs=False):
        if n_envs != 1 or use_async_envs:
            raise ValueError(
                "Use --eval.batch_size=1 --eval.use_async_envs=false on this desktop"
            )
        build_variant()
        factory = partial(EvaluationEnv, **self.gym_kwargs)
        vector = gym.vector.SyncVectorEnv(
            [factory], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        return {self.type: {0: vector}}
