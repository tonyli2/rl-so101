"""Small simulation collection lifecycle around the native LeRobot writer."""

import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np

from .archive import (
    capture_state,
    mark_committed,
    snapshot_scene,
    write_episode_archive,
)
from .schema import (
    CAMERAS,
    JOINTS,
    applied_action,
    check_encoder,
    create_dataset,
    learning_frame,
)
from .self_collision import (
    RecordingCollisionChecker,
    SelfCollisionError,
    collision_message,
)


class RecordingSession:
    def __init__(
        self,
        simulation,
        root,
        repo_id="local/red_cube_demonstrations",
        *,
        seed=0,
        resume=False,
        source="real_leader",
        layout_factory=None,
    ):
        self.simulation = simulation
        self.layout = None
        self._layout_factory = layout_factory
        self.root = Path(root).expanduser().resolve()
        self.repo_id = repo_id
        self.seed = seed
        self.source = source
        self.incomplete_marker = self.root / "COLLECTION_INCOMPLETE.json"
        self.dataset = None
        self.active = False
        self.review = False
        self.invalid = False
        self._resume = resume
        self.rejection_reason = None
        self.self_collision = None
        self._collision_checker = None
        self._prior_physics_observer = getattr(
            simulation, "physics_step_callback", None
        )
        if self.root.exists() and not resume:
            raise FileExistsError(
                f"Output exists: {self.root}; use a new root or --resume"
            )
        if resume:
            self._check_resume()
        try:
            simulation.physics_step_callback = self._observe_physics_step
            if self._layout_factory is not None:
                self.layout = self._layout_factory(
                    self.world,
                    (
                        json.loads((self.root / "meta/info.json").read_text())[
                            "total_episodes"
                        ]
                        if resume
                        else 0
                    ),
                )
            self._open_writer()
            self.reset()
        except BaseException:
            self.close()
            raise

    def _check_resume(self):
        if (self.root / "QUARANTINED.json").exists():
            raise ValueError(
                "Cannot append demonstrations to a quarantined dataset; use the active collection root"
            )
        from .replay import check_append_metadata

        # Saved episodes were decoded and validated when Keep committed them.
        check_append_metadata(self.root)
        info = json.loads((self.root / "meta/info.json").read_text())
        if int(info["fps"]) != self.simulation.fps:
            raise ValueError("Cannot resume with a different FPS")
        for name in CAMERAS:
            shape = info["features"][f"observation.images.{name}"]["shape"]
            if tuple(shape) != (self.simulation.height, self.simulation.width, 3):
                raise ValueError("Cannot resume with a different image size")

    def _open_writer(self):
        self.dataset = create_dataset(
            self.root,
            self.repo_id,
            self.simulation.fps,
            self.simulation.height,
            self.simulation.width,
            resume=self._resume,
        )
        self._resume = True

    @property
    def world(self):
        return self.simulation._ensure_world()

    @property
    def length(self):
        return len(self.raw_actions) if hasattr(self, "raw_actions") else 0

    def reset(self):
        if self.active or self.review:
            self.discard()
        self.observation, _ = self.simulation.reset(seed=self.seed)
        if self.layout is not None:
            self.layout.sample()
            self.observation = self.simulation._obs(self.world)
        self.self_collision = None
        self.rejection_reason = None
        self.observation_wall = time.monotonic()

    def _observe_physics_step(self, model, data):
        if self._prior_physics_observer is not None:
            self._prior_physics_observer(model, data)
        if self.active and self.self_collision is None:
            self.self_collision = self._checker().check(data)

    def _checker(self):
        if self._collision_checker is None:
            self._collision_checker = RecordingCollisionChecker(self.world.model)
        return self._collision_checker

    def begin(self):
        if self.active or self.review:
            raise RuntimeError(
                "Finish/save/discard the current attempt before starting another"
            )
        if self.layout is not None:
            reason = self.layout.validate_start()
            if reason:
                self.rejection_reason = reason
                raise SelfCollisionError(reason)
        contact = self._checker().check(self.world.data)
        if contact is not None:
            self.self_collision = contact
            self.rejection_reason = collision_message(contact)
            raise SelfCollisionError(
                self.rejection_reason + "; move clear or reset before recording"
            )
        self.self_collision = None
        self.rejection_reason = None
        if self.dataset._is_finalized:
            self._open_writer()
        self.scene_id = snapshot_scene(self.world, self.root)
        self.episode_uuid = uuid.uuid4().hex
        self.episode_index = self.dataset.meta.total_episodes
        with self.incomplete_marker.open("x") as handle:
            json.dump(
                {
                    "episode_uuid": self.episode_uuid,
                    "episode_index": self.episode_index,
                    "source": self.source,
                    "status": "unaccepted_attempt",
                },
                handle,
            )
        self.simulation._steps = 0
        self.states = [capture_state(self.world.model, self.world.data)]
        self.sim_times = [float(self.world.data.time)]
        self.raw_actions, self.applied_actions, self.controls, self.timings = (
            [],
            [],
            [],
            [],
        )
        self.settling, self.clipped = [], []
        self.active, self.review, self.invalid = True, False, False

    def step(self, raw_action, *, leader_read_wall=None, settling=False):
        raw = np.asarray(raw_action, dtype=np.float32)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("Leader action must contain six finite joint targets")
        previous = self.observation
        previous_wall = self.observation_wall
        read_wall = time.monotonic() if leader_read_wall is None else leader_read_wall
        apply_wall = time.monotonic()
        observation, reward, terminated, truncated, info = self.simulation.step(raw)
        completed_wall = time.monotonic()
        self.observation, self.observation_wall = observation, completed_wall
        # mj_step contacts describe the state evaluated during integration.
        # The environment's final mj_forward also exposes a new contact formed
        # at the very last substep, before any frame can be accepted or saved.
        if self.active and self.self_collision is None:
            self.self_collision = self._checker().check(self.world.data)
        if self.active and self.self_collision is not None:
            contact = self.self_collision
            self.discard()
            self.self_collision = contact
            self.rejection_reason = collision_message(contact)
            logging.warning(
                "Recording discarded: %s. Move clear or reset, then start again.",
                self.rejection_reason,
            )
            if contact.get("kind") not in ("upright", "curtain"):
                info["recording_rejected_self_collision"] = contact
        if self.active:
            try:
                action, controls = applied_action(self.world)
                self.dataset.add_frame(
                    learning_frame(previous, action, self.simulation.task_description)
                )
                check_encoder(self.dataset)
                recorded_wall = time.monotonic()
                self.raw_actions.append(raw.copy())
                self.applied_actions.append(action)
                self.controls.append(controls)
                self.states.append(capture_state(self.world.model, self.world.data))
                self.sim_times.append(float(self.world.data.time))
                self.timings.append(
                    [
                        previous_wall,
                        read_wall,
                        apply_wall,
                        completed_wall,
                        recorded_wall,
                        max(0, recorded_wall - apply_wall - 1 / self.simulation.fps),
                    ]
                )
                self.settling.append(bool(settling))
                self.clipped.append(np.abs(action - raw) > 1e-3)
            except BaseException:
                self.invalid = True
                raise
        return observation, reward, terminated, truncated, info

    def finish(self):
        if not self.active:
            raise RuntimeError("No active recording")
        self.active, self.review = False, True

    def save(self, outcome="success", *, annotation="human_confirmed"):
        if self.rejection_reason is not None:
            raise SelfCollisionError(
                self.rejection_reason + "; this attempt was discarded"
            )
        if not self.review or self.invalid or self.length == 0:
            raise RuntimeError("Only a finished, nonempty, valid attempt can be saved")
        check_encoder(self.dataset)
        arrays = {
            "states": np.stack(self.states),
            "raw_actions": np.stack(self.raw_actions),
            "applied_actions": np.stack(self.applied_actions),
            "controls": np.stack(self.controls),
            "sim_times": np.asarray(self.sim_times, dtype=np.float64),
            "wall_timings": np.asarray(self.timings, dtype=np.float64),
            "settling": np.asarray(self.settling, dtype=bool),
            "clipped": np.asarray(self.clipped, dtype=bool),
        }
        metadata = {
            "layout_randomization": self.layout.current
            if self.layout is not None
            else None,
            "repo_id": self.repo_id,
            "camera_map": dict(CAMERAS),
            "height": self.simulation.height,
            "width": self.simulation.width,
            "fps": self.simulation.fps,
            "substeps": int(self.world.substeps),
            "physics_timestep": float(self.world.model.opt.timestep),
            "joint_names": list(JOINTS),
            "units": ["degrees"] * 5 + ["percent_open"],
            "task": self.simulation.task_description,
            "seed": self.seed,
            "rng_state": self.simulation.np_random.bit_generator.state,
            "outcome": outcome,
            "annotation": annotation,
            "source": self.source,
            "timing_columns": [
                "observation_wall",
                "leader_read_wall",
                "apply_wall",
                "next_observation_wall",
                "recorded_wall",
                "simulation_and_writer_overrun",
            ],
            "policy_eligible": self.source == "real_leader" and outcome == "success",
            "upright_collision_check": {
                "version": 1,
                "enabled": self._checker().upright_geom is not None,
                "passed": self._checker().upright_geom is not None,
                "physics_substeps": True,
            },
            "curtain_collision_check": {
                "version": 1,
                "enabled": bool(self._checker().curtain_geoms),
                "passed": bool(self._checker().curtain_geoms),
                "physics_substeps": True,
            },
            "self_collision_check": {
                "version": 1,
                "passed": True,
                "physics_substeps": True,
            },
        }
        write_episode_archive(
            self.root,
            self.episode_uuid,
            self.episode_index,
            self.scene_id,
            arrays,
            metadata,
        )
        # Native writers can retain Parquet footers across episodes: finalize at
        # each accepted boundary, then resume natively for the next attempt.
        try:
            self.dataset.save_episode()
            self.dataset.finalize()
            from .replay import validate

            validate(self.root, self.episode_uuid, allow_pending=True)
            mark_committed(self.root, self.episode_uuid)
            self.incomplete_marker.unlink()
        except BaseException:
            self.invalid = True
            raise
        self.review = False
        return self.episode_uuid

    def discard(self):
        if self.dataset is not None and not self.dataset._is_finalized:
            self.dataset.clear_episode_buffer()
        if not self.invalid and (self.active or self.review):
            self.incomplete_marker.unlink(missing_ok=True)
        self.active = self.review = self.invalid = False
        self.raw_actions = []

    def close(self):
        try:
            if self.active or self.review or self.invalid:
                self.discard()
        finally:
            self.simulation.physics_step_callback = self._prior_physics_observer
            if self.layout is not None:
                self.layout.close()
            if self.dataset is not None:
                self.dataset.finalize()


def open_leader(port, calibration_file):
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    calibration_file = Path(calibration_file).expanduser().resolve()
    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"Existing leader calibration not found: {calibration_file}"
        )
    leader = SOLeader(
        SOLeaderTeleopConfig(
            port=port,
            id=calibration_file.stem,
            calibration_dir=calibration_file.parent,
            use_degrees=True,
        )
    )
    if set(leader.calibration) != set(JOINTS):
        raise ValueError("Calibration must contain all six SO-101 leader joints")
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("Calibration mismatch; stopping without recalibrating")
        return leader
    except BaseException:
        if leader.is_connected:
            leader.disconnect()
        raise
