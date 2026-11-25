"""Physics-backed collection rejection without cameras, hardware or real datasets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from env import SO101Env
from recording.self_collision import SelfCollisionChecker, SelfCollisionError
from recording.session import RecordingSession

XML = """
<mujoco><option gravity="0 0 0" timestep="0.005"/>
<worldbody>
  <geom name="table" type="plane" size="1 1 .1" pos="0 0 0"/>
  <body name="base" pos="0 0 .3">
    <body name="shoulder"><joint name="shoulder_pan" type="hinge"/>
      <geom name="shoulder_collision" type="sphere" size=".06"/>
      <body name="arm"><joint name="elbow_flex" type="hinge"/>
        <geom type="sphere" size=".01" pos=".4 0 0" contype="0" conaffinity="0"/>
        <body name="gripper" pos=".2 0 0"><joint name="gripper" type="slide" axis="1 0 0"/>
          <geom name="gripper_collision" type="sphere" size=".06"/>
        </body>
      </body>
    </body>
  </body>
  <body name="red_cube" pos=".3 0 .3"><freejoint/>
    <geom name="cube" type="sphere" size=".02"/>
  </body>
</worldbody>
<actuator><motor joint="gripper"/></actuator>
</mujoco>
"""


class PhysicsOnlyEnv(SO101Env):
    def __init__(self):
        super().__init__(episode_length=100, terminate_on_success=False)
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        self._world = SimpleNamespace(
            model=model,
            data=data,
            substeps=3,
            actuators=np.array([0]),
            to_ctrl=lambda action: np.zeros(1),
            cube_bodies=[model.body("red_cube").id],
            cubes_in_box=lambda: 0,
            instruction="Place the cube.",
        )
        self.reset()

    def reset(self, **kwargs):
        world = self._world
        mujoco.mj_resetData(world.model, world.data)
        world.data.qpos[world.model.joint("gripper").qposadr[0]] = 0.1
        mujoco.mj_forward(world.model, world.data)
        self._steps = 0
        return self._obs(world), {}

    def _obs(self, world):
        return {
            "agent_pos": np.zeros(6, dtype=np.float32),
            "pixels": {
                name: np.zeros((480, 640, 3), dtype=np.uint8)
                for name in ("overhead", "wrist")
            },
        }

    def close(self):
        pass


class FakeWriter:
    _is_finalized = False
    meta = SimpleNamespace(total_episodes=0)

    def __init__(self):
        self.frames = 0
        self.discards = 0
        self.saved = 0

    def clear_episode_buffer(self):
        self.frames = 0
        self.discards += 1

    def add_frame(self, frame):
        self.frames += 1

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        self._is_finalized = True


class SelfCollisionTests(unittest.TestCase):
    def setUp(self):
        self.sim = PhysicsOnlyEnv()
        self.world = self.sim._world
        self.checker = SelfCollisionChecker(self.world.model)
        self.qadr = self.world.model.joint("gripper").qposadr[0]

    def collide(self):
        self.world.data.qpos[self.qadr] = -0.2
        mujoco.mj_forward(self.world.model, self.world.data)

    def session(self, root):
        writer = FakeWriter()

        def create(*args, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            return writer

        create_patch = patch("recording.session.create_dataset", side_effect=create)
        create_patch.start()
        self.addCleanup(create_patch.stop)
        return RecordingSession(self.sim, root), writer

    def test_detects_nonadjacent_robot_contact_but_allows_cube_contact(self):
        # The gripper touches the task object in this initial pose.
        self.assertTrue(
            any(
                self.world.model.geom(c.geom[1]).name == "cube"
                for c in self.world.data.contact
            )
        )
        self.assertIsNone(self.checker.check(self.world.data))
        self.collide()
        hit = self.checker.check(self.world.data)
        self.assertEqual(set(hit["bodies"]), {"shoulder", "gripper"})
        self.assertGreater(hit["penetration_m"], 0)

    def test_initial_collision_blocks_begin_before_marker_or_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            session, writer = self.session(root)
            self.collide()
            with self.assertRaisesRegex(SelfCollisionError, "shoulder.*gripper"):
                session.begin()
            self.assertFalse(session.incomplete_marker.exists())
            self.assertFalse(session.active)
            self.assertEqual(writer.frames, 0)
            self.assertEqual(writer.saved, 0)
            session.close()

    def test_transient_substep_collision_discards_and_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            session, writer = self.session(root)
            # Avoid source-bundling work unrelated to this physics gate.
            with patch("recording.session.snapshot_scene", return_value="test-scene"):
                session.begin()
            calls = []

            def transient(model, data):
                calls.append(data.time)
                # A real penetrating pose exists only at the first internal
                # physics boundary; final 30 Hz observation is clear.
                data.qpos[self.qadr] = -0.2 if len(calls) == 1 else 0.1
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)

            session._prior_physics_observer = transient
            _, _, _, _, info = session.step(np.zeros(6))
            self.assertEqual(len(calls), self.world.substeps)
            self.assertIsNone(self.checker.check(self.world.data))
            self.assertIn("recording_rejected_self_collision", info)
            self.assertFalse(session.active)
            self.assertFalse(session.review)
            self.assertFalse(session.incomplete_marker.exists())
            self.assertEqual(writer.discards, 1)
            self.assertEqual(writer.frames, 0)
            self.assertEqual(writer.saved, 0)
            with self.assertRaises(SelfCollisionError):
                session.save()
            # Automatic collision rejection is recoverable without a restart.
            session._prior_physics_observer = None
            session.reset()
            with patch("recording.session.snapshot_scene", return_value="test-scene"):
                session.begin()
            self.assertTrue(session.active)
            session.close()

    def test_contact_created_at_last_substep_is_rejected_after_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            session, writer = self.session(root)
            with patch("recording.session.snapshot_scene", return_value="test-scene"):
                session.begin()
            calls = []

            def final_contact(model, data):
                calls.append(data.time)
                if len(calls) == self.world.substeps:
                    # Do not forward here: during mj_step the contact list can
                    # still describe its pre-integration state.
                    data.qpos[self.qadr] = -0.2

            session._prior_physics_observer = final_contact
            session.step(np.zeros(6))
            self.assertIsNotNone(self.checker.check(self.world.data))
            self.assertFalse(session.active)
            self.assertEqual(writer.discards, 1)
            self.assertFalse(session.incomplete_marker.exists())
            session.close()

    def test_quarantined_dataset_cannot_be_resumed_for_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "QUARANTINED.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                RecordingSession(self.sim, root, resume=True)

    def test_environment_keeps_batched_fast_path_without_observer(self):
        original = mujoco.mj_step
        calls = []

        def record(model, data, nstep=1):
            calls.append(nstep)
            return original(model, data, nstep=nstep)

        with patch("env.mujoco.mj_step", side_effect=record):
            self.sim.step(np.zeros(6))
        self.assertEqual(calls, [self.world.substeps])


if __name__ == "__main__":
    unittest.main()
