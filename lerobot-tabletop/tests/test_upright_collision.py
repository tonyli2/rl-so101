"""Physics-backed upright recording rejection, including transient contacts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from recording.self_collision import RecordingCollisionChecker, SelfCollisionError
from recording.session import RecordingSession
from test_self_collision import XML, FakeWriter, PhysicsOnlyEnv

POLE = """<body name="overhead_stand" pos=".4 0 .3">
<geom name="overhead_upright_collision" type="box" size=".015 .02 .2"/>
</body>"""


class UprightCollisionTests(unittest.TestCase):
    def setUp(self):
        self.sim = PhysicsOnlyEnv()
        world = self.sim._world
        world.model = mujoco.MjModel.from_xml_string(
            XML.replace("</worldbody>", POLE + "</worldbody>")
        )
        world.data = mujoco.MjData(world.model)
        self.sim.reset()
        self.world = world
        self.qadr = world.model.joint("gripper").qposadr[0]
        self.checker = RecordingCollisionChecker(world.model)

    def collide(self):
        self.world.data.qpos[self.qadr] = 0.2
        self.world.data.qvel[:] = 0
        mujoco.mj_forward(self.world.model, self.world.data)

    def session(self, root):
        writer = FakeWriter()

        def create(*args, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            return writer

        patcher = patch("recording.session.create_dataset", side_effect=create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return RecordingSession(self.sim, root), writer

    def test_cube_contact_allowed_and_arm_pole_contact_detected(self):
        self.assertIsNone(self.checker.check(self.world.data))
        self.collide()
        contact = self.checker.check(self.world.data)
        self.assertEqual(contact["kind"], "upright")
        self.assertEqual(set(contact["bodies"]), {"gripper", "overhead_stand"})
        self.assertGreater(contact["penetration_m"], 0)

    def test_initial_contact_rejects_before_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            session, writer = self.session(Path(directory) / "data")
            self.collide()
            with self.assertRaisesRegex(SelfCollisionError, "Upright collision"):
                session.begin()
            self.assertFalse(session.active)
            self.assertFalse(session.incomplete_marker.exists())
            self.assertEqual(writer.frames, 0)
            session.close()

    def test_transient_or_last_substep_contact_discards_and_blocks_save(self):
        for final_only in (False, True):
            with (
                self.subTest(final_only=final_only),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.sim.reset()
                session, writer = self.session(Path(directory) / "data")
                with patch("recording.session.snapshot_scene", return_value="test"):
                    session.begin()
                ticks = []

                def observer(model, data):
                    ticks.append(data.time)
                    if final_only:
                        if len(ticks) == self.world.substeps:
                            data.qpos[self.qadr] = 0.2
                    else:
                        data.qpos[self.qadr] = 0.2 if len(ticks) == 1 else 0.1
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)

                session._prior_physics_observer = observer
                session.step(np.zeros(6))
                self.assertFalse(session.active)
                self.assertFalse(session.review)
                self.assertEqual(writer.discards, 1)
                self.assertEqual(writer.frames, 0)
                self.assertEqual(writer.saved, 0)
                self.assertIn("Upright collision", session.rejection_reason)
                self.assertFalse(session.incomplete_marker.exists())
                with self.assertRaises(SelfCollisionError):
                    session.save()
                session._prior_physics_observer = None
                session.reset()
                with patch("recording.session.snapshot_scene", return_value="test"):
                    session.begin()
                self.assertTrue(session.active)
                session.close()

    def test_task_object_touching_pole_is_not_an_arm_collision(self):
        body = self.world.model.body("red_cube").id
        qa = self.world.model.jnt_qposadr[self.world.model.body_jntadr[body]]
        self.world.data.qpos[qa : qa + 3] = [0.4, 0, 0.3]
        mujoco.mj_forward(self.world.model, self.world.data)
        self.assertTrue(
            any(self.checker.upright_geom in c.geom for c in self.world.data.contact)
        )
        self.assertIsNone(self.checker.check(self.world.data))


if __name__ == "__main__":
    unittest.main()
