"""Measured curtain placement, render isolation and physics-step recording gate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import env
import mujoco
import numpy as np
from mounts.enclosure import INSPECTION_ALPHA, update_paper_scene
from mounts.environment import build_variant
from recording.archive import capture_state
from recording.self_collision import RecordingCollisionChecker, SelfCollisionError
from recording.session import RecordingSession
from test_self_collision import XML, FakeWriter, PhysicsOnlyEnv

ROOT = Path(__file__).resolve().parents[1]


class SceneRenderer:
    """Use MuJoCo's real visualization scene without a GPU/context."""

    def __init__(self, model):
        self.model = model
        self.scene = mujoco.MjvScene(model, maxgeom=1000)

    def update_scene(self, data, camera, scene_option):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.model.camera(camera).id
        mujoco.mjv_updateScene(
            self.model,
            data,
            scene_option,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )


class EnclosureGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = build_variant()
        cls.model = env.build_model(root / "scene.xml", root)
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)
        cls.config = json.loads((ROOT / "mounts/config.json").read_text())

    def test_measured_interior_faces_and_table_support(self):
        origin = np.array(self.config["stand_position"]) + [0.0127, 0, 0]
        m, d = self.model, self.data
        left, right, front = [
            m.geom("paper_curtain_" + name).id for name in ("left", "right", "front")
        ]
        # Measurement +X is world -Y; measurement +Y is world +X.
        self.assertAlmostEqual(
            d.geom_xpos[left, 1] - m.geom_size[left, 0], origin[1] + 0.53
        )
        self.assertAlmostEqual(
            d.geom_xpos[right, 1] + m.geom_size[right, 0], origin[1] - 0.53
        )
        self.assertAlmostEqual(
            d.geom_xpos[front, 0] - m.geom_size[front, 1], origin[0] + 0.455
        )
        for gid in (left, right):
            self.assertAlmostEqual(d.geom_xpos[gid, 0] - m.geom_size[gid, 1], origin[0])
            self.assertAlmostEqual(
                d.geom_xpos[gid, 0] + m.geom_size[gid, 1], origin[0] + 0.455
            )
        for gid in (left, right, front):
            self.assertAlmostEqual(d.geom_xpos[gid, 2] - m.geom_size[gid, 2], 0)
            self.assertAlmostEqual(d.geom_xpos[gid, 2] + m.geom_size[gid, 2], 0.41)
            self.assertEqual(m.geom_group[gid], 5)
            self.assertTrue(m.geom_contype[gid] and m.geom_conaffinity[gid])
        table = m.geom("table").id
        self.assertAlmostEqual(m.geom_pos[table, 2] + m.geom_size[table, 2], 0)
        self.assertLessEqual(
            m.geom_pos[table, 1] - m.geom_size[table, 1], origin[1] - 0.53 + 1e-10
        )
        self.assertGreaterEqual(
            m.geom_pos[table, 1] + m.geom_size[table, 1], origin[1] + 0.53 - 1e-10
        )

    def test_sensor_opaque_inspection_translucent_without_model_changes(self):
        m, d = self.model, self.data
        rgba = m.geom_rgba.copy()
        state = capture_state(m, d)
        cams = m.cam_pos.copy()
        renderer = SceneRenderer(m)
        for inspection in (False, True, False):
            update_paper_scene(renderer, m, d, "overhead", inspection=inspection)
            alphas = [
                g.rgba[3]
                for g in renderer.scene.geoms[: renderer.scene.ngeom]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                and (m.geom(g.objid).name or "").startswith("paper_curtain_")
            ]
            self.assertEqual(len(alphas), 3)
            np.testing.assert_allclose(alphas, INSPECTION_ALPHA if inspection else 1)
        np.testing.assert_array_equal(m.geom_rgba, rgba)
        np.testing.assert_array_equal(capture_state(m, d), state)
        np.testing.assert_array_equal(m.cam_pos, cams)


class CurtainRecordingTests(unittest.TestCase):
    def make_sim(self, name):
        sim = PhysicsOnlyEnv()
        w = sim._world
        panel = f'<body name="curtains" pos=".4 0 .3"><geom name="paper_curtain_{name}" type="box" size=".015 .02 .2" group="5" rgba="1 1 1 .15"/></body>'
        w.model = mujoco.MjModel.from_xml_string(
            XML.replace("</worldbody>", panel + "</worldbody>")
        )
        w.data = mujoco.MjData(w.model)
        sim.reset()
        return sim

    def make_session(self, sim, root):
        writer = FakeWriter()

        def create(*args, **kwargs):
            root.mkdir(parents=True, exist_ok=True)
            return writer

        patcher = patch("recording.session.create_dataset", side_effect=create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return RecordingSession(sim, root), writer

    def test_start_and_transient_contacts_rejected_for_all_three_curtains(self):
        for name in ("left", "right", "front"):
            with self.subTest(curtain=name), tempfile.TemporaryDirectory() as directory:
                sim = self.make_sim(name)
                w = sim._world
                qa = w.model.joint("gripper").qposadr[0]
                session, writer = self.make_session(sim, Path(directory) / "dataset")
                w.data.qpos[qa] = 0.2
                mujoco.mj_forward(w.model, w.data)
                hit = RecordingCollisionChecker(w.model).check(w.data)
                self.assertEqual(hit["kind"], "curtain")
                self.assertEqual(hit["curtain"], name)
                with self.assertRaisesRegex(
                    SelfCollisionError, f"Curtain collision \\({name}\\)"
                ):
                    session.begin()
                self.assertFalse(session.incomplete_marker.exists())
                session.reset()
                with patch("recording.session.snapshot_scene", return_value="test"):
                    session.begin()
                ticks = []

                def transient(model, data):
                    ticks.append(data.time)
                    data.qpos[qa] = 0.2 if len(ticks) == 1 else 0.1
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)

                session._prior_physics_observer = transient
                session.step(np.zeros(6))
                self.assertIsNone(RecordingCollisionChecker(w.model).check(w.data))
                self.assertFalse(session.active)
                self.assertEqual(writer.discards, 1)
                self.assertEqual(writer.frames, 0)
                self.assertEqual(writer.saved, 0)
                self.assertIn(name, session.rejection_reason)
                with self.assertRaises(SelfCollisionError):
                    session.save()
                session.close()


if __name__ == "__main__":
    unittest.main()
