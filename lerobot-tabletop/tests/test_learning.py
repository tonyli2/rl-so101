"""Check evaluator units, termination and false-positive placement cases."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from learning.tabletop import TabletopConfig  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = TabletopConfig(
            observation_height=120, observation_width=160, episode_length=40
        )
        self.vector = self.config.create_envs(1)[self.config.type][0]
        self.env = self.vector.envs[0]
        self.env.reset(seed=0)
        self.world = self.env._ensure_world()

    def tearDown(self):
        self.vector.close()

    def place(self, offset):
        world = self.world
        cube = world.cube_bodies[0]
        joint = world.model.body_jntadr[cube]
        address = world.model.jnt_qposadr[joint]
        world.data.qpos[address : address + 3] = (
            world.data.xpos[world.box_body] + offset
        )
        world.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
        world.data.qvel[:] = 0
        mujoco.mj_forward(world.model, world.data)

    def test_placement_and_hold_reset(self):
        self.assertFalse(self.env.placement_valid())
        self.place(np.array([0.0, 0.0, 0.019]))
        self.assertTrue(self.env.placement_valid())
        action = self.world.joint_state()
        for _ in range(14):
            _, _, terminated, _, info = self.env.step(action)
            self.assertFalse(terminated)
        _, reward, terminated, _, info = self.env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 1)
        self.assertTrue(info["is_success"])
        self.env.reset(seed=0)
        self.assertEqual(self.env._settled_steps, 0)
        self.assertFalse(self.env.placement_valid())

    def test_reject_wall_overlap_above_and_below_bin(self):
        for offset in ([0.04, 0.0, 0.019], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]):
            self.place(np.array(offset))
            self.assertFalse(self.env.placement_valid())

    def test_reject_robot_contact_and_motion(self):
        self.place(np.array([0.0, 0.0, 0.019]))
        world = self.world
        cube = world.cube_bodies[0]
        cube_geom = int(world.model.body_geomadr[cube])
        robot_geom = world.model.geom("fixed_jaw_box1").id
        fake_data = SimpleNamespace(
            **{
                name: getattr(world.data, name)
                for name in ("xmat", "xpos", "geom_xpos", "geom_xmat", "cvel")
            },
            contact=[SimpleNamespace(geom=[cube_geom, robot_geom])],
        )
        with patch.object(
            self.env,
            "_ensure_world",
            return_value=SimpleNamespace(
                model=world.model,
                data=fake_data,
                cube_bodies=world.cube_bodies,
                box_body=world.box_body,
            ),
        ):
            self.assertFalse(self.env.placement_valid())
        world.data.cvel[cube, 3] = 0.1
        self.assertFalse(self.env.placement_valid())

    def test_native_observation_mapping_and_action_units(self):
        from lerobot.envs.utils import preprocess_observation
        from recording.schema import applied_action

        obs, _ = self.vector.reset(seed=[0])
        mapped = preprocess_observation(obs)
        self.assertEqual(
            tuple(mapped["observation.images.overhead"].shape), (1, 3, 120, 160)
        )
        self.assertEqual(
            tuple(mapped["observation.images.wrist"].shape), (1, 3, 120, 160)
        )
        self.assertEqual(tuple(mapped["observation.state"].shape), (1, 6))
        target = np.array([10.0, -20.0, 30.0, -40.0, 50.0, 60.0], dtype=np.float32)
        self.env.step(target)
        applied, _ = applied_action(self.world)
        np.testing.assert_allclose(applied, target, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
