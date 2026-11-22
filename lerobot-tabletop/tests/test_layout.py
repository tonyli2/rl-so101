"""Layout constraints and visibility without GPU or dataset dependencies."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from recording.layout import LayoutSampler
from recording.self_collision import SelfCollisionChecker

XML = """<mujoco><worldbody>
<body name="robot"><joint name="shoulder_pan"/><geom size=".01" pos="0 0 .3"/></body>
<body name="red_cube" pos=".23 0 .015"><freejoint/><geom size=".015 .015 .015" type="box"/></body>
<body name="box" pos=".22 .125 0"><geom size=".049 .049 .005" type="box"/></body>
</worldbody></mujoco>"""


def sampler(seed=7):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    item = LayoutSampler.__new__(LayoutSampler)
    item.world = SimpleNamespace(model=model, data=data)
    item.rng = np.random.default_rng(seed)
    item.cube, item.box = model.body("red_cube").id, model.body("box").id
    item.qa = int(model.jnt_qposadr[model.body_jntadr[item.cube]])
    item.cube0, item.bin0 = (
        model.body_pos[item.cube].copy(),
        model.body_pos[item.box].copy(),
    )
    direction = item.bin0[:2] - item.cube0[:2]
    item.away = direction / np.linalg.norm(direction)
    item.side = np.array([-item.away[1], item.away[0]])
    item.cube_reference = item.bin_reference = {
        "position": [0.23, 0, 0.1],
        "qpos": data.qpos.copy(),
    }
    item.checker = SelfCollisionChecker(model)
    item.current = None
    return item


class LayoutTests(unittest.TestCase):
    def test_sampling_varies_within_bounds_and_is_reproducible(self):
        a, b = sampler(), sampler()
        a.visible = b.visible = lambda: True
        with patch(
            "recording.layout.find_reachability_witness",
            return_value={"reachable": True},
        ):
            rows = [a.sample() for _ in range(20)]
            other = [b.sample() for _ in range(20)]
        self.assertEqual(
            [r["cube_position"] for r in rows], [r["cube_position"] for r in other]
        )
        self.assertEqual(len({tuple(r["cube_position"]) for r in rows}), 20)
        for row in rows:
            self.assertTrue(np.all(np.abs(row["cube_delta_m"]) <= 0.04))
            self.assertTrue(0 <= row["bin_away_m"] <= 0.04)
            self.assertLessEqual(abs(row["bin_side_m"]), 0.04)
            self.assertTrue(
                np.any(
                    np.abs(
                        np.array(row["cube_position"])[:2]
                        - np.array(row["bin_position"])[:2]
                    )
                    >= 0.074
                )
            )

    def test_unverified_endpoints_are_not_accepted(self):
        item = sampler()
        item.visible = lambda: True
        with patch(
            "recording.layout.find_reachability_witness",
            return_value={"reachable": False},
        ):
            with self.assertRaisesRegex(RuntimeError, "300 attempts"):
                item.sample()

    def test_segmentation_requires_visible_cube_pixels_inside_frame(self):
        item = sampler()
        item.width = item.height = 32
        image = np.full((32, 32, 2), -1, dtype=np.int32)
        item.renderer = Mock()
        item.renderer.render.return_value = image
        self.assertFalse(item.visible())
        cube_geom = np.flatnonzero(item.world.model.geom_bodyid == item.cube)[0]
        image[10:14, 10:14] = [cube_geom, int(mujoco.mjtObj.mjOBJ_GEOM)]
        self.assertTrue(item.visible())
        image[:2, :2] = [cube_geom, int(mujoco.mjtObj.mjOBJ_GEOM)]
        self.assertFalse(item.visible())

    def test_start_rechecks_visibility_and_cube_displacement(self):
        item = sampler()
        item.visible = lambda: True
        item.current = {"cube_position": item.world.data.xpos[item.cube].tolist()}
        self.assertIsNone(item.validate_start())
        item.visible = lambda: False
        self.assertIn("visible", item.validate_start())
        item.visible = lambda: True
        item.world.data.qpos[item.qa] += 0.01
        self.assertIn("moved", item.validate_start())
