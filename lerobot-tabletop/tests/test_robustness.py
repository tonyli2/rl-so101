"""CPU-only checks for the frozen checkpoint-selection benchmark.

These exercise actual MuJoCo state restoration with renderer construction mocked;
no display, GPU, policy weights, robot hardware or recording writer is needed.
"""

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import mujoco
import numpy as np
from learning.robustness import ScenarioEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, FreezeAfterEpisodeEnd
from recording.archive import capture_state, restore_state
from recording.schema import JOINTS
from recording.self_collision import SelfCollisionChecker

BUNDLE = Path(__file__).resolve().parents[2] / "outputs/eval/act-robustness-v1"


def fake_renderer(model, height, width):
    renderer = MagicMock()
    renderer.render.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    return renderer


@unittest.skipUnless(
    (BUNDLE / "manifest.json").is_file(), "Frozen benchmark bundle is not present"
)
class FrozenRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((BUNDLE / "manifest.json").read_text())
        with np.load(BUNDLE / "states.npz", allow_pickle=False) as archive:
            cls.states = archive["states"].copy()
            cls.bins = archive["bin_positions"].copy()
        cls.model = mujoco.MjModel.from_binary_path(str(BUNDLE / "model.mjb"))
        cls.cases = cls.manifest["cases"]

    def test_bundle_hashes_and_case_indices_match_frozen_payloads(self):
        for filename, key in (
            ("states.npz", "states_sha256"),
            ("model.mjb", "model_sha256"),
        ):
            actual = hashlib.sha256((BUNDLE / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, self.manifest[key])
        self.assertEqual(len(self.states), len(self.cases))
        self.assertEqual(len(self.bins), len(self.cases))
        self.assertEqual(
            [case["index"] for case in self.cases], list(range(len(self.cases)))
        )
        self.assertEqual(len({case["id"] for case in self.cases}), len(self.cases))

    def test_primary_starts_are_self_contact_free_and_within_limits(self):
        data = mujoco.MjData(self.model)
        checker = SelfCollisionChecker(self.model)
        joint_ids = [self.model.joint(name).id for name in JOINTS]
        qadr = self.model.jnt_qposadr[joint_ids]
        limits = self.model.jnt_range[joint_ids]
        box = self.model.body("box").id
        for case in self.cases:
            if not case["primary"]:
                continue
            with self.subTest(case=case["id"]):
                self.model.body_pos[box] = self.bins[case["index"]]
                restore_state(self.model, data, self.states[case["index"]])
                self.assertIsNone(checker.check(data))
                self.assertTrue(np.all(data.qpos[qadr] >= limits[:, 0] - 1e-8))
                self.assertTrue(np.all(data.qpos[qadr] <= limits[:, 1] + 1e-8))

    def test_legacy_control_is_excluded_from_primary_score(self):
        legacy = [case for case in self.cases if case["legacy"]]
        self.assertEqual(len(legacy), 1)
        self.assertFalse(legacy[0]["primary"])
        self.assertEqual(legacy[0]["family"], "historical_control")
        self.assertIsNotNone(legacy[0]["initial_contact"])

    def test_object_offset_axes_magnitudes_and_endpoint_witnesses(self):
        anchor = next(case for case in self.cases if case["id"] == "anchor")
        cube0, bin0 = (
            np.asarray(anchor["cube_position"]),
            np.asarray(anchor["bin_position"]),
        )
        away = np.asarray(self.manifest["bin_axis_away"])
        side = np.asarray(self.manifest["bin_axis_sideways"])
        self.assertAlmostEqual(float(np.linalg.norm(away)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(side)), 1.0)
        self.assertAlmostEqual(float(np.dot(away, side)), 0.0)
        self.assertTrue(self.manifest["bin_reachability_checked"])
        for case in self.cases:
            with self.subTest(case=case["id"]):
                np.testing.assert_allclose(
                    np.asarray(case["cube_position"])[:2] - cube0[:2],
                    case["cube_delta_m"],
                    atol=1e-10,
                )
                delta = np.asarray(case["bin_position"]) - bin0
                expected = case["bin_away_m"] * away + case["bin_side_m"] * side
                np.testing.assert_allclose(delta[:2], expected, atol=1e-10)
                self.assertAlmostEqual(float(delta[2]), 0.0)
                witness = case["bin_reachability"]
                self.assertTrue(witness["reachable"])
                if np.linalg.norm(delta) > 1e-9:
                    self.assertLessEqual(witness["position_error_m"], 0.005)
                    self.assertLessEqual(witness["axis_error_rad"], np.deg2rad(15))
                    self.assertTrue(witness["within_limits"])
                    self.assertIsNone(witness["self_contact"])
                    self.assertFalse(witness["obstacle_contact"])

    def test_reset_restores_exact_states_images_and_bin_after_mutation(self):
        index = next(case["index"] for case in self.cases if case["family"] == "bin")
        with patch("learning.robustness.mujoco.Renderer", side_effect=fake_renderer):
            env = ScenarioEnv(BUNDLE, index)
            try:
                first, first_info = env.reset(seed=1000)
                world = env._ensure_world()
                first_state = capture_state(world.model, world.data)
                first_bin = world.model.body_pos[world.box_body].copy()
                world.model.body_pos[world.box_body] += [0.1, -0.1, 0.05]
                world.data.qpos[world.qadr] += 0.01
                second, second_info = env.reset(seed=1000)
                np.testing.assert_array_equal(
                    capture_state(world.model, world.data), first_state
                )
                np.testing.assert_array_equal(
                    world.model.body_pos[world.box_body], first_bin
                )
                np.testing.assert_array_equal(second["agent_pos"], first["agent_pos"])
                for name in first["pixels"]:
                    np.testing.assert_array_equal(
                        second["pixels"][name], first["pixels"][name]
                    )
                self.assertEqual(first_info["scenario_id"], second_info["scenario_id"])
                # A separate instance represents a subsequent checkpoint's fresh env.
                other = ScenarioEnv(BUNDLE, index)
                try:
                    obs, _ = other.reset(seed=1000)
                    np.testing.assert_array_equal(obs["agent_pos"], first["agent_pos"])
                    np.testing.assert_array_equal(
                        capture_state(
                            other._ensure_world().model, other._ensure_world().data
                        ),
                        first_state,
                    )
                finally:
                    other.close()
            finally:
                env.close()


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self):
        self.reset_count = 0
        self.step_count = 0
        self.metrics = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_count += 1
        self.metrics = []
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        self.metrics.append("terminal contact metric")
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {"is_success": True}


class FinishedCaseTests(unittest.TestCase):
    def test_same_step_autoreset_does_not_erase_metrics_or_restart_finished_case(self):
        base = OneStepEnv()
        wrapped = FreezeAfterEpisodeEnd(base)
        vector = gym.vector.SyncVectorEnv(
            [lambda: wrapped], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        try:
            vector.reset(seed=[1000], options={NEW_ROLLOUT_OPTION: True})
            vector.step(np.zeros((1, 1), dtype=np.float32))
            self.assertTrue(wrapped.is_frozen)
            self.assertEqual(base.reset_count, 1)
            self.assertEqual(base.step_count, 1)
            self.assertEqual(base.metrics, ["terminal contact metric"])
            vector.step(np.zeros((1, 1), dtype=np.float32))
            self.assertEqual(base.reset_count, 1)
            self.assertEqual(base.step_count, 1)
            self.assertEqual(base.metrics, ["terminal contact metric"])
            vector.reset(options={NEW_ROLLOUT_OPTION: True})
            self.assertEqual(base.reset_count, 2)
            self.assertFalse(wrapped.is_frozen)
        finally:
            vector.close()


if __name__ == "__main__":
    unittest.main()
