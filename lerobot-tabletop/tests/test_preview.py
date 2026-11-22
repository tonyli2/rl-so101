"""Extra viewpoints must not modify the recording or simulator state."""

import os

os.environ["MUJOCO_GL"] = "egl"

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


class MultiViewTests(unittest.TestCase):
    def test_extra_views_preserve_state_and_recording_cameras(self):
        from mounts.environment import make_single_env
        from recording.archive import capture_state
        from recording.preview import MultiViewRenderer

        simulation = make_single_env(observation_height=120, observation_width=160)
        preview = None
        try:
            observation, _ = simulation.reset(seed=0)
            world = simulation._ensure_world()
            state = capture_state(world.model, world.data)
            camera_positions = world.model.cam_pos.copy()
            camera_orientations = world.model.cam_quat.copy()
            images_before = {
                name: image.copy() for name, image in observation["pixels"].items()
            }
            # Spawned display workers receive an independent pickled model.
            import pickle

            model_copy = pickle.loads(pickle.dumps(world.model))
            np.testing.assert_array_equal(model_copy.cam_pos, world.model.cam_pos)
            preview = MultiViewRenderer(model_copy, height=120, width=160)
            images = preview.render(world)
            self.assertEqual(
                list(images), ["Front", "Overhead / top", "Side", "Diagonal"]
            )
            for image in images.values():
                self.assertEqual(image.shape, (120, 160, 3))
                self.assertEqual(image.dtype, np.uint8)
                self.assertGreater(float(image.std()), 1)
            self.assertFalse(np.array_equal(images["Front"], images["Side"]))
            np.testing.assert_array_equal(state, capture_state(world.model, world.data))
            np.testing.assert_array_equal(camera_positions, world.model.cam_pos)
            np.testing.assert_array_equal(camera_orientations, world.model.cam_quat)
            for name, before in images_before.items():
                np.testing.assert_array_equal(before, observation["pixels"][name])
        finally:
            if preview is not None:
                preview.close()
            simulation.close()

    def test_multi_view_rejects_headless_before_opening_hardware(self):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "record_demonstrations.py"),
                "--root",
                "/tmp/unused-preview-test",
                "--synthetic",
                "--headless",
                "--multi-view",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--multi-view requires a desktop window", result.stderr)


if __name__ == "__main__":
    unittest.main()
