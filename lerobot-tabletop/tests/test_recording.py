"""Hardware-free round-trip tests for the demonstration collection tools.

Run with the project's lerobot interpreter:
    python -m unittest discover -s tests -p 'test_recording.py' -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ["MUJOCO_GL"] = "egl"

PROJECT = Path(__file__).resolve().parents[1]


def run_tool(script, *arguments):
    result = subprocess.run(
        [sys.executable, str(PROJECT / script), *map(str, arguments)],
        cwd=PROJECT,
        env={**os.environ, "MUJOCO_GL": "egl"},
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.returncode:
        raise AssertionError(
            f"{script} exited {result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    return result


class ObservationAlignmentTests(unittest.TestCase):
    def test_post_step_body_pose_matches_advanced_physics_state(self):
        os.environ["MUJOCO_GL"] = "egl"
        import env as tabletop
        import mujoco
        from mounts.environment import make_single_env

        simulation = make_single_env(
            observation_height=120, observation_width=160, terminate_on_success=False
        )
        try:
            simulation.reset(seed=0)
            world = simulation._ensure_world()
            body = world.model.body("red_cube").id
            joint = world.model.body_jntadr[body]
            address = world.model.jnt_qposadr[joint]
            world.data.qpos[address + 2] = 0.2
            mujoco.mj_forward(world.model, world.data)
            simulation.step(np.asarray(tabletop.HOME, dtype=np.float32))
            self.assertLess(world.data.qpos[address + 2], 0.2)
            np.testing.assert_allclose(
                world.data.xpos[body],
                world.data.qpos[address : address + 3],
                atol=1e-12,
            )
        finally:
            simulation.close()


class RecordingRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tabletop-recording-test-")
        cls.root = Path(cls.temporary.name) / "capture"
        run_tool(
            "record_demonstrations.py",
            "--root",
            cls.root,
            "--synthetic",
            "--headless",
            "--steps",
            6,
            "--episodes",
            2,
            "--width",
            160,
            "--height",
            120,
            "--discard-first",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_native_dataset_supports_single_step_and_action_chunks(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_root = self.root
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        fps = info["fps"]
        dataset = LeRobotDataset(
            repo_id="local/recording-tests",
            root=dataset_root,
            video_backend="pyav",
        )
        self.assertEqual(len(dataset), 12)
        self.assertEqual(dataset.num_episodes, 2)
        first = dataset[0]
        self.assertEqual(tuple(first["observation.state"].shape), (6,))
        self.assertEqual(tuple(first["action"].shape), (6,))
        for camera in ("overhead", "wrist"):
            image = first[f"observation.images.{camera}"]
            self.assertEqual(tuple(image.shape), (3, 120, 160))
            self.assertTrue(np.isfinite(image.numpy()).all())
            self.assertGreater(float(image.std()), 0)

        chunks = LeRobotDataset(
            repo_id="local/recording-tests",
            root=dataset_root,
            video_backend="pyav",
            delta_timestamps={"action": [0, 1 / fps, 2 / fps]},
        )
        last_of_first_episode = chunks[5]
        self.assertEqual(tuple(last_of_first_episode["action"].shape), (3, 6))
        self.assertEqual(
            last_of_first_episode["action_is_pad"].tolist(), [False, True, True]
        )

    def test_saved_archive_labels_and_terminal_states(self):
        episode_paths = sorted((self.root / "sim_archive" / "episodes").glob("*.json"))
        self.assertEqual(
            len(episode_paths), 2, "Discarded attempts must not become dataset episodes"
        )
        for path in episode_paths:
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["status"], "committed")
            self.assertEqual(metadata["metadata"]["source"], "synthetic")
            self.assertEqual(metadata["metadata"]["outcome"], "synthetic")
            self.assertEqual(metadata["transition_count"], 6)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
                self.assertEqual(archive["states"].shape[0], 7)
                self.assertEqual(archive["raw_actions"].shape, (6, 6))
                self.assertEqual(archive["applied_actions"].shape, (6, 6))
                self.assertEqual(archive["controls"].shape, (6, 6))
                self.assertEqual(archive["raw_actions"][0, 0], 999)
                self.assertLess(archive["applied_actions"][0, 0], 180)
                self.assertAlmostEqual(
                    float(np.deg2rad(archive["applied_actions"][0, 0])),
                    float(archive["controls"][0, 0]),
                    places=5,
                )
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype, np.dtype(object), name)
                    self.assertTrue(np.isfinite(archive[name]).all(), name)

    def test_validation_and_command_replay(self):
        run_tool(
            "replay_demonstrations.py", "--root", self.root, "--validate", "--headless"
        )
        run_tool(
            "replay_demonstrations.py",
            "--root",
            self.root,
            "--replay",
            "commands",
            "--headless",
        )

    def test_pending_archive_is_refused(self):
        path = next((self.root / "sim_archive/episodes").glob("*.json"))
        original = path.read_text()
        manifest = json.loads(original)
        manifest["status"] = "pending"
        try:
            path.write_text(json.dumps(manifest))
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "replay_demonstrations.py"),
                    "--root",
                    str(self.root),
                    "--validate",
                    "--headless",
                ],
                cwd=PROJECT,
                text=True,
                capture_output=True,
                timeout=180,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pending", (result.stdout + result.stderr).lower())
        finally:
            path.write_text(original)

    def test_rerender_preserves_episode_provenance(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        destination = Path(self.temporary.name) / "rerendered"
        overrides = Path(self.temporary.name) / "camera-overrides.json"
        overrides.write_text(json.dumps({"overhead": {"fovy": 75}}))
        run_tool(
            "replay_demonstrations.py",
            "--root",
            self.root,
            "--rerender",
            destination,
            "--headless",
            "--camera-overrides",
            overrides,
        )
        source = {
            json.loads(path.read_text())["episode_uuid"]
            for path in (self.root / "sim_archive" / "episodes").glob("*.json")
        }
        derived = {
            json.loads(path.read_text())["episode_uuid"]
            for path in (destination / "sim_archive" / "episodes").glob("*.json")
        }
        self.assertEqual(source, derived)
        run_tool(
            "replay_demonstrations.py",
            "--root",
            destination,
            "--validate",
            "--headless",
        )
        original = LeRobotDataset(
            repo_id="local/recording-tests", root=self.root, video_backend="pyav"
        )
        changed = LeRobotDataset(
            repo_id="local/recording-tests", root=destination, video_backend="pyav"
        )
        np.testing.assert_array_equal(
            original[0]["action"].numpy(), changed[0]["action"].numpy()
        )
        np.testing.assert_array_equal(
            original[0]["observation.state"].numpy(),
            changed[0]["observation.state"].numpy(),
        )
        mean_pixel_change = (
            (
                original[0]["observation.images.overhead"]
                - changed[0]["observation.images.overhead"]
            )
            .abs()
            .mean()
        )
        self.assertGreater(
            float(mean_pixel_change),
            0.005,
            "Camera override must change the rendered view",
        )


class RecordingLifecycleTests(unittest.TestCase):
    def test_resume_and_discard_unaccepted_attempt(self):
        import env as tabletop
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from mounts.environment import make_single_env
        from recording.session import RecordingSession

        with tempfile.TemporaryDirectory(
            prefix="tabletop-lifecycle-test-"
        ) as directory:
            root = Path(directory) / "capture"
            simulation = make_single_env(
                observation_height=120,
                observation_width=160,
                terminate_on_success=False,
            )
            try:
                first = RecordingSession(simulation, root, source="synthetic")
                try:
                    first.begin()
                    first.step(np.asarray(tabletop.HOME, dtype=np.float32))
                    first.finish()
                    first.save("synthetic")
                    first.reset()
                    first.begin()
                    first.step(np.asarray(tabletop.HOME, dtype=np.float32))
                    # Closing with an active attempt is the graceful interrupt
                    # cleanup path; it must not commit that partial episode.
                finally:
                    first.close()
                partials = list((root / "sim_archive" / "episodes").glob("*.json"))
                self.assertEqual(len(partials), 1)

                # Resume must not decode prior recordings; Keep already did that.
                from unittest.mock import patch

                with patch(
                    "recording.replay.validate",
                    side_effect=AssertionError("resume decoded old episodes"),
                ):
                    resumed = RecordingSession(
                        simulation, root, resume=True, source="synthetic"
                    )
                try:
                    resumed.begin()
                    resumed.step(np.asarray(tabletop.HOME, dtype=np.float32))
                    resumed.finish()
                    resumed.save("synthetic")
                finally:
                    resumed.close()

                dataset = LeRobotDataset(
                    repo_id="local/recording-tests", root=root, video_backend="pyav"
                )
                self.assertEqual(dataset.num_episodes, 2)
                self.assertEqual(len(dataset), 2)
                indices = {
                    json.loads(path.read_text())["episode_index"]
                    for path in (root / "sim_archive" / "episodes").glob("*.json")
                }
                self.assertEqual(indices, {0, 1})
                self.assertFalse((root / "COLLECTION_INCOMPLETE.json").exists())
                from recording.replay import validate

                (root / "COLLECTION_INCOMPLETE.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Incomplete collection"):
                    validate(root)
            finally:
                simulation.close()


if __name__ == "__main__":
    unittest.main()
