"""Check native camera projection, measured frame conversion and CAD wrist roll."""

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import env
import mujoco
import numpy as np
from mounts.cameras import apply_intrinsics
from mounts.environment import build_variant
from mounts.wrist import configure_wrist

ROOT = Path(__file__).resolve().parents[1]


class CameraCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "mounts/config.json").read_text())
        root = build_variant()
        cls.model = env.build_model(root / "scene.xml", root)
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def test_native_frustum_matches_each_full_opencv_matrix(self):
        for name, section in [("overhead", "overhead_camera"), ("wrist_cam", "wrist")]:
            with self.subTest(camera=name):
                measured = self.config[section]["intrinsics_640x480"]
                K = np.array(measured["camera_matrix"])
                w, h = measured["width"], measured["height"]
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = self.model.camera(name).id
                scene = mujoco.MjvScene(self.model, maxgeom=1000)
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    mujoco.MjvOption(),
                    None,
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    scene,
                )
                gl = scene.camera[0]
                near, half_width = gl.frustum_near, gl.frustum_width
                left = gl.frustum_center - half_width
                # Points are in OpenCV camera coordinates (right, down, forward).
                for x, y, z in [
                    (0, 0, 1),
                    (0.1, 0.08, 0.5),
                    (-0.2, -0.1, 0.8),
                    (0.2, -0.15, 0.6),
                ]:
                    actual = np.array(
                        [
                            w * (near * x / z - left) / (2 * half_width),
                            h
                            * (gl.frustum_top + near * y / z)
                            / (gl.frustum_top - gl.frustum_bottom),
                        ]
                    )
                    expected = np.array(
                        [K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]]
                    )
                    np.testing.assert_allclose(actual, expected, atol=0.0002, rtol=0)

    def test_overhead_world_pose_matches_sim2real_refit(self):
        refit = json.loads(
            (ROOT / "mounts/calibration/overhead_extrinsics_refit.json").read_text()
        )
        cid = self.model.camera("overhead").id
        np.testing.assert_allclose(
            self.data.cam_xpos[cid], refit["world_position_m"], atol=1e-6
        )
        R = self.data.cam_xmat[cid].reshape(3, 3)
        np.testing.assert_allclose(R[:, :2].T.reshape(-1), refit["xyaxes"], atol=1e-6)

    def test_refit_records_the_replaced_charuco_conversion(self):
        source = json.loads(
            (ROOT / "mounts/calibration/overhead_extrinsics.json").read_text()
        )
        provenance = json.loads(
            (ROOT / "mounts/calibration/provenance.json").read_text()
        )
        refit = json.loads(
            (ROOT / "mounts/calibration/overhead_extrinsics_refit.json").read_text()
        )
        expected = np.array(source["mujoco"]["pos"]) + np.array(
            provenance["calibration_world_to_current_world_translation_m"]
        )
        np.testing.assert_allclose(
            refit["replaced"]["world_position_m"], expected, atol=1e-10
        )
        np.testing.assert_allclose(
            refit["replaced"]["xyaxes"], source["mujoco"]["xyaxes"], atol=1e-10
        )

    def test_wrist_grasp_frame_is_below_image_centre_and_pose_stays_cad(self):
        cid = self.model.camera("wrist_cam").id
        R = self.data.cam_xmat[cid].reshape(3, 3)
        point = R.T @ (
            self.data.site_xpos[self.model.site("gripperframe").id]
            - self.data.cam_xpos[cid]
        )
        measured = self.config["wrist"]["intrinsics_640x480"]
        K = np.array(measured["camera_matrix"])
        self.assertLess(point[2], 0)
        v = -K[1, 1] * point[1] / (-point[2]) + K[1, 2]
        self.assertTrue(measured["height"] / 2 < v < measured["height"])
        config = dict(self.config["wrist"])
        config.pop("intrinsics_640x480")
        asset = ROOT / "mounts/assets/SO-ARM101_camera_wrist_mount.stl"
        plain = configure_wrist(
            ET.parse(ROOT / "robotstudio_so101/so101.xml").getroot(), asset, config
        )
        calibrated = configure_wrist(
            ET.parse(ROOT / "robotstudio_so101/so101.xml").getroot(),
            asset,
            self.config["wrist"],
        )
        np.testing.assert_array_equal(plain["camera_pos"], calibrated["camera_pos"])
        np.testing.assert_array_equal(plain["camera_quat"], calibrated["camera_quat"])

    def test_each_active_K_and_D_matches_its_preserved_source(self):
        for key, name in [
            ("overhead_camera", "overhead_intrinsics.json"),
            ("wrist", "wrist_intrinsics.json"),
        ]:
            source = json.loads((ROOT / "mounts/calibration" / name).read_text())
            for field in ["camera_matrix", "distortion"]:
                np.testing.assert_array_equal(
                    self.config[key]["intrinsics_640x480"][field], source[field]
                )

    def test_reject_invalid_intrinsics(self):
        measured = dict(self.config["wrist"]["intrinsics_640x480"])
        measured["camera_matrix"] = [[0, 0, 320], [0, 545, 240], [0, 0, 1]]
        with self.assertRaises(ValueError):
            apply_intrinsics(ET.Element("camera"), measured)


if __name__ == "__main__":
    unittest.main()
