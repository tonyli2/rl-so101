"""Stock hex-nut wrist bracket adapter; nominal CAD fit, not camera calibration."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

SOURCE_REVISION = "141aea7bcac2744d92998e30268c413202a56f96"
SOURCE_PATH = "Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/stl/SO-ARM101_camera_wrist_mount.stl"


def _text(values):
    return " ".join(f"{float(value):.9g}" for value in values)


def _quat(matrix):
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix).reshape(9))
    return quaternion


def configure_wrist(arm_root: ET.Element, asset_path: Path, config=None) -> dict:
    """Replace the upstream camera_mount, keeping the named wrist camera.

    Config mesh_pos/mesh_quat are a rigid transform from source print coordinates
    (after mm->m scaling) to the gripper body. camera_roll_degrees rotates the
    optical image about its axis. camera_plate_side selects which physical face
    carries the PCB/lens; -1 uses the opposite face. Transforms remain nominal.
    """
    config = config or {}
    asset = arm_root.find("asset")
    gripper = arm_root.find(".//body[@name='gripper']")
    if asset is None or gripper is None:
        raise ValueError("Expected the upstream SO-101 arm asset and gripper body")
    old = gripper.find("body[@name='camera_mount']")
    if old is not None:
        gripper.remove(old)

    # The broad PCB plate is identifiable directly in the source STL. Its plane
    # normal and centre are not a calibrated lens centre or measured assembly.
    plate_normal = np.array([-0.4226182617, -0.9063077870, 0.0])
    # Source flange-hole axes are X; their centres lie at Y=-8.15mm,
    # Z=10/18mm. In the gripper body the corresponding side holes run along Y,
    # centred X=3.1/-5.0mm, Y=24.000214mm, Z=-23.399869mm.
    # Align hole midpoints (8mm vs 8.1mm source spacing), not old camera pose.
    # Flip 180 degrees about the screw-axis line through its midpoint, keeping
    # the same flange registration. The camera belongs on the opposite face.
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    quaternion = np.asarray(config.get("mesh_quat", _quat(rotation)), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    rotation_flat = np.zeros(9)
    mujoco.mju_quat2Mat(rotation_flat, quaternion)
    rotation = rotation_flat.reshape(3, 3)
    nominal_pos = np.array([-0.00095, 0.024000214, -0.023399869]) - rotation @ np.array(
        [0, -0.00815, 0.014]
    )
    position = np.asarray(config.get("mesh_pos", nominal_pos), dtype=float)
    plate_center = np.asarray(
        config.get("plate_center", [-0.047976, -0.038755, 0.0175]), dtype=float
    )
    plate_side = float(config.get("camera_plate_side", -1))
    if plate_side not in (-1, 1):
        raise ValueError("camera_plate_side must be -1 or 1")
    # Parallel stock STL plate faces have normal coordinates 55.4 and 51.2mm.
    if plate_side < 0:
        plate_center = plate_center - plate_normal * float(
            config.get("plate_thickness", 0.0042)
        )
    optical_direction = plate_side * (rotation @ plate_normal)
    optical_direction /= np.linalg.norm(optical_direction)
    board_center = position + rotation @ plate_center + optical_direction * 0.001

    # Camera looks along its -Z. The PCB's 32mm axes share its image axes.
    right = -plate_side * (rotation @ np.array([0, 0, 1.0]))
    camera_z = -optical_direction
    camera_y = np.cross(camera_z, right)
    camera_matrix = np.column_stack((right, camera_y, camera_z))
    angle = np.deg2rad(float(config.get("camera_roll_degrees", 0)))
    camera_matrix = camera_matrix @ np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    camera_quat = _quat(camera_matrix)
    lens_center = board_center + optical_direction * 0.005
    camera_pos = board_center + optical_direction * 0.010

    # Reuse the servo housing material, including its colour and highlights.
    material = "sts3215_03a_v1_material"
    ET.SubElement(
        asset,
        "mesh",
        name="hexnut_wrist_mount",
        file=str(Path(asset_path).resolve()),
        scale="0.001 0.001 0.001",
    )
    body = ET.SubElement(gripper, "body", name="camera_mount")
    ET.SubElement(
        body,
        "geom",
        name="hexnut_wrist_mount_visual",
        type="mesh",
        mesh="hexnut_wrist_mount",
        pos=_text(position),
        quat=_text(quaternion),
        material=material,
        group="2",
        contype="0",
        conaffinity="0",
        mass=str(config.get("mount_mass", 0.012)),
    )
    ET.SubElement(
        body,
        "geom",
        name="wrist_camera_pcb",
        type="box",
        size="0.016 0.016 0.0005",
        pos=_text(board_center),
        quat=_text(camera_quat),
        material=material,
        group="2",
        contype="0",
        conaffinity="0",
        mass=str(config.get("camera_mass", 0.005)),
    )
    ET.SubElement(
        body,
        "geom",
        name="wrist_camera_lens",
        type="cylinder",
        size="0.007 0.0045",
        pos=_text(lens_center),
        quat=_text(camera_quat),
        material=material,
        group="2",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    ET.SubElement(
        body,
        "geom",
        name="wrist_camera_collision",
        type="box",
        size="0.016 0.016 0.006",
        pos=_text(lens_center),
        quat=_text(camera_quat),
        group="3",
        mass="0",
    )
    ET.SubElement(
        body,
        "camera",
        name="wrist_cam",
        mode="fixed",
        pos=_text(camera_pos),
        quat=_text(camera_quat),
        fovy=str(config.get("fovy", 61)),
    )
    return {
        "source_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "mesh_pos": position.tolist(),
        "mesh_quat": quaternion.tolist(),
        "camera_pos": camera_pos.tolist(),
        "camera_quat": camera_quat.tolist(),
        "camera_roll_degrees": float(config.get("camera_roll_degrees", 0)),
        "camera_plate_side": plate_side,
        "optical_direction": optical_direction.tolist(),
        "status": "Nominal mesh assembly and lens offset; not measured or calibrated",
        "mass_assumption_kg": {
            "mount": config.get("mount_mass", 0.012),
            "camera": config.get("camera_mass", 0.005),
        },
    }
