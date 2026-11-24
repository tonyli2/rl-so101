"""Build a mounted scene while reusing the upstream SO101Env implementation."""

import copy
import json
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import env as tabletop
import gymnasium as gym

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent
GENERATED = HERE / "generated"


def _numbers(values):
    return " ".join(str(value) for value in values)


def _absolute_assets(root, asset_directory):
    """Resolve source asset paths before moving generated XML elsewhere."""
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for node in root.findall("./asset/*"):
        filename = node.get("file")
        if filename:
            directory = meshdir if node.tag == "mesh" else texturedir
            node.set("file", str((asset_directory / directory / filename).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)


def _configure_task(scene, task):
    """Apply the single-cube task materials, geometry and initial positions."""
    asset = scene.find("asset")
    # A small colour texture suggests print layers without extra physics geometry.
    ET.SubElement(
        asset,
        "texture",
        name="pla_layers",
        type="cube",
        file=str(UPSTREAM / "textures" / "pla_layers.png"),
    )
    plastic = dict(
        texture="pla_layers",
        texuniform="true",
        specular="0.18",
        shininess="0.25",
        reflectance="0",
    )
    bin_material = asset.find("material[@name='box']")
    bin_material.attrib.clear()
    bin_material.attrib.update(name="box", rgba=_numbers(task["bin_rgba"]), **plastic)
    ET.SubElement(
        asset, "material", name="red_pla", rgba=_numbers(task["cube_rgba"]), **plastic
    )
    world = scene.find("worldbody")
    # Single fixed-position task; the original upstream scene remains available.
    for name in ("green_cube", "blue_cube", "yellow_cube"):
        world.remove(world.find(f"body[@name='{name}']"))
    cube = world.find("body[@name='red_cube']")
    cube.set("pos", _numbers(task["cube_position"]))
    cube.find("geom").set("material", "red_pla")
    cube.find("geom").set("mass", str(task["cube_mass_kg"]))
    bin_body = world.find("body[@name='box']")
    bin_body.set("pos", _numbers(task["box_position"]))
    # Butt the X-side walls against the Y-side walls instead of overlapping
    # their top faces at the corners. Inner/outer dimensions stay 90/98 mm.
    for wall in bin_body.findall("geom"):
        x, y, _ = map(float, wall.get("pos").split())
        if x != 0 and y == 0:
            wall.set("size", "0.002 0.045 0.0175")
    scene.find("./custom/text[@name='instruction']").set(
        "data", "Put the red cube in the grey box."
    )


def build_variant():
    """Generate MJCF from untouched upstream XML and the editable mount poses."""
    from .cameras import apply_intrinsics
    from .enclosure import add_enclosure
    from .wrist import configure_wrist

    config = json.loads((HERE / "config.json").read_text())
    scene = ET.parse(UPSTREAM / "so101_tabletop.xml").getroot()
    arm = ET.parse(UPSTREAM / "robotstudio_so101" / "so101.xml").getroot()
    _absolute_assets(scene, UPSTREAM)
    _absolute_assets(arm, UPSTREAM / "robotstudio_so101")

    asset = scene.find("asset")
    # Match the existing servo housing, including inherited highlight settings.
    servo_material = arm.find("./asset/material[@name='sts3215_03a_v1_material']")
    ET.SubElement(
        asset, "material", **{**servo_material.attrib, "name": "printed_mount_black"}
    )
    ET.SubElement(
        asset,
        "material",
        name="paper",
        rgba="0.96 0.96 0.96 1",
        specular="0.02",
        shininess="0.05",
    )
    _configure_task(scene, config["task"])
    world = scene.find("worldbody")
    stand = ET.SubElement(
        world, "body", name="overhead_stand", pos=_numbers(config["stand_position"])
    )
    for part in config["overhead_parts"]:
        name = part["name"]
        ET.SubElement(
            asset,
            "mesh",
            name=name,
            file=str(HERE / "assets" / part["file"]),
            scale="0.001 0.001 0.001",
        )
        ET.SubElement(
            stand,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            pos=_numbers(part["pos"]),
            quat=_numbers(part["quat"]),
            material="printed_mount_black",
            contype="0",
            conaffinity="0",
            mass="0",
            group="2",
        )
    # Only the straight upright is solid; printed feet/adapters remain visual-only.
    upright = config["upright_collision"]
    ET.SubElement(
        stand,
        "geom",
        name="overhead_upright_collision",
        type="box",
        pos=_numbers(upright["pos"]),
        size=_numbers(upright["size"]),
        contype="1",
        conaffinity="1",
        mass="0",
        group="3",
        rgba="1 0.5 0 0.3",
    )
    camera = config["overhead_camera"]
    overhead_camera = ET.SubElement(
        stand,
        "camera",
        name="overhead",
        mode="fixed",
        pos=_numbers(camera["pos"]),
        xyaxes=_numbers(camera["xyaxes"]),
        fovy=str(camera["fovy"]),
    )
    if "intrinsics_640x480" in camera:
        apply_intrinsics(overhead_camera, camera["intrinsics_640x480"])
    # PCB/lens proxies remain nominal CAD geometry; measured optical pose is separate.
    ET.SubElement(
        stand,
        "geom",
        name="overhead_camera_pcb",
        type="box",
        size="0.016 0.016 0.0005",
        pos=_numbers(camera["pcb_pos"]),
        quat=_numbers(camera["pcb_quat"]),
        rgba="0.08 0.15 0.08 1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    ET.SubElement(
        stand,
        "geom",
        name="overhead_camera_lens",
        type="cylinder",
        size="0.006 0.006",
        pos=_numbers(camera["lens_pos"]),
        quat=_numbers(camera["pcb_quat"]),
        rgba="0.025 0.025 0.025 1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    add_enclosure(scene, config)
    site = world.find("site[@name='so101_mount']")
    site.set("pos", _numbers(config["follower_position"]))
    site.set("quat", _numbers(config["follower_quaternion"]))
    configure_wrist(
        arm,
        HERE / "assets" / "SO-ARM101_camera_wrist_mount.stl",
        config.get("wrist", {}),
    )
    # Reuse the servo material for all visible arm parts. The upstream builder
    # preserves this material while recolouring other materials turquoise.
    for geom in arm.findall(".//geom[@material]"):
        geom.set("material", "sts3215_03a_v1_material")
    (GENERATED / "robotstudio_so101").mkdir(parents=True, exist_ok=True)
    for node, path in (
        (scene, GENERATED / "scene.xml"),
        (arm, GENERATED / "robotstudio_so101" / "so101.xml"),
    ):
        ET.indent(node)
        ET.ElementTree(node).write(path, encoding="unicode")
    return GENERATED


class MountedWorld(tabletop._World):
    """Recorded cameras see opaque paper; model alpha stays inspection-friendly."""

    def render(self, camera):
        from .enclosure import update_paper_scene

        update_paper_scene(self.renderer, self.model, self.data, camera)
        return self.renderer.render().copy()


class MountedSO101Env(tabletop.SO101Env):
    """Mounted single-cube task using upstream controls and rendering."""

    def _ensure_world(self):
        if self._world is None:
            self._world = MountedWorld(
                GENERATED / "scene.xml",
                GENERATED,
                self.fps,
                self.height,
                self.width,
                self.randomize,
                self.camera_fovy,
                cube_bodies=("red_cube",),
            )
        return self._world


def make_single_env(**kwargs):
    """Create one environment without vector-wrapper automatic resets."""
    build_variant()
    defaults = copy.deepcopy(tabletop.DEFAULTS)
    defaults["cameras"] = {"overhead": "overhead", "wrist": "wrist_cam"}
    # Repeatable pilot demonstrations: fixed cube, box, cameras and initial arm.
    defaults["randomize"] = {}
    return MountedSO101Env(**{**defaults, **kwargs})


def make_env(n_envs=1, use_async_envs=False, **kwargs):
    factory = partial(make_single_env, **kwargs)
    vector_class = (
        gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    )
    return {"so101_tabletop": {0: vector_class([factory] * n_envs)}}
