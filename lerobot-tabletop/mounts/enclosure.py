"""Measured paper panels: opaque sensor views, translucent inspection views."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

CURTAIN_GROUP = 5
CURTAIN_PREFIX = "paper_curtain_"
INSPECTION_ALPHA = 0.15


def _text(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def reference_origin(config):
    """Tabletop point at upright front face, centred across the pole width."""
    stand = np.asarray(config["stand_position"], float)
    pole = config["upright_collision"]
    return stand + np.array([pole["pos"][0] + pole["size"][0], pole["pos"][1], 0])


def add_enclosure(scene, config):
    """Three bounded thin boxes; reported positions are their interior faces.

    Measurement X maps to world -Y; measurement Y maps to world +X; Z is up.
    The model alpha is translucent for inspection. Sensor rendering overrides
    only the copied visualization geoms to opaque, never the physical model.
    """
    settings = config["paper_enclosure"]
    origin = reference_origin(config)
    # Reuse the fixed stand body: static panels need not change the integration
    # state layout or create additional rigid bodies.
    body = scene.find(".//body[@name='overhead_stand']")
    local_origin = origin - np.asarray(config["stand_position"])
    left, right, front, height = (
        settings[k] for k in ("left_x", "right_x", "front_y", "height")
    )
    half = settings["thickness"] / 2
    for name, position, size in (
        ("left", [left - half, front / 2, height / 2], [half, front / 2, height / 2]),
        ("right", [right + half, front / 2, height / 2], [half, front / 2, height / 2]),
        (
            "front",
            [(left + right) / 2, front + half, height / 2],
            [(right - left) / 2, half, height / 2],
        ),
    ):
        ET.SubElement(
            body,
            "geom",
            name=CURTAIN_PREFIX + name,
            type="box",
            pos=_text(
                local_origin + np.array([position[1], -position[0], position[2]])
            ),
            quat="0.7071067811865476 0 0 -0.7071067811865476",
            size=_text(size),
            material="paper",
            rgba=f"1 1 1 {INSPECTION_ALPHA}",
            group=str(CURTAIN_GROUP),
            contype="1",
            conaffinity="1",
            mass="0",
        )
    _cover_table(scene, origin, left, right, front)


def _cover_table(scene, origin, left, right, front):
    """Cover the enclosure footprint without changing the support height."""
    # White paper covers the existing support. Extend its footprint only as
    # needed to support the measured enclosure; the table's height is unchanged.
    table = scene.find("./worldbody/geom[@name='table']")
    position = np.fromstring(table.get("pos"), sep=" ")
    size = np.fromstring(table.get("size"), sep=" ")
    low, high = position[:2] - size[:2], position[:2] + size[:2]
    enclosure_low = origin[:2] + [0, -right]
    enclosure_high = origin[:2] + [front, -left]
    low, high = np.minimum(low, enclosure_low), np.maximum(high, enclosure_high)
    position[:2], size[:2] = (low + high) / 2, (high - low) / 2
    table.set("pos", _text(position))
    table.set("size", _text(size))
    table.set("material", "paper")


def update_paper_scene(renderer, model, data, camera, *, inspection=False):
    """Update only a renderer's scene; shared model/data stay unchanged."""
    option = mujoco.MjvOption()
    option.geomgroup[CURTAIN_GROUP] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        name = model.geom(geom.objid).name
        if name and name.startswith(CURTAIN_PREFIX):
            geom.rgba[3] = INSPECTION_ALPHA if inspection else 1.0
            geom.transparent = int(inspection)
