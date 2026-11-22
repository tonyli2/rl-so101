"""Render exact 90k/100k target replays with isometric and collision-shape views."""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from recording.archive import load_episode_archive, restore_state  # noqa: E402
from recording.schema import JOINTS  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BLUE = (64, 182, 255)
ORANGE = (255, 163, 60)
RED = (255, 70, 70)


def text(draw, xy, value, size=26, fill="white"):
    draw.text(xy, value, font=ImageFont.truetype(FONT, size), fill=fill)


def add_geom(scene, kind, size, pos, mat, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.asarray(size, float),
        np.asarray(pos, float),
        np.asarray(mat, float).ravel(),
        np.asarray(color, float),
    )
    scene.ngeom += 1
    return geom


def box_outline(scene, model, data, gid, color):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = (corners * model.geom_size[gid]) @ data.geom_xmat[gid].reshape(
        3, 3
    ).T + data.geom_xpos[gid]
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                g = add_geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    [0.0007, 0.001, 0],
                    vertices[i],
                    np.eye(3),
                    color,
                )
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0007, vertices[i], vertices[j]
                )


def camera(zoom):
    cam = mujoco.MjvCamera()
    cam.azimuth = 90
    cam.elevation = -24
    cam.distance = 0.21 if zoom else 0.58
    cam.lookat[:] = [0.042, 0.015, 0.105] if zoom else [0.085, 0, 0.13]
    return cam


def contacts(model, data, pair):
    found = []
    for i, c in enumerate(data.contact):
        if {int(model.geom_bodyid[g]) for g in c.geom} == pair:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1:
                found.append((c.pos.copy(), float(force[0]), tuple(map(int, c.geom))))
    return found


def render(renderer, model, data, shoulder, gripper, gids, zoom, points):
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    renderer.update_scene(data, camera=camera(zoom), scene_option=opt)
    scene = renderer.scene
    for g in scene.geoms[: scene.ngeom]:
        if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        body = int(model.geom_bodyid[g.objid])
        name = model.body(body).name
        if name == "overhead_stand":
            g.rgba[3] = 0
        if body == shoulder:
            g.rgba[:] = [0.15, 0.50, 0.90, 0.6 if zoom else 1]
        elif body == gripper:
            g.rgba[:] = [1.0, 0.46, 0.08, 0.6 if zoom else 1]
        elif zoom and body != 0:
            g.rgba[3] = 0.13
    if zoom:
        for gid, color in zip(gids, ([0.10, 0.70, 1.0, 0.25], [1.0, 0.56, 0.10, 0.25])):
            add_geom(
                scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                model.geom_size[gid],
                data.geom_xpos[gid],
                data.geom_xmat[gid],
                color,
            )
            box_outline(scene, model, data, gid, [*color[:3], 1.0])
    for pos, _, _ in points:
        add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.003, 0, 0],
            pos,
            np.eye(3),
            [1.0, 0.03, 0.03, 1.0],
        )
    image = Image.fromarray(renderer.render().copy())
    if points:
        gl = scene.camera[0]
        eye = (scene.camera[0].pos + scene.camera[1].pos) / 2
        forward = np.asarray(gl.forward, dtype=float)
        up = np.asarray(gl.up, dtype=float)
        right = np.cross(forward, up)
        focal = (
            renderer.height
            * float(gl.frustum_near)
            / float(gl.frustum_top - gl.frustum_bottom)
        )
        point = np.mean([p[0] for p in points], axis=0)
        ray = point - eye
        depth = float(ray @ forward)
        px = renderer.width / 2 + float(ray @ right) / depth * focal
        py = renderer.height / 2 - float(ray @ up) / depth * focal
        draw = ImageDraw.Draw(image)
        radius = 15 if zoom else 10
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            outline="white",
            width=5,
        )
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius), outline=RED, width=3
        )
        if zoom:
            label_x, label_y = 680, 65
            draw.line((label_x, label_y + 18, px + radius, py), fill="white", width=6)
            draw.line((label_x, label_y + 18, px + radius, py), fill=RED, width=3)
            draw.rectangle(
                (label_x, label_y, label_x + 230, label_y + 36), fill=(95, 10, 10)
            )
            text(draw, (label_x + 8, label_y + 3), "CONTACT HERE", 24)
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--trace90", type=Path, required=True)
    p.add_argument("--trace100", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, archive, model = load_episode_archive(args.dataset, 45)
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 600
    shoulder, gripper = model.body("shoulder").id, model.body("gripper").id
    pair = {shoulder, gripper}
    traces = [np.load(args.trace90), np.load(args.trace100)]
    datas = [mujoco.MjData(model), mujoco.MjData(model)]
    for d in datas:
        restore_state(model, d, archive["states"][0])
    actuators = np.array([model.actuator(n).id for n in JOINTS])
    qadr = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    lo, hi = model.actuator_ctrlrange[actuators].T
    # Identify the exact shoulder collision geom from a replayed contact.
    probe = mujoco.MjData(model)
    restore_state(model, probe, archive["states"][0])
    gids = None
    for target in traces[1]["applied"][:70]:
        ctrl = np.deg2rad(target.astype(float))
        ctrl[5] = lo[5] + target[5] / 100 * (hi[5] - lo[5])
        probe.ctrl[actuators] = np.clip(ctrl, lo, hi)
        mujoco.mj_step(model, probe, nstep=7)
        mujoco.mj_forward(model, probe)
        cs = contacts(model, probe, pair)
        if cs:
            gids = sorted(
                cs[0][2],
                key=lambda gid: 0 if int(model.geom_bodyid[gid]) == shoulder else 1,
            )
            break
    if gids is None:
        raise RuntimeError("Expected shoulder-gripper contact was not reproduced")
    full = mujoco.Renderer(model, 520, 960)
    close = mujoco.Renderer(model, 360, 960)
    errors = [0.0, 0.0]
    contact_frames = [[], []]
    writer = slow = None
    if not args.preview:
        writer = av.open(str(args.output / "90k_vs_100k_contact_isometric.mp4"), "w")
        slow = av.open(str(args.output / "90k_vs_100k_contact_isometric_slow.mp4"), "w")
        streams = []
        for container in (writer, slow):
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
            stream.options = {"crf": "19", "preset": "fast"}
            streams.append(stream)
    try:
        for k in range(151):
            if k:
                for i, d in enumerate(datas):
                    target = traces[i]["applied"][k - 1]
                    ctrl = np.deg2rad(target.astype(float))
                    ctrl[5] = lo[5] + target[5] / 100 * (hi[5] - lo[5])
                    d.ctrl[actuators] = np.clip(ctrl, lo, hi)
                    mujoco.mj_step(model, d, nstep=7)
                    mujoco.mj_forward(model, d)
            for i, d in enumerate(datas):
                q = np.rad2deg(d.qpos[qadr])
                q[5] = (d.qpos[qadr[5]] - lo[5]) / (hi[5] - lo[5]) * 100
                errors[i] = max(
                    errors[i], float(np.max(np.abs(q - traces[i]["positions"][k])))
                )
            if args.preview and k != 60:
                continue
            frame = Image.new("RGB", (1920, 1080), (13, 20, 30))
            draw = ImageDraw.Draw(frame)
            for i, d in enumerate(datas):
                cs = contacts(model, d, pair)
                if cs:
                    contact_frames[i].append(k)
                x = i * 960
                label = (
                    "90k TE — avoids this self-contact"
                    if i == 0
                    else "100k TE — shoulder/gripper contact"
                )
                text(draw, (x + 22, 10), label, 29)
                text(
                    draw,
                    (x + 22, 47),
                    f"Simulation time {k / 30:.2f} s",
                    24,
                    (180, 195, 215),
                )
                frame.paste(
                    render(full, model, d, shoulder, gripper, gids, False, cs), (x, 80)
                )
                frame.paste(
                    render(close, model, d, shoulder, gripper, gids, True, cs), (x, 600)
                )
                draw = ImageDraw.Draw(frame)
                text(
                    draw,
                    (x + 18, 608),
                    "Close-up: exact collision-box outlines",
                    23,
                    "white",
                )
                if cs:
                    draw.rectangle((x + 15, 545, x + 600, 590), fill=(100, 15, 20))
                    text(
                        draw,
                        (x + 25, 550),
                        f"CONTACT: fixed jaw presses shoulder ({sum(v[1] for v in cs):.0f} N)",
                        23,
                    )
                else:
                    draw.rectangle((x + 15, 545, x + 410, 590), fill=(15, 80, 45))
                    text(draw, (x + 25, 550), "No shoulder–gripper contact", 23)
            draw.line((960, 0, 960, 1080), fill=(230, 230, 230), width=3)
            text(
                draw,
                (20, 976),
                "BLUE: shoulder     ORANGE: gripper     RED DOT: actual MuJoCo contact point",
                26,
            )
            text(
                draw,
                (20, 1015),
                "Same archived start and physics. Different learned paths. Outlines show collision shapes, not added obstacles.",
                24,
                (185, 198, 215),
            )
            if k == 60:
                frame.save(args.output / "contact_at_2_seconds.png")
            if args.preview:
                print("PREVIEW", args.output / "contact_at_2_seconds.png", flush=True)
                break
            rgb = np.asarray(frame)
            for container, stream, repeat in (
                (writer, streams[0], 1),
                (slow, streams[1], 4),
            ):
                if repeat == 4:
                    slow_img = frame.copy()
                    sd = ImageDraw.Draw(slow_img)
                    sd.rectangle((1530, 44, 1900, 76), fill=(13, 20, 30))
                    text(sd, (1540, 45), "0.25x slow motion", 22)
                    arr = np.asarray(slow_img)
                else:
                    arr = rgb
                for _ in range(repeat):
                    for packet in stream.encode(
                        av.VideoFrame.from_ndarray(arr, format="rgb24")
                    ):
                        container.mux(packet)
            if k % 30 == 0:
                print("FRAME", k, "alignment", errors, flush=True)
        if writer:
            for container, stream in zip((writer, slow), streams):
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
        metadata = {
            "replay_max_position_error_deg_or_gripper_pp": errors,
            "contact_frames": contact_frames,
            "collision_geoms": [
                {
                    "id": int(g),
                    "name": model.geom(g).name,
                    "body": model.body(int(model.geom_bodyid[g])).name,
                    "size": model.geom_size[g].tolist(),
                }
                for g in gids
            ],
            "camera": {
                "azimuth": 90,
                "elevation": -24,
                "full_lookat": [0.085, 0, 0.13],
                "zoom_lookat": [0.042, 0.015, 0.105],
            },
            "source_frames": 151,
            "fps": 30,
            "slow_repeat": 4,
        }
        if not args.preview:
            assert max(errors) < 1e-4, (
                "Replay no longer matches captured joint positions"
            )
            (args.output / "isometric_video_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )
    finally:
        full.close()
        close.close()


if __name__ == "__main__":
    main()
