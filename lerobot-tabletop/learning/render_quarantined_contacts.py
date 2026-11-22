"""Render representative quarantined demonstrations by exact colliding geometry pair."""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from recording.archive import load_episode_archive, restore_state  # noqa: E402

from learning.render_contact_comparison import (  # noqa: E402
    BLUE,
    ORANGE,
    RED,
    box_outline,
    text,
)

GROUPS = [
    dict(
        id="gripper_holder",
        episode=24,
        geoms=[29, 47],
        bodies=["shoulder", "gripper"],
        title="Fixed gripper jaw / shoulder holder",
        members=[5, 6, 11, 19, 22, 24, 31, 38, 39, 49],
    ),
    dict(
        id="gripper_servo",
        episode=33,
        geoms=[27, 47],
        bodies=["shoulder", "gripper"],
        title="Fixed gripper jaw / shoulder servo",
        members=[33],
    ),
    dict(
        id="forearm_servo",
        episode=25,
        geoms=[27, 36],
        bodies=["shoulder", "lower_arm"],
        title="Forearm / shoulder servo",
        members=[23, 25, 29, 32, 34, 45, 47, 49],
    ),
    dict(
        id="camera_holder",
        episode=29,
        geoms=[29, 70],
        bodies=["shoulder", "camera_mount"],
        title="Wrist camera / shoulder holder",
        members=[29],
        note="This demonstration also has forearm/shoulder contact.",
    ),
]


def pair_contacts(model, data, gids):
    found = []
    for index, contact in enumerate(data.contact):
        if set(map(int, contact.geom)) == set(gids) and contact.dist <= 0:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            found.append(
                dict(
                    position=contact.pos.copy(),
                    penetration=float(-contact.dist),
                    force=float(force[0]),
                )
            )
    return found


def make_camera(lookat, zoom=False, azimuth=90):
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = azimuth, -24
    cam.lookat[:] = lookat if zoom else [0.085, 0, 0.13]
    cam.distance = 0.20 if zoom else 0.57
    return cam


def project(scene, point, width, height):
    gl = scene.camera[0]
    eye = (scene.camera[0].pos + scene.camera[1].pos) / 2
    forward, up = np.asarray(gl.forward), np.asarray(gl.up)
    right = np.cross(forward, up)
    focal = height * float(gl.frustum_near) / float(gl.frustum_top - gl.frustum_bottom)
    ray = np.asarray(point) - eye
    depth = float(ray @ forward)
    if depth <= 0:
        return None
    return width / 2 + float(ray @ right) / depth * focal, height / 2 - float(
        ray @ up
    ) / depth * focal


def render_view(
    renderer,
    model,
    data,
    gids,
    lookat,
    zoom=False,
    azimuth=90,
    geometry=False,
    points=(),
):
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    renderer.update_scene(
        data, camera=make_camera(lookat, zoom, azimuth), scene_option=opt
    )
    scene = renderer.scene
    body_ids = [int(model.geom_bodyid[g]) for g in gids]
    for geom in scene.geoms[: scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        gid = geom.objid
        body = int(model.geom_bodyid[gid])
        if model.body(body).name == "overhead_stand":
            geom.rgba[3] = 0
            continue
        if int(model.geom_group[gid]) >= 3:
            if geometry and gid in gids:
                color = BLUE if gid == gids[0] else ORANGE
                geom.rgba[:] = [*(np.asarray(color) / 255), 0.28]
            else:
                geom.rgba[3] = 0
            continue
        if body in body_ids:
            color = BLUE if body == body_ids[0] else ORANGE
            geom.rgba[:] = [*(np.asarray(color) / 255), 0.35 if geometry else 1.0]
        elif geometry and body != 0:
            geom.rgba[3] = 0.10
    if geometry:
        for gid, color in zip(gids, (BLUE, ORANGE)):
            if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                box_outline(scene, model, data, gid, [*(np.asarray(color) / 255), 1.0])
    image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image)
    if points:
        # Ring drawn after rendering stays visible even when the physical surface occludes it.
        target = min(points, key=lambda x: -x["penetration"])
        pixel = project(scene, target["position"], renderer.width, renderer.height)
        if pixel:
            px, py = pixel
            radius = 14 if zoom else 10
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                outline="white",
                width=6,
            )
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                outline=RED,
                width=3,
            )
            if zoom:
                lx, ly = 680, 45
                draw.line((lx, ly + 18, px + radius, py), fill="white", width=6)
                draw.line((lx, ly + 18, px + radius, py), fill=RED, width=3)
                draw.rectangle((lx, ly, lx + 240, ly + 38), fill=(100, 15, 20))
                text(draw, (lx + 8, ly + 4), "CONTACT HERE", 24)
    return image


def composite(
    model, data, renderer, close, group, lookat, index, azimuth=90, slow=False
):
    contacts = pair_contacts(model, data, group["geoms"])
    frame = Image.new("RGB", (1920, 1080), (13, 20, 30))
    draw = ImageDraw.Draw(frame)
    text(draw, (20, 10), group["title"], 30)
    text(
        draw,
        (20, 47),
        f"Original demonstration {group['episode']} | recorded time {index / 30:.2f} s",
        23,
        (185, 198, 215),
    )
    text(draw, (980, 10), "Exact collision shapes + transparent surfaces", 28)
    text(
        draw,
        (980, 47),
        "0.25x slow motion" if slow else "Recorded motion at 1x",
        23,
        (185, 198, 215),
    )
    for column, geometry in enumerate((False, True)):
        x = column * 960
        frame.paste(
            render_view(
                renderer,
                model,
                data,
                group["geoms"],
                lookat,
                False,
                azimuth,
                geometry,
                contacts,
            ),
            (x, 80),
        )
        frame.paste(
            render_view(
                close,
                model,
                data,
                group["geoms"],
                lookat,
                True,
                azimuth,
                geometry,
                contacts,
            ),
            (x, 600),
        )
        draw = ImageDraw.Draw(frame)
        text(draw, (x + 16, 608), "Close-up of the same instant", 23)
        if contacts:
            status = f"CONTACT | penetration {max(c['penetration'] for c in contacts) * 1000:.3f} mm"
            draw.rectangle((x + 15, 545, x + 700, 590), fill=(100, 15, 20))
        else:
            status = "This highlighted pair is clear"
            draw.rectangle((x + 15, 545, x + 510, 590), fill=(15, 80, 45))
        text(draw, (x + 25, 550), status, 23)
    draw.line((960, 80, 960, 960), fill=(230, 230, 230), width=3)
    text(
        draw,
        (20, 975),
        f"BLUE: {group['bodies'][0]}     ORANGE: {group['bodies'][1]}     RED RING: actual contact point",
        25,
    )
    footer = group.get(
        "note",
        "Both columns show identical saved states. Outlines are the simulator's collision boxes, not new obstacles.",
    )
    text(draw, (20, 1018), footer, 23, (185, 198, 215))
    return frame


def decode_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(stream))
        return dict(
            frames=count,
            fps=float(stream.average_rate),
            width=stream.width,
            height=stream.height,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--azimuth", type=float, default=90)
    parser.add_argument("--groups", nargs="+", default=[g["id"] for g in GROUPS])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((args.quarantine / "episode_migration.json").read_text())[
        "original_to_new"
    ]
    reports = []
    for group in GROUPS:
        if group["id"] not in args.groups:
            continue
        manifest, arrays, model = load_episode_archive(args.dataset, group["episode"])
        for gid, body in zip(group["geoms"], group["bodies"]):
            assert model.body(int(model.geom_bodyid[gid])).name == body
        data = mujoco.MjData(model)
        hit_frames = []
        positions = []
        depths = []
        for index, state in enumerate(arrays["states"]):
            restore_state(model, data, state)
            found = pair_contacts(model, data, group["geoms"])
            if found:
                hit_frames.append(index)
                positions.extend(p["position"] for p in found)
                depths.append((max(p["penetration"] for p in found), index))
        if not hit_frames:
            raise RuntimeError(f"No representative contact found: {group['id']}")
        start = max(0, min(hit_frames) - 24)
        end = min(len(arrays["states"]) - 1, max(max(hit_frames) + 45, start + 120))
        still = max(depths)[1]
        lookat = np.mean(positions, axis=0)
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 600
        full = mujoco.Renderer(model, 520, 960)
        close = mujoco.Renderer(model, 360, 960)
        prefix = f"{group['id']}_original_ep{group['episode']:02d}"
        reports.append(
            dict(
                group=group,
                quarantine_episode=int(mapping[str(group["episode"])]),
                episode_uuid=manifest["episode_uuid"],
                scene_id=manifest["scene_id"],
                source="Exact archived integration states; no physics changed",
                contact_frames=hit_frames,
                clip_start_state=start,
                clip_end_state=end,
                still_state=still,
                lookat=lookat.tolist(),
                azimuth=args.azimuth,
                collision_geometries=[
                    {
                        "id": g,
                        "name": model.geom(g).name,
                        "body": model.body(int(model.geom_bodyid[g])).name,
                        "type": int(model.geom_type[g]),
                        "size": model.geom_size[g].tolist(),
                    }
                    for g in group["geoms"]
                ],
            )
        )
        try:
            restore_state(model, data, arrays["states"][still])
            composite(
                model, data, full, close, group, lookat, still, args.azimuth
            ).save(args.output / (prefix + ".png"))
            if args.preview:
                print("PREVIEW", args.output / (prefix + ".png"), flush=True)
                continue
            paths = [
                args.output / (prefix + ".mp4"),
                args.output / (prefix + "_slow.mp4"),
            ]
            if any(p.exists() for p in paths):
                raise FileExistsError(paths)
            containers = [av.open(str(p), "w") for p in paths]
            streams = []
            for container in containers:
                stream = container.add_stream("libx264", rate=30)
                stream.width = 1920
                stream.height = 1080
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "19", "preset": "fast"}
                streams.append(stream)
            for index in range(start, end + 1):
                restore_state(model, data, arrays["states"][index])
                frame = composite(
                    model, data, full, close, group, lookat, index, args.azimuth
                )
                for container, stream, repeat in zip(containers, streams, (1, 4)):
                    output = frame
                    if repeat == 4:
                        output = frame.copy()
                        draw = ImageDraw.Draw(output)
                        draw.rectangle((980, 45, 1500, 77), fill=(13, 20, 30))
                        text(draw, (980, 47), "0.25x slow motion", 23, (185, 198, 215))
                    rgb = np.asarray(output)
                    for _ in range(repeat):
                        for packet in stream.encode(
                            av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        ):
                            container.mux(packet)
                if (index - start) % 30 == 0:
                    print("FRAME", group["id"], index, flush=True)
            for container, stream in zip(containers, streams):
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
            reports[-1]["validation"] = {p.name: decode_info(p) for p in paths}
            assert reports[-1]["validation"][paths[0].name]["frames"] == end - start + 1
            assert reports[-1]["validation"][paths[1].name]["frames"] == 4 * (
                end - start + 1
            )
            print(
                "COMPLETE",
                group["id"],
                json.dumps(reports[-1]["validation"]),
                flush=True,
            )
        finally:
            full.close()
            close.close()
    (
        args.output
        / ("preview_metadata.json" if args.preview else "group_metadata.json")
    ).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
