"""Render substep-resolved, validated replays of saved robustness-suite targets."""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from recording.archive import capture_state, restore_state
from recording.schema import JOINTS
from recording.self_collision import SelfCollisionChecker

from learning.render_contact_comparison import text
from learning.render_quarantined_contacts import decode_info, pair_contacts, render_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", default="030000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads((args.suite / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    with np.load(args.suite / "states.npz") as z:
        states = z["states"].copy()
        bins = z["bin_positions"].copy()
    for record_path in sorted(
        (args.results / "results" / args.checkpoint).glob("*.json")
    ):
        record = json.loads(record_path.read_text())
        if not record.get("primary") or record.get("first_contact") is None:
            continue
        case = next(c for c in suite["cases"] if c["id"] == record["scenario_id"])
        index = case["index"]
        model = mujoco.MjModel.from_binary_path(str(args.suite / "model.mjb"))
        data = mujoco.MjData(model)
        model.body_pos[model.body("box").id] = bins[index]
        restore_state(model, data, states[index])
        t0 = float(data.time)
        with np.load(record_path.with_suffix(".npz")) as z:
            targets = z["targets"].copy()
            positions = z["positions"].copy()
        ids = np.array([model.actuator(n).id for n in JOINTS])
        qadr = model.jnt_qposadr[[model.joint(n).id for n in JOINTS]]
        lo, hi = model.actuator_ctrlrange[ids].T
        checker = SelfCollisionChecker(model)
        first_t = record["first_contact"]["sim_time"] - t0
        begin = max(0, first_t - 0.25)
        end = first_t + 1.75
        gids = []
        for name in record["first_contact"]["geoms"]:
            gids.append(
                int(name[5:]) if name.startswith("geom_") else model.geom(name).id
            )
        snapshots = []
        max_error = 0.0
        contacts = []

        def collect():
            hit = checker.check(data)
            if hit:
                contacts.append({"elapsed_s": float(data.time) - t0, **hit})
            elapsed = float(data.time) - t0
            if begin - 1e-8 <= elapsed <= end + 1e-8:
                snapshots.append(
                    (
                        capture_state(model, data),
                        elapsed,
                        pair_contacts(model, data, gids),
                    )
                )

        def joint_state():
            q = data.qpos[qadr]
            value = np.rad2deg(q)
            value[5] = (q[5] - lo[5]) / (hi[5] - lo[5]) * 100
            return value.astype(np.float32)

        collect()
        max_error = float(np.max(np.abs(joint_state() - positions[0])))
        for k, target in enumerate(targets):
            ctrl = np.deg2rad(target.astype(np.float64))
            ctrl[5] = lo[5] + target[5] / 100 * (hi[5] - lo[5])
            data.ctrl[ids] = np.clip(ctrl, lo, hi)
            for sub in range(suite["substeps"]):
                mujoco.mj_step(model, data)
                collect()
            mujoco.mj_forward(model, data)
            # Keep final-forward detections too, but render only uniformly timed physics samples.
            hit = checker.check(data)
            if hit:
                contacts.append({"elapsed_s": float(data.time) - t0, **hit})
            max_error = max(
                max_error, float(np.max(np.abs(joint_state() - positions[k + 1])))
            )
        if max_error > 0.002:
            raise RuntimeError(f"Replay mismatch {case['id']}: {max_error}")
        if (
            not contacts
            or abs(contacts[0]["sim_time"] - record["first_contact"]["sim_time"])
            > model.opt.timestep * 1.1
        ):
            raise RuntimeError("First contact not reproduced")
        contact_frames = [(i, s) for i, s in enumerate(snapshots) if s[2]]
        if not contact_frames:
            raise RuntimeError("Missing contact in video window")
        peak_i, peak = max(
            contact_frames, key=lambda item: max(p["penetration"] for p in item[1][2])
        )
        lookat = np.mean([p["position"] for p in peak[2]], axis=0)
        path = args.output / f"{case['id']}_contact_isometric_slow.mp4"
        if path.exists():
            raise FileExistsError(path)
        scratch = mujoco.MjData(model)
        # Rendering-only framebuffer size; replay physics above uses the frozen model.
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 450
        renderer = mujoco.Renderer(model, height=450, width=960)
        output = av.open(str(path), "w")
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        try:
            for i, (state, elapsed, points) in enumerate(snapshots):
                restore_state(model, scratch, state)
                frame = Image.new("RGB", (1920, 1080), (13, 20, 30))
                draw = ImageDraw.Draw(frame)
                text(
                    draw,
                    (20, 8),
                    f"Clean ACT {int(args.checkpoint) // 1000}k | {case['id']} | t={elapsed:.3f}s",
                    28,
                )
                text(
                    draw,
                    (20, 46),
                    "Same replay in both columns | physics samples at 210 Hz shown at 30 fps (7x slower)",
                    22,
                )
                for column, geometry in enumerate((False, True)):
                    for row, zoom in enumerate((False, True)):
                        frame.paste(
                            render_view(
                                renderer,
                                model,
                                scratch,
                                gids,
                                lookat,
                                zoom,
                                90,
                                geometry,
                                points,
                            ),
                            (column * 960, 80 + row * 450),
                        )
                draw = ImageDraw.Draw(frame)
                text(draw, (24, 90), "Isometric surfaces", 24)
                text(draw, (984, 90), "Collision boxes + transparent surfaces", 24)
                text(
                    draw,
                    (24, 990),
                    "BLUE: shoulder | ORANGE: lower arm | RED RING: actual simulated contact",
                    24,
                )
                status = (
                    f"CONTACT: {max(p['penetration'] for p in points) * 1000:.3f} mm penetration"
                    if points
                    else "Highlighted pair currently clear"
                )
                text(draw, (24, 1030), status, 24)
                if i == peak_i:
                    frame.save(args.output / f"{case['id']}_peak_contact.png")
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
                ):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        finally:
            output.close()
            renderer.close()
        decoded = decode_info(path)
        assert (
            decoded["frames"] == len(snapshots)
            and decoded["width"] == 1920
            and decoded["height"] == 1080
        )
        report = {
            "scenario": case["id"],
            "checkpoint": args.checkpoint,
            "task_success": record["task_success"],
            "first_contact": record["first_contact"],
            "max_replay_position_error_native": max_error,
            "contact_samples": len(contacts),
            "last_contact_elapsed_s": max(c["elapsed_s"] for c in contacts),
            "all_contact_times_in_clip": all(begin - 1e-8 <= c["elapsed_s"] <= end + 1e-8 for c in contacts),
            "clip_start_s": begin,
            "clip_end_s": end,
            "slowdown": 7,
            "video": path.name,
            "decoded": decoded,
        }
        reports.append(report)
        (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
