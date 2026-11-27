"""Compare the sim overhead camera (as configured in mounts/config.json) with the real one.

Uses saved captures (sim2real_capture.py / calib_poses.py runs) and, with --live, a fresh frame from
the real camera at the arm's current pose (read-only: torque is not touched). For each frame it
scores the overlap (IoU) of the real dark-arm silhouette with the sim arm segmentation, for the
configured camera and, for reference, the replaced ChArUco pose, and writes overlay images.

    python compare_overhead.py --live --overhead 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if os.uname().sysname == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import cv2
import mujoco
import numpy as np

from envs import So101World, prepare_mounted
from run_real import SIM_MINUS_REAL_DEG

ROOT = Path(__file__).resolve().parent.parent
REFIT = ROOT / "lerobot-tabletop/mounts/calibration/overhead_extrinsics_refit.json"
H, W = 240, 320


def xyaxes_quat(xyaxes) -> np.ndarray:
    x, y = np.asarray(xyaxes[:3], float), np.asarray(xyaxes[3:], float)
    R = np.column_stack([x, y, np.cross(x, y)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def frames_from_runs(runs: list[Path]):
    """(label, overhead BGR image, sim-convention joints) from saved captures."""
    off = np.asarray(SIM_MINUS_REAL_DEG, float)
    for run in runs:
        if (run / "capture.json").exists():
            cap = json.loads((run / "capture.json").read_text())
            for r in cap["A"]:
                yield f"{run.name}/{r['tag']}", cv2.imread(str(run / f"{r['tag']}_overhead.png")), np.asarray(r["measured_real"]) + off
        elif (run / "poses.json").exists():
            for k, r in enumerate(json.loads((run / "poses.json").read_text())):
                img = cv2.imread(str(run / f"{k}_{r['pose']}_overhead_real.jpg"))
                yield f"{run.name}/{k}_{r['pose']}", img, np.asarray(r["measured"], float) + off


def live_frame(port: str, index: int):
    from run_real import Camera, make_robot, read_deg

    cam = Camera(index, "overhead")
    robot = make_robot(port, None)
    robot.bus.connect()  # no configure(): torque stays as it is
    try:
        deg = read_deg(robot)
        time.sleep(0.3)
        raw, _ = cam.read()
    finally:
        robot.bus.disconnect(disable_torque=False)
        cam.close()
    return "live", cv2.undistort(raw, cam.K, cam.D, None, cam.K), deg + np.asarray(SIM_MINUS_REAL_DEG, float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", type=Path, default=[
        Path("outputs/sim2real/20261126-211901"), Path("outputs/calib_poses/20261126-205627"),
        Path("outputs/calib_poses/20261126-210015")])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--overhead", type=int, default=0)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--out", type=Path, default=Path("outputs/compare_overhead") / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    world = So101World()
    m, d = world.model, world.data
    cid = m.camera("overhead").id
    configured = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
    replaced = json.loads(REFIT.read_text())["replaced"]
    old = (np.asarray(replaced["pos"], float), xyaxes_quat(replaced["xyaxes"]))
    seg = mujoco.Renderer(m, H, W)
    seg.enable_segmentation_rendering()
    color = mujoco.Renderer(m, 480, 640)
    arm = list(range(m.body("shoulder").id, m.body("camera_mount").id + 1))
    args.out.mkdir(parents=True, exist_ok=True)

    frames = list(frames_from_runs(args.runs))
    if args.live:
        frames.append(live_frame(args.port, args.overhead))
    rows, tiles = [], []
    for label, img, deg in frames:
        world.reset(world.deg_to_ctrl(deg))
        small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        real = cv2.morphologyEx((cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2] < 60).astype(np.uint8),
                                cv2.MORPH_OPEN, np.ones((2, 2))).astype(bool)
        scores = {}
        overlay = img.copy()
        for name, (pos, quat), col in (("replaced", old, (0, 0, 255)), ("configured", configured, (0, 220, 0))):
            m.cam_pos[cid], m.cam_quat[cid] = pos, quat
            mujoco.mj_forward(m, d)
            seg.update_scene(d, camera="overhead", scene_option=world._scene_option)
            s = seg.render()
            ok = (s[..., 0] >= 0) & (s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
            sim = np.isin(np.where(ok, m.geom_bodyid[np.clip(s[..., 0], 0, None)], -1), arm)
            scores[name] = float((real & sim).sum() / max(1, (real | sim).sum()))
            big = cv2.resize(sim.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
            cnt, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, cnt, -1, col, 2)
        m.cam_pos[cid], m.cam_quat[cid] = configured
        mujoco.mj_forward(m, d)
        color.update_scene(d, camera="overhead", scene_option=world._scene_option)
        sim_img = cv2.cvtColor(color.render(), cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(img, 0.5, sim_img, 0.5, 0)
        cv2.putText(overlay, f"{label}  green=configured {scores['configured']:.2f}  red=replaced {scores['replaced']:.2f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(str(args.out / f"{label.replace('/', '__')}.jpg"), np.hstack([overlay, sim_img, blend]))
        rows.append({"frame": label, **scores})
        tiles.append(cv2.resize(overlay, (320, 240)))
    (args.out / "scores.json").write_text(json.dumps(rows, indent=1))
    pick = [t for t in tiles[:: max(1, len(tiles) // 8)]][:8]
    if args.live:
        pick[-1] = tiles[-1]
    cv2.imwrite(str(args.out / "summary.jpg"), np.vstack([np.hstack(pick[:4]), np.hstack(pick[4:8])]))
    for group in sorted({r["frame"].split("/")[0] for r in rows}):
        g = [r for r in rows if r["frame"].split("/")[0] == group]
        print(f"{group:28s} {len(g):2d} frames  arm IoU  configured {np.mean([r['configured'] for r in g]):.3f} "
              f"(min {min(r['configured'] for r in g):.2f})   replaced ChArUco {np.mean([r['replaced'] for r in g]):.3f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
