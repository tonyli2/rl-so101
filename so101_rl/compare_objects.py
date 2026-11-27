"""Do the cube and bin line up between real and sim, in the overhead and wrist images?

For each frame (saved captures, and optionally a fresh read-only live frame) the sim is posed at the
measured joints with the cube and bin at their configured (measured-layout) positions, and the real
cube (red) and bin (grey) silhouettes are compared with the sim's: IoU, and centroid offset in
pixels. Overhead offsets are also converted to centimetres on the table through the configured
camera. Pixels covered by the arm in sim are ignored.

    python compare_objects.py --live
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

H, W = 480, 640
INTR = Path(__file__).resolve().parent.parent / "lerobot-tabletop/mounts/calibration"


def real_masks(img, window=None):
    """Real cube / bin silhouettes. ``window``: bool mask restricting the search (around the sim
    objects), so the grey wall or curtain is never taken for the bin."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if window is not None:
        hsv = hsv.copy()
        hsv[~window] = (0, 0, 255)  # white: neither red nor grey
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    cube = ((s > 110) & ((h < 8) | (h > 172)) & (v > 70)).astype(np.uint8)
    cube = cv2.morphologyEx(cube, cv2.MORPH_OPEN, np.ones((7, 7)))  # drop thin red servo wires
    n, lab, st, _ = cv2.connectedComponentsWithStats(cube)
    cube = np.zeros(cube.shape, bool)
    if n > 1:
        i = max(range(1, n), key=lambda k: st[k, 4])
        if st[i, 4] > 150:
            cube = lab == i
    grey = ((v > 60) & (v < 160) & (s < 45)).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones((5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grey)
    binm = np.zeros(grey.shape, bool)
    if n > 1:
        i = max(range(1, n), key=lambda k: st[k, 4])
        if st[i, 4] > 800:
            hull = np.zeros(grey.shape, np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero((lab == i).astype(np.uint8))), 1)
            binm = hull.astype(bool)
    return cube, binm


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def main() -> None:
    import compare_overhead as co

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("outputs/sim2real/20261126-211901"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--overhead", type=int, default=0)
    parser.add_argument("--wrist", type=int, default=1)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--out", type=Path, default=Path("outputs/compare_objects") / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--cube-shift-cm", type=float, nargs=2, default=(0.0, 0.0),
                        help="move the sim cube by (x, y) cm, e.g. to where the overhead camera locates the real one")
    parser.add_argument("--bin-shift-cm", type=float, nargs=2, default=(0.0, 0.0))
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    world = So101World()
    m, d = world.model, world.data
    seg = mujoco.Renderer(m, H, W)
    seg.enable_segmentation_rendering()
    col = mujoco.Renderer(m, H, W)
    arm = list(range(m.body("base").id, m.body("camera_mount").id + 1))
    CUBE, BOX = m.body("red_cube").id, m.body("box").id
    m.body_pos[BOX][:2] += np.asarray(args.bin_shift_cm) / 100
    cube_qadr = int(m.jnt_qposadr[m.body_jntadr[CUBE]])
    cube_xy = m.qpos0[cube_qadr:cube_qadr + 2] + np.asarray(args.cube_shift_cm) / 100
    K = np.array(json.loads((INTR / "overhead_intrinsics.json").read_text())["camera_matrix"])
    args.out.mkdir(parents=True, exist_ok=True)
    off = np.asarray(SIM_MINUS_REAL_DEG, float)

    frames = []  # (label, {camera: image}, sim joints)
    cap = json.loads((args.run / "capture.json").read_text())
    for r in cap["A"]:
        frames.append((r["tag"], {"overhead": cv2.imread(str(args.run / f"{r['tag']}_overhead.png")),
                                  "wrist_cam": cv2.imread(str(args.run / f"{r['tag']}_wrist.png"))},
                       np.asarray(r["measured_real"]) + off))  # current offsets, not the capture-time ones
    if args.live:
        from run_real import Camera, make_robot, read_deg
        cams = {"overhead": Camera(args.overhead, "overhead"), "wrist_cam": Camera(args.wrist, "wrist")}
        robot = make_robot(args.port, None)
        robot.bus.connect()  # read-only: torque untouched
        try:
            deg = read_deg(robot)
            time.sleep(0.3)
            imgs = {k: cv2.undistort(c.read()[0], c.K, c.D, None, c.K) for k, c in cams.items()}
        finally:
            robot.bus.disconnect(disable_torque=False)
            for c in cams.values():
                c.close()
        frames.append(("live", imgs, deg + off))

    def table_point(u, v, z):
        cid = m.camera("overhead").id
        o, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3)
        ray = R @ np.array([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -1.0])
        return o + (z - o[2]) / ray[2] * ray

    results = {"overhead": {"cube": [], "bin": []}, "wrist_cam": {"cube": [], "bin": []}}
    sheet = []
    for label, imgs, deg in frames:
        world.reset(world.deg_to_ctrl(deg))
        d.qpos[cube_qadr:cube_qadr + 2] = cube_xy
        mujoco.mj_forward(m, d)
        row = []
        for cam in ("overhead", "wrist_cam"):
            img = imgs[cam]
            seg.update_scene(d, camera=cam, scene_option=world._scene_option)
            s = seg.render()
            ok = (s[..., 0] >= 0) & (s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
            body = np.where(ok, m.geom_bodyid[np.clip(s[..., 0], 0, None)], -1)
            armmask = cv2.dilate(np.isin(body, arm).astype(np.uint8), np.ones((9, 9))).astype(bool)
            sc, sb = body == CUBE, body == BOX
            grow = lambda mk, px: cv2.dilate(mk.astype(np.uint8), np.ones((px, px))).astype(bool)  # noqa: E731
            rc, _ = real_masks(img, grow(sc, 161) if sc.any() else None)
            _, rb = real_masks(img, grow(sb, 241) if sb.any() else None)
            ov = img.copy()
            for name, real, sim in (("cube", rc, sc), ("bin", rb, sb)):
                real, sim = real & ~armmask, sim & ~armmask
                if real.sum() < 150 or sim.sum() < 150 or real.sum() < 0.4 * sim.sum():
                    continue  # occluded or out of view in one of the images
                iou = float((real & sim).sum() / (real | sim).sum())
                cr, cs = centroid(real), centroid(sim)
                rec = {"frame": label, "iou": iou, "offset_px": (cr - cs).tolist()}
                if cam == "overhead":
                    z = 0.03 if name == "cube" else 0.0
                    rec["offset_cm"] = ((table_point(*cr, z) - table_point(*cs, z))[:2] * 100).tolist()
                results[cam][name].append(rec)
                for mask, c in ((real, (255, 0, 255)), (sim, (0, 220, 0))):
                    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cnt, -1, c, 2)
            col.update_scene(d, camera=cam, scene_option=world._scene_option)
            blend = cv2.addWeighted(img, 0.5, cv2.cvtColor(col.render(), cv2.COLOR_RGB2BGR), 0.5, 0)
            cv2.putText(ov, f"{label} {cam}: magenta=real green=sim", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            row += [ov, blend]
        cv2.imwrite(str(args.out / f"{label}.jpg"), np.hstack(row))
        sheet.append(cv2.resize(np.hstack(row), None, fx=0.3, fy=0.3))
    cv2.imwrite(str(args.out / "summary.jpg"), np.vstack(sheet[:: max(1, len(sheet) // 6)][:6] + ([sheet[-1]] if args.live else [])))
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    for cam in results:
        for name, recs in results[cam].items():
            if not recs:
                print(f"{cam:9s} {name:4s}: not visible in any usable frame")
                continue
            iou = np.array([r["iou"] for r in recs])
            px = np.array([r["offset_px"] for r in recs])
            line = (f"{cam:9s} {name:4s}: {len(recs):2d} frames  IoU mean {iou.mean():.2f} (min {iou.min():.2f})  "
                    f"centroid offset real-sim px median ({np.median(px[:, 0]):+.0f}, {np.median(px[:, 1]):+.0f})")
            if cam == "overhead":
                cm = np.array([r["offset_cm"] for r in recs])
                line += f" = table ({np.median(cm[:, 0]):+.1f}, {np.median(cm[:, 1]):+.1f}) cm world x,y"
            live = [r for r in recs if r["frame"] == "live"]
            if live:
                line += f"   | live: IoU {live[0]['iou']:.2f}" + (
                    f", offset ({live[0]['offset_cm'][0]:+.1f}, {live[0]['offset_cm'][1]:+.1f}) cm" if "offset_cm" in live[0] else
                    f", offset ({live[0]['offset_px'][0]:+.0f}, {live[0]['offset_px'][1]:+.0f}) px")
            print(line)
    print("saved", args.out)


if __name__ == "__main__":
    main()
