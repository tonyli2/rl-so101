"""Fingertip height check: does the sim agree that the real fingertips are on the table?

Torque stays OFF. Move the arm by hand until the gripper fingertips just touch the table, hold it,
and press Enter. For each pose the script prints the sim's height of the lowest gripper point
(fixed + moving jaw meshes) at the measured joints, using run_real.SIM_MINUS_REAL_DEG; a correct
calibration gives about 0 cm. Vary the poses (near / far from the base, wrist straight down /
tilted) so lift, elbow and wrist flex can be separated. At the end it fits the extra lift /
elbow / flex offsets that bring every pose to 0 and saves everything.

    python height_check.py            # Enter = record a pose, q + Enter = finish
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

if os.uname().sysname == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco
import numpy as np

from envs import So101World, prepare_mounted
from run_real import SIM_MINUS_REAL_DEG, make_robot, read_deg


class Fingertips:
    def __init__(self, world: So101World):
        self.w, m = world, world.model
        bodies = {m.body("gripper").id, m.body("moving_jaw_so101_v1").id}
        self.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.verts = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:
                                     m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(np.float64) for g in self.geoms}

    def lowest(self, deg_sim) -> float:
        """Height (m) of the lowest jaw-mesh vertex above the table at sim-convention joints."""
        self.w.reset(self.w.deg_to_ctrl(np.asarray(deg_sim, float)))
        d = self.w.data
        # only the z row is needed: z = pos_z + verts . R[2, :]
        return float(min(d.geom_xpos[g][2] + (self.verts[g] * d.geom_xmat[g][6:9]).sum(1).min() for g in self.geoms))


def fit(tips: Fingertips, poses_sim: np.ndarray, reg: float = 1e-8):
    """Extra (lift, elbow, flex) offsets in deg minimizing sum(height^2) + reg * |offset|^2 (Gauss-Newton)."""
    x = np.zeros(3)
    idx = [1, 2, 3]

    def res(x):
        out = []
        for p in poses_sim:
            q = p.copy()
            q[idx] += x
            out.append(tips.lowest(q))
        return np.r_[np.array(out), np.sqrt(reg) * x]

    for _ in range(15):
        r = res(x)
        J = np.stack([(res(x + e * 0.2) - r) / 0.2 for e in np.eye(3)], axis=1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + np.clip(step, -5, 5)
        if np.abs(step).max() < 0.05:
            break
    return x, res(x)[: len(poses_sim)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/tty.usbmodem5B790810461")
    parser.add_argument("--out", type=Path, default=Path("outputs/height_check") / f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    parser.add_argument("--analyze", type=Path, help="re-fit a saved file instead of recording")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    tips = Fingertips(So101World())
    off = np.asarray(SIM_MINUS_REAL_DEG, float)

    if args.analyze:
        recs = json.loads(args.analyze.read_text())["poses"]
    else:
        robot = make_robot(args.port, None)
        robot.bus.connect()  # read-only: torque is never enabled
        recs = []
        print("Torque is OFF. Put the fingertips on the table, hold still, press Enter. 'q' + Enter to finish.")
        try:
            while True:
                if input(f"pose {len(recs) + 1}> ").strip().lower() == "q":
                    break
                deg = np.mean([read_deg(robot) for _ in range(10)], axis=0)
                z = tips.lowest(deg + off)
                recs.append({"real_deg": deg.tolist()})
                print(f"   joints (real) {np.round(deg, 1)}  ->  sim fingertip height {z * 100:+.1f} cm")
        finally:
            robot.bus.disconnect(disable_torque=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)

    poses = np.array([np.asarray(r["real_deg"]) + off for r in recs])
    if not len(poses):
        return
    z0 = np.array([tips.lowest(p) for p in poses])
    print(f"\n{len(poses)} poses with the current offsets {tuple(off)}: sim fingertip heights (cm) {np.round(z0 * 100, 1)}; "
          f"mean {z0.mean() * 100:+.1f}, rms {np.sqrt((z0 ** 2).mean()) * 100:.1f}")
    result = {"sim_minus_real_deg": off.tolist(), "poses": recs, "heights_cm": (z0 * 100).tolist()}
    if len(poses) >= 3:
        x, z1 = fit(tips, poses)
        print(f"best extra offsets (lift, elbow, flex) {np.round(x, 1)} deg -> heights (cm) {np.round(z1 * 100, 1)}, "
              f"rms {np.sqrt((z1 ** 2).mean()) * 100:.1f}")
        result.update({"fit_extra_lift_elbow_flex_deg": x.tolist(), "fit_heights_cm": (z1 * 100).tolist()})
    if not args.analyze:
        args.out.write_text(json.dumps(result, indent=1))
        print("saved", args.out)


if __name__ == "__main__":
    main()
