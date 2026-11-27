"""Build a fresh held-out rollout suite in the format of Tony's evaluation bundles (eval_suite.py runs it).

Tony's reserved final test (dataset-so101-physical-v1-200ep evaluation/final-test) was used once
for the v1 runs, and its results shaped the v2 changes, so it no longer gives an unbiased number
for v2. This makes a new suite the same way, with fresh seeds and several layouts:

* layouts: cube +-4 cm / bin rules as in Tony's recording/layout.py (our env's sampler), seeded;
  each cube at least --min-dist from every recorded demo cube start and from the validation and
  final-test cube positions;
* start: the final test's reference arm pose (Tony's bootstrap pose), the case's cube and bin;
* appearance: one modest/stress pair per layout, Tony's modest profile and five stress factors
  (learning/visual_test_suite.py), with a fresh profile seed;
* preflight: no prohibited contact (RecordingCollisionChecker) at the start or after one held step.

    python make_heldout_suite.py --out outputs/suite_eval/heldout_v2 --layouts 10 --seed 20261126
Built once; never regenerate or retune it in response to results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE = Path.home() / "datasets" / "so101-physical-v1-200ep" / "datasets" / "sim-varying-1-visual-v1"
OVERLAY = Path.home() / "datasets" / "so101-physical-v1-200ep" / "training-tools-overlay"
FACTORS = ("dim_light", "bright_light", "side_light", "blur", "warm_light")


def load_overlay_vr():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "visual_randomization", OVERLAY / "lerobot-tabletop" / "recording" / "visual_randomization.py")
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    return vr


def modest_profile(vr, pair: int, seed: int) -> dict:
    """Tony's learning/visual_test_suite.modest_profile with another seed."""
    p = vr.sample_profile(pair, 1, seed=seed)
    p["light_intensity"]["scale"] = 1 + 0.5 * (p["light_intensity"]["scale"] - 1)
    p["light_direction"]["tilt_x"] *= 0.6
    p["light_direction"]["tilt_y"] *= 0.6
    p["light_color"]["rgb_gain"] = [1 + 0.5 * (v - 1) for v in p["light_color"]["rgb_gain"]]
    p["camera_effects"]["exposure"] = 1 + 0.5 * (p["camera_effects"]["exposure"] - 1)
    p["camera_effects"]["noise_sigma_255"] = 1.5
    p["camera_effects"]["blur_radius_px"] = 0.2
    return p


def stress_case(case: dict, factor: str, cube: np.ndarray) -> dict:
    """Tony's stress variants (learning/visual_test_suite.prepare)."""
    profile = case["appearance_profile"]
    if factor == "dim_light":
        profile["light_intensity"] = {"scale": 0.35, "ambient_scale": 0.55}
    elif factor == "bright_light":
        profile["light_intensity"] = {"scale": 1.9, "ambient_scale": 1.2}
    elif factor == "side_light":
        pos = cube + np.array([0.1, 0.15, 0.35])
        d = cube - pos
        case["secondary_light"] = {"position": pos.tolist(), "direction": (d / np.linalg.norm(d)).tolist(),
                                   "diffuse": [1.15, 1.0, 0.85], "specular": [0.5, 0.5, 0.5]}
    elif factor == "blur":
        profile["camera_effects"]["blur_radius_px"] = 1.8
    elif factor == "warm_light":
        profile["light_color"]["rgb_gain"] = [1.2, 0.88, 0.65]
    return case


def recorded_cubes() -> np.ndarray:
    """Cube start positions (world x, y) of every original demonstration in the release."""
    import make_bc_dataset as M
    from envs import So101PickPlaceEnv

    M.configure(RELEASE, "heldout-probe")
    env = So101PickPlaceEnv(action_mode="absolute", image=False)
    rep = M.ArchiveReplayer(env)
    out = []
    for item in M.episodes():
        model, data = rep.archived(item["scene"])
        with np.load(item["npz"]) as z:
            mujoco.mj_setState(model, data, z["states"][0], M.STATE_MASK)
        adr = model.jnt_qposadr[model.body_jntadr[model.body("red_cube").id]]
        out.append(data.qpos[adr: adr + 2].copy())
    env.close()
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "outputs" / "suite_eval" / "heldout_v2")
    parser.add_argument("--layouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261126, help="layout sampling seed")
    parser.add_argument("--profile-seed", type=int, default=202611262, help="appearance seed (Tony used 202611251)")
    parser.add_argument("--min-dist", type=float, default=0.02, help="metres from any recorded/eval cube")
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists: the suite is built once and frozen")
    sys.path.insert(0, str(HERE.parent / "lerobot-tabletop"))
    from recording.self_collision import RecordingCollisionChecker

    from envs import So101PickPlaceEnv

    vr = load_overlay_vr()
    final = RELEASE / "evaluation" / "final-test"
    val = RELEASE / "evaluation" / "validation"
    ref_manifest = json.loads((final / "manifest.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(final / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(final / "states.npz") as z:
        ref_state = z["states"][0].copy()
    cube_adr = model.jnt_qposadr[model.body_jntadr[model.body("red_cube").id]]
    box = model.body("box").id

    # Positions to stay away from: recorded demo starts, validation and final-test cubes.
    avoid = [recorded_cubes()]
    for suite in (val, final):
        m = mujoco.MjModel.from_binary_path(str(suite / "model.mjb"))
        d = mujoco.MjData(m)
        with np.load(suite / "states.npz") as z:
            for st in z["states"]:
                mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
                avoid.append(d.qpos[cube_adr: cube_adr + 2][None].copy())
    avoid = np.concatenate(avoid)

    env = So101PickPlaceEnv(action_mode="absolute", image=False, cube_xy_range=(0.04, 0.04),
                            bin_away_range=(0.0, 0.04), bin_side_range=0.04)
    env.reset(seed=args.seed)
    checker = RecordingCollisionChecker(model)
    cases, states, bins, layouts = [], [], [], []
    tries = 0
    while len(layouts) < args.layouts:
        tries += 1
        if tries > 5000:
            sys.exit("could not place enough layouts")
        xy, _, bin_pos = env._sample_layout()
        if np.min(np.linalg.norm(avoid - xy, axis=1)) < args.min_dist:
            continue
        if layouts and np.min(np.linalg.norm(np.array([l[0] for l in layouts]) - xy, axis=1)) < args.min_dist:
            continue
        mujoco.mj_setState(model, data, ref_state, mujoco.mjtState.mjSTATE_INTEGRATION)
        model.body_pos[box] = bin_pos
        data.qpos[cube_adr: cube_adr + 2] = xy
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if checker.check(data) is not None:
            continue
        state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        probe = mujoco.MjData(model)  # one held control step: still clear, cube resting
        mujoco.mj_setState(model, probe, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_step(model, probe, nstep=int(ref_manifest["substeps"]))
        if checker.check(probe) is not None or np.linalg.norm(probe.qpos[cube_adr: cube_adr + 2] - xy) > 0.002:
            continue
        layouts.append((xy.copy(), bin_pos.copy(), state, float(data.qpos[cube_adr + 2])))
    env.close()

    for k, (xy, bin_pos, state, cube_z) in enumerate(layouts):
        factor = FACTORS[k % len(FACTORS)]
        mild = modest_profile(vr, k, args.profile_seed)
        cube3 = np.array([xy[0], xy[1], cube_z])
        for severity in ("modest", "stress"):
            case = {"id": f"layout{k + 1:02d}_{severity}_{factor}", "family": severity, "index": len(cases),
                    "pair": k + 1, "layout": k + 1, "stress_factor": factor, "primary": True,
                    "appearance_profile": copy.deepcopy(mild),
                    "cube_position": cube3.tolist(), "bin_position": bin_pos.tolist()}
            if severity == "stress":
                case = stress_case(case, factor, cube3)
            cases.append(case)
            states.append(state)
            bins.append(bin_pos)

    args.out.mkdir(parents=True)
    shutil.copy(final / "model.mjb", args.out / "model.mjb")
    np.savez_compressed(args.out / "states.npz", states=np.array(states), bin_positions=np.array(bins))
    d_min = [float(np.min(np.linalg.norm(avoid - l[0], axis=1))) for l in layouts]
    manifest = {
        "schema_version": 1, "name": "heldout-v2-visual-layouts", "role": "final_test",
        "appearance_protocol": "visual-v1", "fps": ref_manifest["fps"], "deadline": ref_manifest["deadline"],
        "height": ref_manifest["height"], "width": ref_manifest["width"], "substeps": ref_manifest["substeps"],
        "instruction": ref_manifest["instruction"], "bin_reachability_checked": True,
        "generator": "so101_rl/make_heldout_suite.py", "layout_seed": args.seed, "profile_seed": args.profile_seed,
        "layouts": args.layouts, "min_distance_from_recorded_or_eval_cube_m": min(d_min),
        "reference_arm_state": "final-test states[0] (Tony's bootstrap reference pose)",
        "scope": "10 layouts x modest/stress appearance at one arm pose. Built 2026-11-26 before any v2 test "
                 "result; run once on the v2 round's selected checkpoints.",
        "cases": cases,
        "model_sha256": hashlib.sha256((args.out / "model.mjb").read_bytes()).hexdigest(),
        "states_sha256": hashlib.sha256((args.out / "states.npz").read_bytes()).hexdigest(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for (xy, bin_pos, _, _), d in zip(layouts, d_min):
        print(f"cube {np.round(xy, 3).tolist()} bin {np.round(bin_pos[:2], 3).tolist()}  nearest known cube {100 * d:.1f} cm")
    print(f"{len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
