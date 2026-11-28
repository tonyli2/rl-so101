"""Success map over cube positions: where in the +-range does a policy fail?

Runs each policy on a G x G grid of cube offsets (bin at its measured spot unless --bin-random),
N episodes per cell from jittered start poses, deterministic actions, optional delay.

    python grid_eval.py --policies outputs/ppo/physA_nominal_s1/model_900000.zip ... --delay 2:0:1
Writes outputs/grid_eval/<name>.json and prints one success map per policy (rows: forward
offset +x .. -x, columns: lateral offset +y .. -y, metres in the world frame).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "cgl" if sys.platform == "darwin" else "wgl")

import numpy as np

HERE = Path(__file__).resolve().parent


def work(job: dict) -> dict:
    import torch

    import deploy
    from envs import So101PickPlaceEnv

    torch.set_num_threads(1)
    policy = deploy.load_policy(job["policy"], device="cpu")
    a, j, i = (int(v) for v in job["delay"].split(":"))
    extra = {"fixed_factors": {"actuator": {"latency_steps": a, "obs_latency_steps": j,
                                            "image_latency_steps": i}}} if (a or j or i) else {}
    if job.get("randomize"):  # training-style randomization (its own delays) instead of a fixed setting
        extra = {"randomize": job["randomize"]}
    env = So101PickPlaceEnv(**policy.env_kwargs(**extra))
    w = env.world
    env.reset(seed=0)
    cube0 = w.cube_start[:2].copy()
    box0 = env._box0[:2].copy()
    out = []
    for k in range(job["episodes"]):
        xy = cube0 + np.asarray(job["offset"])
        obs, _ = env.reset(seed=job["seed"] + k, options={"layout": {"cube_xy": xy, "bin_xy": box0}})
        lifted = success = False
        for _ in range(env.episode_steps):
            action, _ = policy.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            lifted |= bool(info["cube_lifted"])
            if term or trunc:
                success = bool(info["is_success"])
                break
        out.append({"success": success, "lifted": lifted})
    env.close()
    return {**{k: job[k] for k in ("policy", "cell", "offset")},
            "success": sum(r["success"] for r in out), "lifted": sum(r["lifted"] for r in out),
            "episodes": len(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", nargs="+", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--range", type=float, default=0.04, help="+- metres of cube offset covered")
    parser.add_argument("--episodes", type=int, default=4, help="per cell")
    parser.add_argument("--delay", default="2:0:1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--name", default="grid")
    parser.add_argument("--randomize", nargs="*", metavar="FACTOR",
                        help="resample these randomization factors every episode (e.g. appearance camera friction "
                             "actuator calibration) with --randomize-ranges; replaces --delay")
    parser.add_argument("--randomize-ranges", type=Path, default=HERE / "randomization_ranges_v3_real.json")
    args = parser.parse_args()
    randomize = None
    if args.randomize:
        randomize = {"factors": args.randomize, "ranges": json.loads(args.randomize_ranges.read_text())["ranges"]}
    offs = np.linspace(-args.range, args.range, args.grid)
    jobs = [{"policy": str(p), "cell": [r, c], "offset": [float(offs[::-1][r]), float(offs[::-1][c])],
             "episodes": args.episodes, "delay": args.delay, "seed": 5000 + 100 * (r * args.grid + c),
             "randomize": randomize}
            for p in args.policies for r in range(args.grid) for c in range(args.grid)]
    start = time.time()
    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(work, jobs))
    out = HERE / "outputs" / "grid_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"args": vars(args) | {"policies": [str(p) for p in args.policies]},
                                                      "offsets_m": offs.tolist(), "cells": rows}, indent=2, default=str))
    for p in args.policies:
        grid = np.zeros((args.grid, args.grid), int)
        for r in rows:
            if r["policy"] == str(p):
                grid[r["cell"][0], r["cell"][1]] = r["success"]
        cond = f"randomized {args.randomize} from {args.randomize_ranges.name}" if args.randomize else f"delay {args.delay}"
        print(f"\n{p} (successes of {args.episodes} per cell, {cond}; rows +x..-x, cols +y..-y):")
        for row in grid:
            print("  " + " ".join(f"{v:2d}" for v in row))
    print(f"\n{len(jobs)} cells in {time.time() - start:.0f} s -> {out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
