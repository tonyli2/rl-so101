"""Calibrate the randomization ranges on the BC starting policy (proposal: "ranges established by an
early pilot").

Pilot finding (2026-11-23): randomized PPO at the full randomization.Ranges defaults collapses -
the BC policy cannot succeed there, so PPO gets no successes to learn from. This script tests the
BC policy with one factor randomized at a time, at several strengths (a fraction of the full
ranges), and picks for each factor the largest strength at which the policy still lifts the cube
at least ``--keep`` times as often as in the nominal scene. Those per-factor caps become the
training ranges (``randomization_ranges.json``, read by train_ppo.py --randomize). Evaluation
sweeps can still test stronger and held-out values.

    python calibrate_ranges.py outputs/bc/<run>/bc_model.zip            # report + suggestion
    python calibrate_ranges.py outputs/bc/<run>/bc_model.zip --write    # also write the ranges file
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_bc
import train_ppo
from envs import So101PickPlaceEnv, So101World, prepare_mounted

HERE = Path(__file__).parent
FACTORS = ("texture", "lighting", "camera", "friction")


def probe(model: PPO, starts: list, randomize: dict | None, strength: float, n_envs: int = 8) -> tuple[float, float]:
    """Success and lift rates of ``model`` (deterministic) from ``starts``, ``n_envs`` at a time."""
    kw = train_bc.env_kwargs() | ({"randomize": randomize} if randomize else {})
    venv = SubprocVecEnv([partial(So101PickPlaceEnv, **kw)] * n_envs)
    venv.set_attr("randomize_strength", strength)
    succ, lift = [], []
    try:
        for c in range(0, len(starts), n_envs):
            chunk = starts[c : c + n_envs]
            venv.set_options([{"joints": q} for q in chunk + [chunk[-1]] * (n_envs - len(chunk))])
            obs = venv.reset()
            done, s, lf = np.zeros(n_envs, bool), np.zeros(n_envs, bool), np.zeros(n_envs, bool)
            for _ in range(900):
                a, _ = model.predict(obs, deterministic=True)
                obs, _, d, infos = venv.step(a)
                for i, info in enumerate(infos):
                    if done[i]:
                        continue
                    lf[i] |= info["cube_lifted"]
                    if d[i]:
                        done[i], s[i] = True, info["is_success"]
                if done.all():
                    break
            succ.extend(s[: len(chunk)])
            lift.extend(lf[: len(chunk)])
    finally:
        venv.close()
    return float(np.mean(succ)), float(np.mean(lift))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="the BC checkpoint PPO starts from")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--keep", type=float, default=2 / 3, help="required fraction of the nominal lift rate")
    parser.add_argument("--write", action="store_true", help=f"write {HERE / 'randomization_ranges.json'}")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    train_bc.load_config(args.model)
    model = PPO.load(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    starts = list(train_ppo.start_joints(train_ppo.demo_starts(), So101World(), set(range(45)))[::2][: args.episodes])

    nominal = probe(model, starts, None, 0.0)
    print(f"nominal: success {nominal[0]:.0%}  lifted {nominal[1]:.0%}  ({len(starts)} episodes)", flush=True)
    table: dict = {"nominal": {"success": nominal[0], "lifted": nominal[1]}}
    caps = {}
    for f in FACTORS:
        table[f] = {}
        for s in args.strengths:
            sr, lr = probe(model, starts, {"factors": [f]}, s)
            table[f][str(s)] = {"success": sr, "lifted": lr}
            print(f"{f:9s} strength {s:.2f}: success {sr:.0%}  lifted {lr:.0%}", flush=True)
        ok = [s for s in args.strengths if table[f][str(s)]["lifted"] >= args.keep * nominal[1]]
        caps[f] = max(ok) if ok else min(args.strengths) / 2  # nothing held up: half the mildest tested
    print(f"\nsuggested per-factor strength caps (lift >= {args.keep:.0%} of nominal): {caps}")
    out = {"model": str(args.model), "episodes": len(starts), "keep_fraction_of_nominal_lift": args.keep,
           "table": table, "suggested_caps": caps}
    (train_ppo.OUT / "calibration.json").write_text(json.dumps(out, indent=2))
    if args.write:
        (HERE / "randomization_ranges.json").write_text(json.dumps(
            {"source": "calibrate_ranges.py", "calibration": out, "ranges": {"factor_strength": caps}}, indent=2))
        print(f"wrote {HERE / 'randomization_ranges.json'}")


if __name__ == "__main__":
    sys.exit(main())
