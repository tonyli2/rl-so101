"""Which camera does the BC policy rely on in each phase of the task?

Hypothesis: the overhead camera guides reaching and carrying (global view); the wrist camera
matters when grasping and releasing (local view; the arm blocks the overhead view there).

Each validation frame is labelled with a task phase from its archived MuJoCo state (gripper-cube
distance, cube height, cube in box: information the policy never sees). Within each phase, one input
is replaced by the same input from random other validation frames ("scrambled"), which removes its
information while keeping realistic values. Reported per phase:

* change: RMS change of the predicted action vs the intact input -> how much the policy relies on it
* error:  RMS error vs the demonstrated action, intact and scrambled -> how much the input helps

Arm joints in degrees (mean over the five arm joints); gripper in percent of its range.

    python camera_ablation.py outputs/bc/resnet18_e80_seed0/bc_model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import make_bc_dataset as mb
import train_bc
from envs import CUBE_HALF, So101PickPlaceEnv, prepare_mounted

PHASES = ("reach", "grasp", "carry", "release")
NEAR_CUBE = 0.06  # m, gripper site to cube centre
LIFTED = CUBE_HALF + 0.01


def phase_labels(episodes: set[int]) -> dict[int, np.ndarray]:
    """Per-frame phase index (into PHASES) for the given episodes, from the archived states."""
    env = So101PickPlaceEnv(action_mode="absolute", image=False, include_targets=False)
    replayer = mb.ArchiveReplayer(env)
    labels = {}
    for item in mb.episodes():
        if item["index"] not in episodes:
            continue
        states = np.load(item["npz"])["states"][:-1]  # frames that have an action
        phase = np.empty(len(states), np.int64)
        for t, state in enumerate(states):
            replayer.load(item["scene"], state)
            w = env.world
            cube = w.cube_pos()
            if w.cube_in_box():
                phase[t] = 3
            elif cube[2] > LIFTED:
                phase[t] = 2
            elif np.linalg.norm(w.ee_pos() - cube) < NEAR_CUBE:
                phase[t] = 1
            else:
                phase[t] = 0
        labels[item["index"]] = phase
    env.close()
    return labels


UNITS: dict = {}  # half_range (radians per action unit) and span, set from the loaded Demos


def to_units(delta_norm: torch.Tensor, span: np.ndarray) -> tuple[float, float]:
    """RMS over frames of normalized action differences -> (mean arm-joint degrees, gripper %).

    Works for absolute and relative actions: one action unit is ``half_range`` radians.
    """
    rms = delta_norm.pow(2).mean(0).sqrt().cpu().numpy() * UNITS["half_range"]
    arm_deg = np.rad2deg(rms[:5]).mean()
    return float(arm_deg), float(rms[5] / UNITS["span"][5] * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeats", type=int, default=5, help="random scrambles per condition")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bc.load_config(args.model)  # the model's observation settings (history, velocities)
    demos = train_bc.Demos(train_bc.data_path(), device)
    model = PPO.load(args.model, device=device)
    policy = model.policy
    policy.set_training_mode(False)
    world = So101PickPlaceEnv(**train_bc.env_kwargs()).world
    span = world.hi - world.lo
    UNITS.update(half_range=demos.half_range, span=demos.span)
    world.close()

    labels = phase_labels(train_bc.val_episodes())
    val_idx = demos.val_idx
    phase = np.empty(len(val_idx), np.int64)
    for ep, ep_labels in labels.items():
        rows = np.flatnonzero(demos.episode[val_idx] == ep)  # contiguous, ascending frame order
        phase[rows] = ep_labels
    rng = np.random.default_rng(args.seed)

    def predict(obs: dict) -> torch.Tensor:
        with torch.no_grad():
            return policy.get_distribution(obs).distribution.mean.clamp(-1, 1)

    conditions = {"overhead": ["front"], "wrist": ["wrist"], "both cameras": ["front", "wrist"], "joint state": ["state"]}
    results = {}
    for p, name in enumerate(PHASES):
        idx = val_idx[phase == p]
        if len(idx) == 0:
            continue
        obs, act = demos.batch(idx)
        intact = predict(obs)
        row = {"frames": int(len(idx)), "error_intact": to_units(intact - act, span)}
        for cond, keys in conditions.items():
            changes, errors = [], []
            for _ in range(args.repeats):
                donor, _ = demos.batch(rng.choice(val_idx, size=len(idx)))  # random validation frames
                scrambled = {k: (donor[k] if k in keys else v) for k, v in obs.items()}
                pred = predict(scrambled)
                changes.append(to_units(pred - intact, span))
                errors.append(to_units(pred - act, span))
            row[cond] = {"change": tuple(np.mean(changes, 0)), "error": tuple(np.mean(errors, 0))}
        results[name] = row

    print(f"\nModel: {args.model}   validation frames per phase: "
          + ", ".join(f"{k} {v['frames']}" for k, v in results.items()))
    print("\nHow much the predicted action CHANGES when an input is scrambled (arm deg | gripper %):")
    print(f"{'phase':8s}" + "".join(f"{c:>20s}" for c in conditions))
    for name, row in results.items():
        print(f"{name:8s}" + "".join(f"{row[c]['change'][0]:11.2f} | {row[c]['change'][1]:5.1f}" for c in conditions))
    print("\nAction ERROR vs the demonstration (arm deg | gripper %):")
    print(f"{'phase':8s}{'intact':>20s}" + "".join(f"{c:>20s}" for c in conditions))
    for name, row in results.items():
        print(f"{name:8s}{row['error_intact'][0]:11.2f} | {row['error_intact'][1]:5.1f}"
              + "".join(f"{row[c]['error'][0]:11.2f} | {row[c]['error'][1]:5.1f}" for c in conditions))

    out = args.model.parent / "camera_ablation.json"
    out.write_text(json.dumps({"model": str(args.model), "near_cube_m": NEAR_CUBE, "lifted_m": LIFTED,
                               "repeats": args.repeats, "phases": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    sys.exit(main())
