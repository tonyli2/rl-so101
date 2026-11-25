"""Factor sweeps (proposal sec. 3, "Evaluation"; timeline Nov. 27).

    "We will vary background texture, lighting, external-camera pose, and object-finger friction
    one at a time, keeping other factors nominal. Tests will include nominal, moderate, and strong
    changes, held-out textures, and feasible values outside the training ranges. Exact settings
    will be fixed before evaluation. ... A small combined-shift test will examine interactions."

Every policy (nominal and randomized PPO seeds at one checkpoint step, plus the BC starting policy
as a reference) is evaluated on every setting below with the same 40 starts (matched episodes) and
deterministic actions. Settings are fixed values, not samples, so all policies see identical
conditions. "moderate" levels lie inside the calibrated training ranges (randomization_ranges.json:
texture 0.5, lighting 0.25, camera 0.75, friction 0.75 of the full ranges); "strong"/"extreme"
levels and the held-out textures lie outside them.

Results are appended to outputs/sweeps/<name>/results.jsonl one row per (policy, setting), so an
interrupted sweep resumes where it stopped; summary.md is rewritten after every setting.

    python factor_sweep.py                       # 500k checkpoints, all settings, all policies
    python factor_sweep.py --step 300000 --name step300k
    python factor_sweep.py --settings nominal lighting_dim_strong --policies main_nominal_s1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import randomization as R
import train_bc
import train_ppo
from envs import So101PickPlaceEnv, So101World, prepare_mounted

HERE = Path(__file__).parent
PPO_OUT = HERE / "outputs" / "ppo"
OUT = HERE / "outputs" / "sweeps"
BC = HERE / "outputs" / "bc" / "resnet18_h1_dart_rel_e20_seed0_last" / "bc_model.zip"
N_LIGHTS = 2  # the scene's directional lights (checked against the model at start-up)
SURFACES = ("table", "floor", "wall")


def texture(index: int, tint: float = 0.0) -> dict:
    return {s: {"tex": index, "tint": [tint, tint, tint]} for s in SURFACES}


def lighting(scale: float, ambient: float, tilt_deg: float) -> dict:
    """Brightness and ambient on all lights; direction change on the key light only.

    Tilting the directional overhead light (index 0) renders shadow-map aliasing (speckled stripes
    at ~5 deg, washed-out shadows at ~15 deg) - a rendering artifact, not a lighting change - so it
    stays fixed. (Training randomization tilted both lights; noted as a limitation.)
    """
    return {"tilt_deg": [[0.0, 0.0], [tilt_deg, tilt_deg]], "scale": scale, "tint": [0.0, 0.0, 0.0],
            "ambient": ambient}


def camera(shift_m: float, rot_deg: float, fovy_deg: float) -> dict:
    return {"pos_m": [shift_m, -shift_m, shift_m / 2], "rot_deg": [rot_deg, -rot_deg, rot_deg / 2],
            "fovy_deg": fovy_deg}


# Training ranges for reference (calibrated, at full curriculum strength): texture: each surface
# textured with p=0.5 from TRAIN_TEXTURES, tint +-0.075; light brightness x0.9-1.1, tilt +-6.25 deg,
# ambient 0.17-0.24; camera +-1.1 cm, +-2.25 deg, fovy +-2.25 deg; friction 0.55-1.15 (nominal 1.0,
# grasp slips below ~0.12-0.15).
SETTINGS: dict[str, tuple[str, str, dict]] = {
    # name: (factor, level, fixed_factors)
    "nominal": ("none", "nominal", {}),
    "texture_train": ("texture", "in training set", {"texture": texture(R.TRAIN_TEXTURES[0])}),
    "texture_heldout_red": ("texture", "held-out", {"texture": texture(R.HELDOUT_TEXTURES[0])}),
    "texture_heldout_purple": ("texture", "held-out", {"texture": texture(R.HELDOUT_TEXTURES[1])}),
    "texture_heldout_bright": ("texture", "held-out", {"texture": texture(R.HELDOUT_TEXTURES[2])}),
    "lighting_dim_moderate": ("lighting", "moderate", {"lighting": lighting(0.9, 0.17, 5.0)}),
    "lighting_bright_moderate": ("lighting", "moderate", {"lighting": lighting(1.1, 0.24, 5.0)}),
    "lighting_dim_strong": ("lighting", "strong", {"lighting": lighting(0.7, 0.12, 15.0)}),
    "lighting_bright_strong": ("lighting", "strong", {"lighting": lighting(1.3, 0.30, 15.0)}),
    "lighting_dim_extreme": ("lighting", "extreme", {"lighting": lighting(0.55, 0.08, 25.0)}),
    "camera_moderate": ("camera", "moderate", {"camera": camera(0.008, 1.5, 1.5)}),
    "camera_strong": ("camera", "strong", {"camera": camera(0.02, 4.0, 4.0)}),
    "camera_extreme": ("camera", "extreme", {"camera": camera(0.03, 6.0, -5.0)}),
    "friction_0.70": ("friction", "moderate", {"friction": 0.70}),
    "friction_0.45": ("friction", "strong", {"friction": 0.45}),
    "friction_0.30": ("friction", "strong", {"friction": 0.30}),
    "friction_0.20": ("friction", "extreme", {"friction": 0.20}),
    # Actuator dynamics (added 2026-11-23 at the user's request; evaluation only - no policy was
    # trained with it). Torque 0.7: the 7.4 V STS3215 gives ~65% of the 12 V torque the model uses.
    "actuator_torque_0.7": ("actuator", "moderate", {"actuator": {"torque_scale": 0.7}}),
    "actuator_torque_0.5": ("actuator", "strong", {"actuator": {"torque_scale": 0.5}}),
    "actuator_stiffness_0.5": ("actuator", "strong", {"actuator": {"kp_scale": 0.5}}),
    "actuator_stiffness_1.5": ("actuator", "strong", {"actuator": {"kp_scale": 1.5}}),
    "actuator_latency_1": ("actuator", "moderate", {"actuator": {"latency_steps": 1}}),
    "actuator_latency_2": ("actuator", "strong", {"actuator": {"latency_steps": 2}}),
    "actuator_latency_3": ("actuator", "measured", {"actuator": {"latency_steps": 3}}),  # real arm ~2.3-3.3
    "actuator_combined": ("actuator", "combined", {"actuator": {"torque_scale": 0.7, "kp_scale": 0.7,
                                                                "latency_steps": 1}}),
    "combined_moderate": ("combined", "moderate", {
        "texture": texture(R.TRAIN_TEXTURES[0]), "lighting": lighting(0.9, 0.17, 5.0),
        "camera": camera(0.008, 1.5, 1.5), "friction": 0.70}),
    "combined_strong": ("combined", "strong", {
        "texture": texture(R.HELDOUT_TEXTURES[0]), "lighting": lighting(0.7, 0.12, 15.0),
        "camera": camera(0.02, 4.0, 4.0), "friction": 0.45}),
}


def policies(step: int) -> dict[str, tuple[str, int | None, Path]]:
    """name -> (condition, seed, checkpoint). Runs without that checkpoint are skipped."""
    out: dict = {"bc": ("bc", None, BC)}
    for cond, prefix in (("nominal", "main_nominal_s"), ("randomized", "main_randomized_cal_s")):
        for seed in (1, 2, 3):
            path = PPO_OUT / f"{prefix}{seed}" / f"model_{step}.zip"
            if path.is_file():
                out[f"{prefix}{seed}"] = (cond, seed, path)
    return out


def run_episodes(venv, model: PPO, starts: list, n_envs: int, steps: int = 900) -> list[dict]:
    results = []
    for c in range(0, len(starts), n_envs):
        chunk = starts[c : c + n_envs]
        venv.set_options([{"joints": q} for q in chunk + [chunk[-1]] * (n_envs - len(chunk))])
        obs = venv.reset()
        done = np.zeros(n_envs, bool)
        lifted, in_box = np.zeros(n_envs, bool), np.zeros(n_envs, bool)
        outcome: list = [None] * n_envs
        for t in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            for i, info in enumerate(infos):
                if done[i]:
                    continue
                lifted[i] |= info["cube_lifted"]
                in_box[i] |= info["cube_in_box"]
                if dones[i]:
                    done[i] = True
                    outcome[i] = {"success": bool(info["is_success"]), "lifted": bool(lifted[i]),
                                  "in_box": bool(in_box[i]), "steps": t + 1}
            if done.all():
                break
        results.extend(outcome[: len(chunk)])
    return results


def summarize(rows: list[dict], path: Path) -> None:
    """Markdown table: success per setting, per condition (mean over seeds, seeds in brackets)."""
    by = {}
    for r in rows:
        by.setdefault(r["setting"], {}).setdefault(r["condition"], []).append(r)
    lines = ["| setting | factor | level | BC | nominal PPO (mean [seeds]) | randomized PPO (mean [seeds]) |",
             "|---|---|---|---|---|---|"]
    for name, (factor, level, _) in SETTINGS.items():
        if name not in by:
            continue
        cells = []
        for cond in ("bc", "nominal", "randomized"):
            rs = sorted(by[name].get(cond, []), key=lambda r: r["seed"] or 0)
            if not rs:
                cells.append("-")
            elif cond == "bc":
                cells.append(f"{rs[0]['success_rate']:.0%}")
            else:
                vals = [r["success_rate"] for r in rs]
                cells.append(f"{np.mean(vals):.0%} [{', '.join(f'{v:.0%}' for v in vals)}]")
        lines.append(f"| {name} | {factor} | {level} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=500_000, help="PPO checkpoint step (model_<step>.zip)")
    parser.add_argument("--name", help="output folder under outputs/sweeps (default step<N>)")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), help="subset of settings")
    parser.add_argument("--policies", nargs="+", help="subset of policies (bc, main_nominal_s1, ...)")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, help="use only the first N of the 40 starts (smoke tests)")
    parser.add_argument("--relative-base", choices=("measured", "command", "resync"), default="measured",
                        help="base of relative actions at test time: measured joints (as trained), the "
                             "previous command (robust to control latency), or the previous command with "
                             "blocked joints re-synced to their reading (no gripper wind-up)")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bc.load_config(BC)  # every PPO run shares the BC observation/action settings
    world = So101World()
    if world.model.nlight != N_LIGHTS or int(world.model.light_type[0]) != 1:  # 1 = directional
        sys.exit(f"sweep lighting assumes {N_LIGHTS} lights with the directional sun first; "
                 f"scene has {world.model.nlight}, types {world.model.light_type.tolist()}")
    groups = train_ppo.eval_starts(train_ppo.demo_starts(), world)  # the same 40 starts as training evals
    starts = [q for g in groups.values() for q in g][: args.episodes]

    pols = policies(args.step)
    if args.policies:
        pols = {k: v for k, v in pols.items() if k in args.policies}
    settings = {k: v for k, v in SETTINGS.items() if not args.settings or k in args.settings}
    out = OUT / (args.name or f"step{args.step}")
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.is_file() else []
    done = {(r["policy"], r["setting"]) for r in rows}
    cfg_path = out / "config.json"  # merged across runs into the same folder (e.g. added settings)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {"settings": {}, "policies": {}}
    if cfg_path.is_file() and cfg.get("relative_base", "measured") != args.relative_base:
        sys.exit(f"{out} holds results for relative_base={cfg.get('relative_base', 'measured')}; use --name")
    cfg.update(step=args.step, episodes_per_cell=len(starts), relative_base=args.relative_base)
    cfg["policies"].update({k: [c, s, str(p)] for k, (c, s, p) in pols.items()})
    cfg["settings"].update({k: {"factor": f, "level": lv, "fixed_factors": fx} for k, (f, lv, fx) in settings.items()})
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"{len(pols)} policies x {len(settings)} settings x {len(starts)} episodes -> {out}", flush=True)

    models = {name: PPO.load(path, device=device) for name, (_, _, path) in pols.items()}
    start_time = time.time()
    for name, (factor, level, fixed) in settings.items():
        todo = [p for p in pols if (p, name) not in done]
        if not todo:
            continue
        kw = train_bc.env_kwargs(fixed_factors=fixed) if fixed else train_bc.env_kwargs()
        venv = SubprocVecEnv([partial(So101PickPlaceEnv, **kw)] * args.n_envs)
        venv.set_attr("relative_base", args.relative_base)
        venv.seed(1234)
        try:
            for p in todo:
                cond, seed, _ = pols[p]
                eps = run_episodes(venv, models[p], starts, args.n_envs)
                row = {"policy": p, "condition": cond, "seed": seed, "step": args.step, "setting": name,
                       "factor": factor, "level": level, "episodes": len(eps),
                       "success_rate": float(np.mean([e["success"] for e in eps])),
                       "lifted_rate": float(np.mean([e["lifted"] for e in eps])),
                       "in_box_rate": float(np.mean([e["in_box"] for e in eps])),
                       "per_episode": eps}
                rows.append(row)
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{(time.time() - start_time) / 60:5.1f} min] {name:26s} {p:24s} success {row['success_rate']:5.1%}  "
                      f"lifted {row['lifted_rate']:5.1%}", flush=True)
        finally:
            venv.close()
        summarize(rows, out / "summary.md")
    summarize(rows, out / "summary.md")
    print(f"done: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
