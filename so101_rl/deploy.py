"""Load a policy bundle for deployment, with its action mapping taken from the bundle itself.

A bundle is a folder with ``model.zip`` (SB3 PPO) and ``bc_config.json``. The config's ``obs``
block fixes the observation and how actions become joint targets, including
``relative_base`` (``"resync"`` for deployment, see action_mapping.py), so whoever loads the
bundle gets the mapping it was validated with instead of a default.

    from deploy import load_policy
    policy = load_policy("outputs/deploy/ppo_randomized_s1_500k")
    env = So101PickPlaceEnv(**policy.env_kwargs())            # sim: mapping built in
    mapper = policy.mapper(world.lo, world.hi)                 # hardware: same mapping object

Sim check of a bundle (the sweep's 3-step latency setting by default):

    python deploy.py outputs/deploy/ppo_randomized_s1_500k --latency 3 --episodes 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from action_mapping import ActionMapper

DEPLOY_BASE = "resync"


@dataclass
class Policy:
    path: Path
    obs: dict
    model: object = None

    def env_kwargs(self, **extra) -> dict:
        import train_bc

        train_bc.configure(**self.obs)
        return train_bc.env_kwargs(**extra)

    def mapper(self, lo, hi) -> ActionMapper:
        return ActionMapper.from_config(self.obs, lo, hi)


def load_policy(bundle: str | Path, device: str = "auto", allow_other_base: bool = False) -> Policy:
    bundle = Path(bundle)
    model_path = bundle / "model.zip" if bundle.is_dir() else bundle
    cfg_path = model_path.parent / "bc_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} missing: the bundle's config defines the observation and "
                                "action mapping; refusing to guess")
    obs = json.loads(cfg_path.read_text())["obs"]
    base = obs.get("relative_base", "measured")
    if obs.get("action") == "relative" and base != DEPLOY_BASE and not allow_other_base:
        raise ValueError(f"{cfg_path} sets relative_base={base!r}; deployment needs {DEPLOY_BASE!r} "
                         "(the measured base fails at the real arm's ~3-step latency). Pass "
                         "allow_other_base=True for experiments.")
    from stable_baselines3 import PPO

    print(f"policy {model_path}: action {obs.get('action')}, relative_base {base}, "
          f"resync {obs.get('resync') or 'defaults'}", flush=True)
    return Policy(model_path, obs, PPO.load(model_path, device=device))


def main() -> None:
    import numpy as np
    import torch

    import train_bc
    import train_ppo
    from envs import So101PickPlaceEnv, So101World, prepare_mounted

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--latency", type=int, default=3, help="control latency in steps (real arm ~2.3-3.3)")
    parser.add_argument("--episodes", type=int, default=8, help="validation starts to run (max 20)")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    policy = load_policy(args.bundle)
    env = So101PickPlaceEnv(**policy.env_kwargs(
        episode_steps=900, fixed_factors={"actuator": {"latency_steps": args.latency}}))
    print(f"env action base: {env.relative_base}", flush=True)
    train_bc.configure(**policy.obs)
    g = train_ppo.eval_starts(train_ppo.demo_starts(), So101World())
    starts = (g["val_exact"] + g["val_jitter3deg"])[: args.episodes]
    wins = []
    for q in starts:
        obs, _ = env.reset(options={"joints": q})
        for _ in range(900):
            action, _ = policy.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        wins.append(bool(info["is_success"]))
    print(f"latency {args.latency}: success {np.mean(wins):.0%} ({sum(wins)}/{len(wins)}); "
          f"gripper re-syncs per episode {env.mapper.resync_count[5]} (last episode)")


if __name__ == "__main__":
    main()
