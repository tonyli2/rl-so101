"""Stage 1 (proposal sec. 3, "Staged objectives"): PPO reaching on the SO-101.

The gripper site must end each episode within ``--tol`` of a sampled reachable target. Success rate
is N_success / N_total over ``--eval-episodes`` deterministic episodes; stage 1 passes at >= 80%.

    python train_reach.py                          # mounted model, 1M steps, then evaluate
    python train_reach.py --steps 300000 --seed 1
    python train_reach.py --variant legacy         # the original Menagerie arm at the origin
    python train_reach.py --eval outputs/reach/mounted/seed0/model.zip
    python train_reach.py --view outputs/reach/mounted/seed0/model.zip
    tensorboard --logdir outputs/reach
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs import VARIANTS, So101ReachEnv, prepare_mounted

OUT = Path(__file__).parent / "outputs" / "reach"  # runs go in OUT / <variant> / seed<N>
PASS_RATE = 0.80


def evaluate(model: PPO, episodes: int, seed: int, tol: float, variant: str) -> dict:
    env = So101ReachEnv(success_tol=tol, variant=variant)
    successes, finals = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(info["is_success"])
        finals.append(info["distance"])
    env.close()
    finals = np.array(finals)
    return {
        "variant": variant,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "tolerance_m": tol,
        "final_distance_mean_m": float(finals.mean()),
        "final_distance_median_m": float(np.median(finals)),
        "final_distance_p90_m": float(np.percentile(finals, 90)),
    }


def view(model: PPO, tol: float, variant: str) -> None:
    """Run the policy in real time in the interactive viewer; a new target every episode."""
    import mujoco.viewer

    env = So101ReachEnv(success_tol=tol, variant=variant)
    obs, _ = env.reset(seed=0)
    with mujoco.viewer.launch_passive(env.world.model, env.world.data) as viewer:
        episode = 0
        while viewer.is_running():
            tick = time.monotonic()
            action, _ = model.predict(obs, deterministic=True)
            with viewer.lock():
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"episode {episode}: final distance {info['distance'] * 100:.1f} cm "
                          f"{'SUCCESS' if info['is_success'] else 'miss'}", flush=True)
                    episode += 1
                    obs, _ = env.reset(seed=episode)
            viewer.sync()
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    env.close()


class SuccessLogger(BaseCallback):
    """Log the training success rate (episodes ending within tolerance) to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.buffer: list[bool] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.buffer.append(bool(info.get("is_success", False)))
        if len(self.buffer) >= 100:
            self.logger.record("rollout/success_rate", float(np.mean(self.buffer)))
            self.buffer.clear()
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--tol", type=float, default=0.02, help="success distance (m)")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval", type=Path, help="evaluate a saved model instead of training")
    parser.add_argument("--view", type=Path, help="watch a saved model in the MuJoCo viewer")
    parser.add_argument("--variant", choices=VARIANTS, default="mounted",
                        help="arm/scene model: mounted (current stock-mount assembly) or legacy")
    args = parser.parse_args()

    if args.variant == "mounted":
        prepare_mounted(rebuild=True)  # once here; worker processes reuse the files
    if args.view:
        view(PPO.load(args.view, device="cpu"), args.tol, args.variant)
        return
    if args.eval:
        model = PPO.load(args.eval, device="cpu")
        result = evaluate(model, args.eval_episodes, seed=10_000, tol=args.tol, variant=args.variant)
        print(json.dumps(result, indent=2))
        return

    run = OUT / args.variant / f"seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(
        So101ReachEnv, n_envs=args.n_envs, seed=args.seed,
        env_kwargs={"success_tol": args.tol, "variant": args.variant}, vec_env_cls=SubprocVecEnv,
    )
    model = PPO(
        "MlpPolicy", venv, seed=args.seed, device="cpu", verbose=1,
        n_steps=512, batch_size=512, n_epochs=10, learning_rate=3e-4,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -0.5},
        tensorboard_log=str(OUT / args.variant),
    )
    start = time.time()
    model.learn(args.steps, callback=SuccessLogger(), tb_log_name=f"seed{args.seed}")
    train_s = time.time() - start
    model.save(run / "model.zip")
    venv.close()

    result = evaluate(model, args.eval_episodes, seed=10_000, tol=args.tol, variant=args.variant)
    result.update({
        "seed": args.seed, "timesteps": args.steps, "train_seconds": round(train_s, 1),
        "stage1_pass": result["success_rate"] >= PASS_RATE,
    })
    (run / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"STAGE 1 {'PASS' if result['stage1_pass'] else 'FAIL'} "
          f"({result['success_rate']:.0%} >= {PASS_RATE:.0%} needed)")


if __name__ == "__main__":
    main()
