"""PPO fine-tuning of a BC policy (proposal sec. 3, "Training"; timeline Nov. 18).

The BC checkpoint (train_bc.py) is already a Stable-Baselines3 PPO model: same network, same env
interface (bc_config.json). PPO starts from its actor and visual encoder with a new critic:

1. Critic warm-up: the actor (policy MLP, action head, std and encoder heads) is frozen for
   ``--warmup-steps`` while the value network learns from the BC policy's own rollouts. Without
   it, the untrained critic's noisy advantages can wreck the BC policy in the first updates.
2. Fine-tuning: the actor MLP, action head and std train (plus the encoder heads with
   ``--train-encoder``), with a small learning rate, a small clip range and ``target_kl`` early
   stopping to stay near the BC policy. The ResNet backbones stay frozen throughout.

Episodes start from the demonstrations' first poses (training episodes 0-44, +-3 deg jitter), with
the cube and box as in the demonstrations. Reward: potential-based progress (reach, lift, carry
toward the box, cube in the box) plus a success bonus (envs.So101PickPlaceEnv).

The closed-loop evaluation (deterministic, same starts every time) runs before training and every
``--eval-every`` steps; results go to TensorBoard and eval_history.json, and each evaluated
checkpoint is saved.

    python train_ppo.py --bc outputs/bc/<run>/bc_model.zip --run-name pilot_nominal
    tensorboard --logdir outputs/ppo/tb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

import train_bc
from envs import So101PickPlaceEnv, So101World, prepare_mounted

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "ppo"
TRAIN_EPISODES = set(range(45))


def demo_starts() -> SimpleNamespace:
    """First observation of every demonstration, in the shape train_bc.start_poses expects."""
    d = np.load(train_bc.DATA)  # lazy: only these arrays are read, not the images
    first = d["frame"] == 0
    return SimpleNamespace(starts=d["states"][first], start_episode=d["episode"][first].astype(np.int64))


def start_joints(starts: SimpleNamespace, world: So101World, episodes: set[int]) -> np.ndarray:
    return np.asarray(train_bc.start_poses(starts, world, episodes))


def eval_starts(starts: SimpleNamespace, world: So101World, seed: int = 10_000) -> dict[str, list]:
    """A fixed 40-episode subset of train_bc's 115-episode evaluation (same seeds, so comparable)."""
    val, train = train_bc.VAL_EPISODES, TRAIN_EPISODES
    return {
        "val_exact": train_bc.start_poses(starts, world, val),
        "val_jitter3deg": train_bc.start_poses(starts, world, val, 3.0, 4, seed),
        "train_exact": train_bc.start_poses(starts, world, train)[::3],
    }


def encoder_params(policy) -> list:
    """Trainable encoder layers shared by actor and critic (projection heads, state MLP)."""
    fe = policy.features_extractor
    modules = [fe.heads] if hasattr(fe, "heads") else [fe]
    if getattr(fe, "state_net", None) is not None:
        modules.append(fe.state_net)
    return [p for m in modules for p in m.parameters() if p.requires_grad]


class ActorFreeze:
    """Freeze or unfreeze everything the actor uses, leaving the value network trainable."""

    def __init__(self, policy):
        actor = [policy.mlp_extractor.policy_net, policy.action_net]
        self.params = [p for m in actor for p in m.parameters()] + [policy.log_std] + encoder_params(policy)
        self.trainable = [p.requires_grad for p in self.params]

    def set(self, frozen: bool) -> None:
        for p, trainable in zip(self.params, self.trainable):
            p.requires_grad_(trainable and not frozen)


class CurriculumCallback(BaseCallback):
    """Ramp the randomization strength linearly from 0 (at ``start``) to 1 (at ``end``) steps.

    Pilot finding: the BC policy scores 0% under full-range lighting or texture randomization, so
    randomized PPO starting at full strength gets no successful episodes to learn from.
    """

    def __init__(self, start: int, end: int):
        super().__init__()
        self.start, self.end = start, max(end, start + 1)

    def strength(self) -> float:
        return float(np.clip((self.num_timesteps - self.start) / (self.end - self.start), 0.0, 1.0))

    def _on_rollout_start(self) -> None:
        s = self.strength()
        self.training_env.set_attr("randomize_strength", s)
        self.logger.record("train/randomize_strength", s)

    def _on_step(self) -> bool:
        return True


class EvalCallback(BaseCallback):
    """Deterministic closed-loop evaluation every ``every`` steps; saves each evaluated model."""

    def __init__(self, starts: dict, run: Path, every: int, history: list, writer_dir: Path,
                 stress: dict | None = None):
        super().__init__()
        self.starts, self.run, self.every, self.history = starts, run, every, history
        # Extra groups under fixed factors (name -> (starts, factors)), reported separately so the
        # headline nominal numbers stay comparable across runs.
        self.stress = stress or {}
        self.next_eval = every
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(writer_dir))
        self.successes: list[bool] = []

    def evaluate(self, step: int) -> dict:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        groups = {name: train_bc.rollout_parallel(self.model, s) for name, s in self.starts.items()}
        n = sum(g["episodes"] for g in groups.values())
        row = {"step": step, **{m: sum(g[m] * g["episodes"] for g in groups.values()) / n
                                for m in ("success_rate", "lifted_rate", "in_box_rate")},
               "groups": {k: {m: g[m] for m in ("success_rate", "lifted_rate")} for k, g in groups.items()}}
        for name, (starts, factors) in self.stress.items():
            g = train_bc.rollout_parallel(self.model, starts, fixed_factors=factors)
            row.setdefault("stress", {})[name] = {m: g[m] for m in ("success_rate", "lifted_rate")}
            self.writer.add_scalar(f"stress/{name}_success_rate", g["success_rate"], step)
        self.history.append(row)
        for m in ("success_rate", "lifted_rate", "in_box_rate"):
            self.writer.add_scalar(f"eval/{m}", row[m], step)
        self.writer.flush()
        (self.run / "eval_history.json").write_text(json.dumps(self.history, indent=2))
        self.model.save(self.run / f"model_{step}.zip")
        stress = "".join(f"  | {k} {v['success_rate']:.0%}" for k, v in row.get("stress", {}).items())
        print(f"[eval @ {step:>8d}] success {row['success_rate']:.1%}  lifted {row['lifted_rate']:.1%}  "
              f"in box {row['in_box_rate']:.1%}  ({n} episodes){stress}", flush=True)
        return row

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.successes.append(bool(info.get("is_success", False)))
        if len(self.successes) >= 32:
            self.logger.record("rollout/success_rate", float(np.mean(self.successes)))
            self.successes.clear()
        if self.num_timesteps >= self.next_eval:
            self.evaluate(self.num_timesteps)
            self.next_eval += self.every
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bc", type=Path, required=True, help="BC checkpoint (bc_model.zip) to start from")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=50_000, help="critic-only steps before the actor trains")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256, help="rollout steps per env per update")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--init-std", type=float, default=None,
                        help="reset the action std (normalized units) for exploration; default keeps BC's")
    parser.add_argument("--episode-steps", type=int, default=750)
    parser.add_argument("--start-jitter", type=float, default=3.0, help="+- degrees on the demo start poses")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick-eval", action="store_true", help="evaluate on the 5 validation starts only (smoke tests)")
    parser.add_argument("--randomize", nargs="*", metavar="FACTOR",
                        choices=("texture", "lighting", "camera", "friction", "actuator"),
                        help="randomized condition: resample these factors every training episode "
                             "(no names = all four). Evaluation stays nominal; sweeps are separate.")
    parser.add_argument("--randomize-ranges", type=Path, default=HERE / "randomization_ranges.json",
                        help="randomization.Ranges overrides (e.g. per-factor strength caps from "
                             "calibrate_ranges.py); used when the file exists")
    parser.add_argument("--randomize-curriculum", type=float, default=0.6,
                        help="randomization strength ramps from 0 after the warm-up to full ranges at this "
                             "fraction of --steps (0 = full strength from the start)")
    parser.add_argument("--eval-latency", type=int, nargs="*", default=[], metavar="STEPS",
                        help="also evaluate the 20 val starts under these fixed control latencies")
    parser.add_argument("--train-encoder", action="store_true",
                        help="also fine-tune the encoder heads (shared with the new critic, whose early "
                             "gradients can disturb the actor's features); default keeps them at BC")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bc.load_config(args.bc)  # BC's observation/action settings define the env
    run = OUT / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "ppo_config.json").write_text(json.dumps(
        {"bc": str(args.bc), "obs": train_bc.OBS, "args": {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(args).items()}}, indent=2))
    (run / "bc_config.json").write_text(json.dumps({"obs": train_bc.OBS}, indent=2))  # for train_bc --eval

    world = So101World()
    starts = demo_starts()
    train_starts = start_joints(starts, world, TRAIN_EPISODES)
    evals = eval_starts(starts, world)
    if args.quick_eval:
        evals = {"val_exact": evals["val_exact"]}
    env_kwargs = train_bc.env_kwargs(episode_steps=args.episode_steps, start_joints=train_starts,
                                     start_jitter_deg=args.start_jitter, shaping_gamma=args.gamma)
    if args.randomize is not None:
        factors = args.randomize or ["texture", "lighting", "camera", "friction"]
        ranges = {}
        if args.randomize_ranges.is_file():  # calibrated by the pilot (calibrate_ranges.py)
            ranges = json.loads(args.randomize_ranges.read_text())["ranges"]
        env_kwargs["randomize"] = {"factors": factors, "ranges": ranges}
        print(f"randomized training: {factors}; ranges from "
              f"{args.randomize_ranges.name if ranges else 'randomization.Ranges defaults'}: {ranges}", flush=True)
        cfg = json.loads((run / "ppo_config.json").read_text())
        cfg["randomize"] = env_kwargs["randomize"]
        (run / "ppo_config.json").write_text(json.dumps(cfg, indent=2))
    venv = SubprocVecEnv([partial(So101PickPlaceEnv, **env_kwargs)] * args.n_envs)
    venv.seed(args.seed)

    model = PPO.load(
        args.bc, env=venv, device=device, seed=args.seed,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs,
        clip_range=args.clip, target_kl=args.target_kl, gamma=args.gamma, gae_lambda=0.95,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=str(OUT / "tb"), verbose=1,
    )
    fe = model.policy.features_extractor
    if hasattr(fe, "matches_imagenet") and not fe.matches_imagenet():
        sys.exit("ResNet backbones lost their ImageNet weights")
    std = model.policy.log_std.exp().detach().cpu().numpy()
    print(f"BC action std (normalized): {std.round(3).tolist()}")
    if args.init_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.init_std)))
        print(f"action std reset to {args.init_std}")

    history: list = []
    stress = {f"latency{k}": (evals["val_exact"] + evals.get("val_jitter3deg", []), {"actuator": {"latency_steps": k}})
              for k in args.eval_latency}
    eval_cb = EvalCallback(evals, run, args.eval_every, history, OUT / "tb" / f"{args.run_name}_eval", stress)
    eval_cb.init_callback(model)
    eval_cb.evaluate(0)  # the BC policy itself, same episodes: the bar PPO has to clear
    callback = eval_cb
    if args.randomize is not None and args.randomize_curriculum > 0:
        end = int(args.randomize_curriculum * args.steps)
        venv.set_attr("randomize_strength", 0.0)  # the first resets happen before any rollout callback
        callback = CallbackList([eval_cb, CurriculumCallback(args.warmup_steps, end)])
        print(f"randomization curriculum: strength 0 until {args.warmup_steps} steps, full at {end}", flush=True)

    if not args.train_encoder:  # actor MLP, action head and std fine-tune; the critic learns on BC features
        for p in encoder_params(model.policy):
            p.requires_grad_(False)
    freeze = ActorFreeze(model.policy)
    start = time.time()
    if args.warmup_steps:
        freeze.set(True)
        print(f"critic warm-up: {args.warmup_steps} steps with the actor frozen", flush=True)
        model.learn(args.warmup_steps, callback=callback, tb_log_name=args.run_name, reset_num_timesteps=True)
        freeze.set(False)
    model.learn(args.steps - args.warmup_steps, callback=callback, tb_log_name=args.run_name,
                reset_num_timesteps=not args.warmup_steps)
    eval_cb.evaluate(model.num_timesteps)
    model.save(run / "model_final.zip")
    venv.close()
    best = max(history, key=lambda r: (r["success_rate"], r["lifted_rate"]))
    print(f"done in {(time.time() - start) / 3600:.1f} h; BC (step 0): success {history[0]['success_rate']:.1%}; "
          f"best: step {best['step']} success {best['success_rate']:.1%} lifted {best['lifted_rate']:.1%}")


if __name__ == "__main__":
    main()
