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


def demo_starts() -> SimpleNamespace:
    """First observation of every demonstration, in the shape train_bc.start_poses expects."""
    d = np.load(train_bc.data_path())  # lazy: only these arrays are read, not the images
    first = d["frame"] == 0
    return SimpleNamespace(starts=d["states"][first], start_episode=d["episode"][first].astype(np.int64))


def start_joints(starts: SimpleNamespace, world: So101World, episodes: set[int]) -> np.ndarray:
    return np.asarray(train_bc.start_poses(starts, world, episodes))


def eval_starts(starts: SimpleNamespace, world: So101World, seed: int = 10_000) -> dict[str, list]:
    """About 40-55 fixed starts: every validation episode exactly and jittered (~20 jittered in
    all), plus ~15 training episodes. For Tony's 50 demos this is the original 5 + 20 + 15, a
    subset of train_bc's 115-episode evaluation (same seeds, so comparable)."""
    val, train = sorted(train_bc.val_episodes()), sorted(train_bc.train_episodes())
    copies = max(1, 20 // len(val))
    return {
        "val_exact": train_bc.start_poses(starts, world, set(val)),
        "val_jitter3deg": train_bc.start_poses(starts, world, set(val), 3.0, copies, seed),
        "train_exact": train_bc.start_poses(starts, world, set(train))[:: max(1, len(train) // 15)][:15],
    }


def encoder_params(policy) -> list:
    """Trainable encoder layers shared by actor and critic (projection heads, state MLP)."""
    fe = policy.features_extractor
    modules = [fe.heads] if hasattr(fe, "heads") else [fe]
    if getattr(fe, "state_net", None) is not None:
        modules.append(fe.state_net)
    extra = [fe.log_temp] if hasattr(fe, "log_temp") else []  # keypoint head's softmax temperature
    return [p for m in modules for p in m.parameters() if p.requires_grad] + [p for p in extra if p.requires_grad]


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

    def __init__(self, start: int, end: int, randomize: bool = True, cube: tuple | None = None):
        super().__init__()
        self.start, self.end = start, max(end, start + 1)
        self.randomize = randomize
        self.cube = cube  # (start, final) cube position half-range in metres, or None

    def strength(self) -> float:
        return float(np.clip((self.num_timesteps - self.start) / (self.end - self.start), 0.0, 1.0))

    def _on_rollout_start(self) -> None:
        s = self.strength()
        if self.randomize:
            self.training_env.set_attr("randomize_strength", s)
            self.logger.record("train/randomize_strength", s)
        if self.cube is not None:
            r = self.cube[0] + s * (self.cube[1] - self.cube[0])
            self.training_env.set_attr("cube_xy_range", np.array([r, r]))
            self.logger.record("train/cube_xy_range_m", r)

    def _on_step(self) -> bool:
        return True


class LayoutCurriculumCallback(BaseCallback):
    """Hard-layout sampling: track success per cube-position cell (env layout_grid x layout_grid)
    over each cell's recent training episodes, and draw cells that fail more often. Weights are
    ``mix`` uniform + (1 - mix) proportional to each cell's failure rate (unvisited cells count as
    failing), so every cell keeps being sampled."""

    def __init__(self, grid: int, mix: float = 0.5, window: int = 30):
        super().__init__()
        from collections import deque

        self.grid, self.mix = grid, mix
        self.outcomes = [deque(maxlen=window) for _ in range(grid * grid)]

    def weights(self) -> np.ndarray:
        fail = np.array([1.0 - np.mean(o) if o else 1.0 for o in self.outcomes])
        hard = fail / fail.sum() if fail.sum() > 0 else np.full(len(fail), 1 / len(fail))
        return self.mix / len(fail) + (1 - self.mix) * hard

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            cell = info.get("layout_cell", -1)
            if done and 0 <= cell < len(self.outcomes):
                self.outcomes[cell].append(float(info.get("is_success", False)))
        return True

    def _on_rollout_start(self) -> None:
        w = self.weights()
        self.training_env.env_method("set_layout_weights", w)
        seen = [np.mean(o) for o in self.outcomes if o]
        if seen:
            self.logger.record("layout/worst_cell_success", float(np.min(seen)))
            self.logger.record("layout/mean_cell_success", float(np.mean(seen)))
        self.logger.record("layout/max_weight_ratio", float(w.max() / w.min()))


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
                        choices=("texture", "lighting", "camera", "friction", "actuator", "appearance"),
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
    parser.add_argument("--eval-delay", nargs="*", default=[], metavar="A:J:I",
                        help="also evaluate the val starts under fixed delays: command A, joint "
                             "readings J, camera frames I steps (e.g. 2:2:1)")
    parser.add_argument("--relative-base", choices=("measured", "command", "resync"),
                        help="train with this action mapping (default: the BC config's); saved in the "
                             "run's bc_config.json, so deployment uses it too")
    parser.add_argument("--cube-curriculum", type=float, metavar="M",
                        help="start training with the cube within +-M metres of its spot and widen to the "
                             "demos' range (bc_config cube_xy_range) on the --randomize-curriculum schedule; "
                             "evaluations always use the full range")
    parser.add_argument("--resume", type=Path, metavar="CHECKPOINT",
                        help="continue this run from one of its model_<step>.zip checkpoints (policy, value "
                             "net and optimizer state): the step count, curriculum and evaluation schedule "
                             "carry on from <step>; no warm-up, std kept; eval history kept up to <step>")
    parser.add_argument("--hard-layouts", type=float, metavar="MIX", default=None,
                        help="sample cube positions by per-cell failure rate (LayoutCurriculumCallback); "
                             "MIX = uniform share of the draw (e.g. 0.5)")
    parser.add_argument("--train-encoder", action="store_true",
                        help="also fine-tune the encoder heads (shared with the new critic, whose early "
                             "gradients can disturb the actor's features); default keeps them at BC")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bc.load_config(args.bc)  # BC's observation/action settings define the env
    if args.relative_base:
        train_bc.OBS["relative_base"] = args.relative_base
    run = OUT / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "ppo_config.json").write_text(json.dumps(
        {"bc": str(args.bc), "obs": train_bc.OBS, "args": {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(args).items()}}, indent=2))
    (run / "bc_config.json").write_text(json.dumps({"obs": train_bc.OBS}, indent=2))  # for train_bc --eval

    world = So101World()
    starts = demo_starts()
    train_starts = start_joints(starts, world, train_bc.train_episodes())
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

    resume_step = 0
    if args.resume is not None:
        resume_step = int(args.resume.stem.split("_")[-1])
        if args.resume.parent.resolve() != run.resolve():
            sys.exit("--resume must be a checkpoint of this run (outputs/ppo/<run-name>/model_<step>.zip)")
    model = PPO.load(
        args.resume or args.bc, env=venv, device=device, seed=args.seed,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs,
        clip_range=args.clip, target_kl=args.target_kl, gamma=args.gamma, gae_lambda=0.95,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, tensorboard_log=str(OUT / "tb"), verbose=1,
    )
    fe = model.policy.features_extractor
    if hasattr(fe, "matches_imagenet") and not fe.matches_imagenet():
        sys.exit("ResNet backbones lost their ImageNet weights")
    std = model.policy.log_std.exp().detach().cpu().numpy()
    print(f"BC action std (normalized): {std.round(3).tolist()}")
    if args.init_std is not None and not resume_step:
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.init_std)))
        print(f"action std reset to {args.init_std}")

    history: list = []
    if resume_step:
        old = json.loads((run / "eval_history.json").read_text()) if (run / "eval_history.json").is_file() else []
        history = [r for r in old if r["step"] <= resume_step]
        model.num_timesteps = resume_step
    stress = {f"latency{k}": (evals["val_exact"] + evals.get("val_jitter3deg", []), {"actuator": {"latency_steps": k}})
              for k in args.eval_latency}
    for spec in args.eval_delay:
        a, j, i = (int(x) for x in spec.split(":"))
        stress[f"delay{a}{j}{i}"] = (evals["val_exact"] + evals.get("val_jitter3deg", []),
                                     {"actuator": {"latency_steps": a, "obs_latency_steps": j,
                                                   "image_latency_steps": i}})
    eval_cb = EvalCallback(evals, run, args.eval_every, history, OUT / "tb" / f"{args.run_name}_eval", stress)
    eval_cb.init_callback(model)
    if resume_step:
        eval_cb.next_eval = resume_step + args.eval_every
        print(f"resumed from {args.resume} at step {resume_step}", flush=True)
    else:
        eval_cb.evaluate(0)  # the BC policy itself, same episodes: the bar PPO has to clear
    callback = eval_cb
    randomized = args.randomize is not None and args.randomize_curriculum > 0
    cube = None
    if args.cube_curriculum is not None:  # widen the cube placement from this half-range to the demos'
        cube = (args.cube_curriculum, float(train_bc.OBS["cube_xy_range"]))
        venv.set_attr("cube_xy_range", np.array([cube[0], cube[0]]))
    if randomized or cube is not None:
        end = int(args.randomize_curriculum * args.steps)
        if randomized:  # the first resets happen before any rollout callback
            venv.set_attr("randomize_strength", float(np.clip((resume_step - args.warmup_steps)
                                                              / max(1, end - args.warmup_steps), 0, 1)))
        callback = CallbackList([eval_cb, CurriculumCallback(args.warmup_steps, end, randomized, cube)])
        print(f"curriculum ({'randomization' if randomized else ''}{' + ' if randomized and cube else ''}"
              f"{f'cube {cube[0]:.3f}->{cube[1]:.3f} m' if cube else ''}): starts at {args.warmup_steps} "
              f"steps, full at {end}", flush=True)

    if args.hard_layouts is not None:
        grid = int(venv.get_attr("layout_grid")[0])
        callback = CallbackList([callback, LayoutCurriculumCallback(grid, mix=args.hard_layouts)])
        print(f"hard-layout sampling: {grid}x{grid} cube cells, {args.hard_layouts:.0%} uniform", flush=True)

    if not args.train_encoder:  # actor MLP, action head and std fine-tune; the critic learns on BC features
        for p in encoder_params(model.policy):
            p.requires_grad_(False)
    freeze = ActorFreeze(model.policy)
    start = time.time()
    if args.warmup_steps and not resume_step:
        freeze.set(True)
        print(f"critic warm-up: {args.warmup_steps} steps with the actor frozen", flush=True)
        model.learn(args.warmup_steps, callback=callback, tb_log_name=args.run_name, reset_num_timesteps=True)
        freeze.set(False)
    if resume_step:
        model.learn(args.steps - resume_step, callback=callback, tb_log_name=args.run_name,
                    reset_num_timesteps=False)
    else:
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
