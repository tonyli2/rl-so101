"""Behavior cloning on Tony's demonstrations, trained as the PPO policy network itself.

The policy is Stable-Baselines3's actor-critic for ``So101PickPlaceEnv`` (overhead camera CNN +
state MLP, diagonal Gaussian actions). BC maximizes the log-likelihood of the demonstrated
actions; the critic is left untrained. The saved ``bc_model.zip`` is a normal PPO checkpoint, so
stage 2 can start from exactly these weights (proposal: "PPO then starts from the trained policy
and visual encoder, with a new critic").

Data: ``make_bc_dataset.py`` output. Split: episodes 0-44 train, 45-49 validation (Tony's).
Inputs: joint positions and velocities (no previous commands: those let BC copy its last action),
overhead and wrist cameras, each with its own CNN.
Observations get the same 3-step history as the env; image batches get a random shift of up to
``--shift`` pixels (the same shift for all stacked frames) against overfitting to exact pixels.

    python train_bc.py                       # frozen ResNet18 per camera; train, then evaluate closed-loop
    python train_bc.py --encoder nature      # small CNN from scratch (earlier baseline)
    python train_bc.py --epochs 60 --seed 1
    python train_bc.py --eval outputs/bc/resnet18_seed0/bc_model.zip
    python train_bc.py --view outputs/bc/resnet18_seed0/bc_model.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

from encoders import ResNetExtractor
from envs import So101PickPlaceEnv, So101World, prepare_mounted

HERE = Path(__file__).parent
DATA = HERE / "outputs" / "bc_data" / "sim-baseline.npz"
OUT = HERE / "outputs" / "bc"
VAL_EPISODES = set(range(45, 50))
# Observation/action settings of the current run; saved as bc_config.json next to each model so
# --eval and --view rebuild the same env. Runs without that file (or keys) used these defaults.
OBS = {"history": 3, "velocities": True, "action": "absolute", "relative_scale": None, "trim_idle": False}
IDLE_DEG = 1.0  # leading frames whose command is within this of the measured arm count as idle


def configure(history: int = 3, velocities: bool = True, action: str = "absolute",
              relative_scale: list | None = None, trim_idle: bool = False) -> None:
    OBS.update(history=history, velocities=velocities, action=action,
               relative_scale=relative_scale, trim_idle=trim_idle)


def load_config(model_path: Path) -> None:
    path = Path(model_path).parent / "bc_config.json"
    configure(**(json.loads(path.read_text())["obs"] if path.is_file() else {}))
    print(f"settings: {OBS}")


def env_kwargs(**extra) -> dict:
    """The env configuration BC is trained for (and PPO must use): fixed cube, the run's action mode."""
    kwargs = {"action_mode": OBS["action"], "image": True, "history": OBS["history"], "cube_xy_range": (0.0, 0.0),
              "include_targets": False, "wrist": True, "include_velocities": OBS["velocities"]}
    if OBS["action"] == "relative":
        kwargs["relative_scale"] = OBS["relative_scale"]
    return {**kwargs, **extra}


def state_columns() -> slice:
    """make_bc_dataset.py stores [qpos_n(6), qvel(6), targets(6)] per frame; BC drops the targets."""
    return slice(0, 12 if OBS["velocities"] else 6)


class Demos:
    """Single-frame demo arrays, stacked into env-style histories on demand.

    ``perturbed``: also load make_bc_dataset.py's noise-augmented copies of the training frames
    (training only; validation stays clean). Each copy is its own single-frame sample, so this
    requires history 1. With it, images live in CPU memory (4x the frames do not fit on the GPU).
    """

    def __init__(self, path: Path, device: torch.device, perturbed: Path | None = None):
        d = np.load(path)
        clean = len(d["frame"])
        arrays = {k: d[k] for k in ("states", "images", "wrist_images", "actions", "episode", "frame")
                  if k in d.files}
        if "wrist_images" not in arrays:
            raise SystemExit("Dataset has no wrist images: re-run make_bc_dataset.py")
        if perturbed is not None:
            if OBS["history"] != 1:
                raise SystemExit("Perturbed copies are single frames: use --history 1")
            p = np.load(perturbed)
            arrays = {k: np.concatenate([v, p[k]]) for k, v in arrays.items()}
        self.episode = arrays["episode"].astype(np.int64)
        frame = arrays["frame"].astype(np.int64)
        n = len(frame)
        self.is_perturbed = np.arange(n) >= clean
        # History indices [t-h+1, ..., t] within the clean episode, repeating the first frame at the
        # start like env.reset. Perturbed copies (history 1) index only themselves.
        idx = np.arange(n)
        history = OBS["history"]
        self.hist = np.stack([idx - np.minimum(frame, k) for k in range(history - 1, -1, -1)], axis=1)
        self.hist[self.is_perturbed] = idx[self.is_perturbed, None]
        self.device = device
        self.image_device = torch.device("cpu") if perturbed is not None else device
        self.states = torch.as_tensor(arrays["states"][:, state_columns()], device=device)
        # Camera key -> N,3,H,W uint8 frames.
        self.cameras = {
            key: torch.as_tensor(arrays[src], device=self.image_device).permute(0, 3, 1, 2).contiguous()
            for key, src in (("front", "images"), ("wrist", "wrist_images"))
        }
        first = (frame == 0) & ~self.is_perturbed
        self.starts = arrays["states"][first]  # first observation of each episode (normalized joints first)
        self.start_episode = self.episode[first]
        val = np.isin(self.episode, list(VAL_EPISODES))
        if (val & self.is_perturbed).any():
            raise SystemExit("Perturbed data contains validation episodes")

        # Labels. Stored actions are absolute targets in [-1, 1] over each actuator's range; the
        # first six state values are the measured joints on the same scale (nudged ones for copies).
        world = So101World()
        lo, hi = world.lo, world.hi
        self.span = hi - lo
        joints = (arrays["states"][:, :6] + 1) / 2 * self.span + lo
        target = (arrays["actions"] + 1) / 2 * self.span + lo
        motion = target - joints                                   # radians the command asks to move
        # Leading idle frames: the human had not started moving (command ~ measured arm).
        moving = np.rad2deg(np.abs(motion[:, :5])).max(1) > IDLE_DEG
        lead = {ep: int(np.argmax(moving[(self.episode == ep) & ~self.is_perturbed]))
                for ep in np.unique(self.episode)}
        idle = frame < np.array([lead[ep] for ep in self.episode])
        keep = ~idle if OBS["trim_idle"] else np.ones(n, bool)
        if OBS["action"] == "relative":
            if OBS["relative_scale"] is None:  # 99th percentile of real motion per joint (training, clean)
                ref = ~val & ~self.is_perturbed & keep
                OBS["relative_scale"] = np.maximum(np.percentile(np.abs(motion[ref]), 99, axis=0),
                                                   np.deg2rad(0.5)).tolist()
            scale = np.asarray(OBS["relative_scale"])
            labels = np.clip(motion / scale, -1, 1)
            self.half_range = scale            # one unit of action = scale radians
        else:
            labels = arrays["actions"]
            self.half_range = self.span / 2
        self.actions = torch.as_tensor(labels.astype(np.float32), device=device)
        self.train_idx = np.flatnonzero(~val & keep)
        self.val_idx = np.flatnonzero(val & keep)
        print(f"frames: {clean} clean + {int(self.is_perturbed.sum())} perturbed; images on {self.image_device}; "
              f"actions {OBS['action']}; idle start frames {'trimmed' if OBS['trim_idle'] else 'kept'} "
              f"({int((idle & ~self.is_perturbed).sum())} clean idle frames)")
        if OBS["action"] == "relative":
            print("relative action scale (deg; gripper rad->deg too):", np.rad2deg(scale).round(2).tolist())

    def batch(self, idx: np.ndarray, shift: int = 0) -> tuple[dict, torch.Tensor]:
        h = torch.as_tensor(self.hist[idx])                                       # B,history
        obs = {"state": self.states[h.to(self.device)].reshape(len(idx), -1)}     # B,history*S
        for key, frames in self.cameras.items():
            image = frames[h.to(self.image_device)].reshape(len(idx), -1, *frames.shape[2:])
            image = image.to(self.device, non_blocking=True)                     # B,3*history,H,W
            obs[key] = random_shift(image, shift) if shift else image
        return obs, self.actions[torch.as_tensor(idx, device=self.device)]


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style augmentation: replicate-pad and crop back at a random offset per sample."""
    n, c, h, w = images.shape
    padded = F.pad(images.float(), (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    rows = (torch.arange(h, device=images.device)[None, :] + dy[:, None])     # n,h
    cols = (torch.arange(w, device=images.device)[None, :] + dx[:, None])     # n,w
    out = padded[torch.arange(n, device=images.device)[:, None, None, None],
                 torch.arange(c, device=images.device)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
    return out.to(images.dtype)


def evaluate_offline(model: PPO, demos: Demos, idx: np.ndarray, batch: int = 512) -> dict:
    """Validation NLL and per-joint RMSE of the mean action (arm joints in degrees, gripper in %)."""
    policy = model.policy
    policy.set_training_mode(False)
    nll, se = [], []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            obs, act = demos.batch(idx[i : i + batch])
            dist = policy.get_distribution(obs)
            nll.append(-dist.log_prob(act).cpu())
            se.append(((dist.distribution.mean.clamp(-1, 1) - act) ** 2).cpu())
    rmse_norm = torch.cat(se).mean(0).sqrt().numpy()
    rmse = rmse_norm * demos.half_range              # normalized -> actuator radians (either action mode)
    units = np.rad2deg(rmse)
    units[5] = rmse[5] / demos.span[5] * 100         # gripper: percent of its range
    return {"nll": float(torch.cat(nll).mean()), "action_rmse": units.round(3).tolist()}


def rollout_eval(model: PPO, starts: list[np.ndarray], seed: int, episode_steps: int = 900,
                 video: Path | None = None) -> dict:
    """Closed-loop episodes in the env from given start joints (radians); deterministic actions."""
    env = So101PickPlaceEnv(**env_kwargs(episode_steps=episode_steps))
    results, frames = [], []
    for k, q in enumerate(starts):
        obs, _ = env.reset(seed=seed + k, options={"joints": q})
        lifted = in_box = False
        for _ in range(episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            lifted |= info["cube_lifted"]
            in_box |= info["cube_in_box"]
            if video is not None and k == 0:
                frames.append(np.concatenate([env.world.render(env.camera, 240, 320),
                                              env.world.render(env.WRIST_CAMERA, 240, 320)], axis=1))
            if terminated or truncated:
                break
        results.append({"success": bool(info["is_success"]), "lifted": bool(lifted), "in_box": bool(in_box),
                         "steps": env._steps})
    env.close()
    if video is not None and frames:
        import imageio.v2 as imageio

        imageio.mimsave(video, frames, fps=30)
    n = len(results)
    return {
        "episodes": n,
        "success_rate": sum(r["success"] for r in results) / n,
        "lifted_rate": sum(r["lifted"] for r in results) / n,
        "in_box_rate": sum(r["in_box"] for r in results) / n,
        "per_episode": results,
    }


def start_poses(demos: Demos, world, episodes: set[int], noise_deg: float = 0.0, copies: int = 1,
                seed: int = 0) -> list[np.ndarray]:
    """Start joints of the given demo episodes: exact (noise 0) or ``copies`` distinct jittered ones."""
    rng = np.random.default_rng(seed)
    out = []
    for s, ep in zip(demos.starts, demos.start_episode):
        if ep not in episodes:
            continue
        q = (s[:6] + 1) / 2 * (world.hi - world.lo) + world.lo  # first 6 state entries: normalized joints
        for _ in range(copies):
            jitter = np.zeros(6)
            if noise_deg:
                jitter[:5] = np.deg2rad(rng.uniform(-noise_deg, noise_deg, 5))
            out.append(q + jitter)
    return out


def rollout_parallel(model: PPO, starts: list[np.ndarray], n_envs: int = 8, episode_steps: int = 900,
                     fixed_factors: dict | None = None) -> dict:
    """Closed-loop episodes from the given start joints, ``n_envs`` at a time in subprocesses
    (optionally under fixed randomization factors, e.g. ``{"actuator": {"latency_steps": 2}}``)."""
    from functools import partial

    from stable_baselines3.common.vec_env import SubprocVecEnv

    kwargs = env_kwargs(episode_steps=episode_steps)
    if fixed_factors is not None:
        kwargs["fixed_factors"] = fixed_factors
    venv = SubprocVecEnv([partial(So101PickPlaceEnv, **kwargs)] * n_envs)
    results = []
    try:
        for c in range(0, len(starts), n_envs):
            chunk = starts[c : c + n_envs]
            padded = chunk + [chunk[-1]] * (n_envs - len(chunk))  # extra envs repeat a start; ignored
            venv.set_options([{"joints": q} for q in padded])
            obs = venv.reset()
            done = np.zeros(n_envs, bool)
            lifted, in_box = np.zeros(n_envs, bool), np.zeros(n_envs, bool)
            depths = [[] for _ in range(n_envs)]  # cube position along the fingers while lifted
            outcome = [None] * n_envs
            steps = 0
            while not done.all() and steps < episode_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = venv.step(action)
                steps += 1
                for i, info in enumerate(infos):
                    if done[i]:
                        continue  # this env auto-reset into another episode: ignore it
                    lifted[i] |= info["cube_lifted"]
                    in_box[i] |= info["cube_in_box"]
                    if np.isfinite(info.get("grasp_depth_cm", np.nan)):
                        depths[i].append(info["grasp_depth_cm"])
                    if dones[i]:
                        done[i] = True
                        outcome[i] = {"success": bool(info["is_success"]), "lifted": bool(lifted[i]),
                                      "in_box": bool(in_box[i]), "steps": steps,
                                      "grasp_depth_cm": float(np.median(depths[i])) if depths[i] else None,
                                      "lifted_steps": len(depths[i])}
            results.extend(outcome[: len(chunk)])
    finally:
        venv.close()
    n = len(results)
    return {
        "episodes": n,
        "success_rate": sum(r["success"] for r in results) / n,
        "lifted_rate": sum(r["lifted"] for r in results) / n,
        "in_box_rate": sum(r["in_box"] for r in results) / n,
        "per_episode": results,
    }


def full_eval(model: PPO, demos: Demos, run: Path | None, seed: int) -> dict:
    """115 distinct closed-loop episodes from demonstration starts (cube and box as in the demos)."""
    torch.backends.cudnn.deterministic = True   # repeatable inference: identical starts, identical outcomes
    torch.backends.cudnn.benchmark = False
    world = So101PickPlaceEnv(**env_kwargs()).world
    val, train = VAL_EPISODES, set(range(45))
    groups = {
        "val_exact": start_poses(demos, world, val),                                # 5
        "val_jitter3deg": start_poses(demos, world, val, 3.0, 4, seed),             # 20
        "train_exact": start_poses(demos, world, train),                            # 45
        "train_jitter3deg": start_poses(demos, world, train, 3.0, 1, seed + 1),     # 45
    }
    world.close()
    if run is not None:  # one recorded episode for a visual check
        rollout_eval(model, groups["val_exact"][:1], seed, video=run / "rollout_val0.mp4")
    result = {name: rollout_parallel(model, starts) for name, starts in groups.items()}
    total = sum(r["episodes"] for r in result.values())
    for name, r in result.items():
        print(f"{name:18s} success {r['success_rate']:5.1%}  lifted {r['lifted_rate']:5.1%}  "
              f"in box {r['in_box_rate']:5.1%}  ({r['episodes']} episodes)")
    overall = {m: sum(r[m] * r["episodes"] for r in result.values()) / total
               for m in ("success_rate", "lifted_rate", "in_box_rate")}
    print(f"{'ALL':18s} success {overall['success_rate']:5.1%}  lifted {overall['lifted_rate']:5.1%}  "
          f"in box {overall['in_box_rate']:5.1%}  ({total} episodes)")
    episodes = [e for r in result.values() for e in r["per_episode"] if e.get("grasp_depth_cm") is not None]
    for label, group in (("successful", [e for e in episodes if e["success"]]),
                         ("failed", [e for e in episodes if not e["success"]])):
        if group:
            depth = np.median([e["grasp_depth_cm"] for e in group])
            held = np.median([e["lifted_steps"] for e in group]) / 30
            overall[f"grasp_depth_cm_{label}"] = float(depth)
            print(f"grasp depth, {label:10s} carries: median {depth:+.2f} cm along the fingers "
                  f"(demos: +0.15), held {held:.1f} s  ({len(group)} episodes)")
    result["all"] = {"episodes": total, **overall}
    return result


def view(model: PPO, demos: Demos) -> None:
    import mujoco.viewer

    env = So101PickPlaceEnv(**env_kwargs())
    starts = start_poses(demos, env.world, VAL_EPISODES, 0.0, 1, 0)
    k = 0
    obs, _ = env.reset(options={"joints": starts[k]})
    with mujoco.viewer.launch_passive(env.world.model, env.world.data) as viewer:
        while viewer.is_running():
            tick = time.monotonic()
            action, _ = model.predict(obs, deterministic=True)
            with viewer.lock():
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"start {k}: {'SUCCESS' if info['is_success'] else 'fail'} after {env._steps} steps")
                    k = (k + 1) % len(starts)
                    obs, _ = env.reset(options={"joints": starts[k]})
            viewer.sync()
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--shift", type=int, default=4, help="random image shift in pixels (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", choices=("resnet18", "nature"), default="resnet18",
                        help="resnet18: frozen ImageNet ResNet18 per camera; nature: small CNN from scratch")
    parser.add_argument("--run-name", help="output folder under outputs/bc (default <encoder>_seed<N>)")
    parser.add_argument("--state-net", action=argparse.BooleanOptionalAction, default=True,
                        help="balanced branches: state MLP + LayerNorm per branch (resnet18 only)")
    parser.add_argument("--history", type=int, default=3, help="observation history length (1 = current only)")
    parser.add_argument("--velocities", action=argparse.BooleanOptionalAction, default=True,
                        help="include joint velocities in the state")
    parser.add_argument("--perturbed", action="store_true",
                        help="add the noise-augmented training copies (make_bc_dataset.py --perturb K); needs --history 1")
    parser.add_argument("--action", choices=("relative", "absolute"), default="relative",
                        help="relative: motion from the measured joints; absolute: target positions")
    parser.add_argument("--trim-idle", action=argparse.BooleanOptionalAction, default=True,
                        help="drop each demonstration's leading idle frames (human not moving yet)")
    parser.add_argument("--eval", type=Path, help="evaluate a saved BC model")
    parser.add_argument("--view", type=Path, help="watch a saved BC model from the validation starts")
    args = parser.parse_args()
    if not DATA.is_file():
        sys.exit(f"{DATA} missing: run make_bc_dataset.py first")

    prepare_mounted(rebuild=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.view or args.eval:
        load_config(args.view or args.eval)
    else:
        configure(args.history, args.velocities, args.action, None, args.trim_idle)
    perturbed = None
    if args.perturbed and not (args.view or args.eval):
        perturbed = DATA.with_name("sim-baseline-perturbed.npz")
        if not perturbed.is_file():
            sys.exit(f"{perturbed} missing: run make_bc_dataset.py --perturb 3")
    demos = Demos(DATA, device, perturbed=perturbed)
    print(f"{len(demos.train_idx)} train / {len(demos.val_idx)} val frames on {device}")

    if args.view or args.eval:
        model = PPO.load(args.view or args.eval, device=device)
        if args.view:
            view(model, demos)
        else:
            result = full_eval(model, demos, None, seed=10_000)
            out = Path(args.eval).parent / "closed_loop_eval.json"
            out.write_text(json.dumps(result, indent=2))
            print(f"saved {out}")
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = OUT / (args.run_name or f"{args.encoder}_seed{args.seed}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "bc_config.json").write_text(json.dumps({"obs": OBS, "args": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, indent=2))
    venv = make_vec_env(So101PickPlaceEnv, n_envs=1, seed=args.seed, env_kwargs=env_kwargs())
    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -1.0}
    if args.encoder == "resnet18":
        extractor_kwargs = {"freeze": True}
        if args.state_net:
            extractor_kwargs["state_dim_out"] = 256  # same width as each camera branch
        policy_kwargs.update(features_extractor_class=ResNetExtractor, features_extractor_kwargs=extractor_kwargs)
    model = PPO("MultiInputPolicy", venv, seed=args.seed, device=device, verbose=0, policy_kwargs=policy_kwargs)
    if args.encoder == "resnet18" and not model.policy.features_extractor.matches_imagenet():
        sys.exit("ResNet backbones lost their ImageNet weights during policy construction")
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"encoder {args.encoder}: {trainable / 1e6:.2f}M trainable of {total / 1e6:.2f}M parameters", flush=True)
    span = venv.envs[0].unwrapped.world.hi - venv.envs[0].unwrapped.world.lo
    venv.close()
    policy = model.policy
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = len(demos.train_idx) // args.batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)

    writer = SummaryWriter(str(OUT / "tb" / run.name))  # tensorboard --logdir outputs/bc/tb
    joint_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    history, best, start = [], np.inf, time.time()
    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        order = rng.permutation(demos.train_idx)
        losses = []
        for i in range(steps_per_epoch):
            obs, act = demos.batch(order[i * args.batch : (i + 1) * args.batch], shift=args.shift)
            dist = policy.get_distribution(obs)
            loss = -dist.log_prob(act).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val = evaluate_offline(model, demos, demos.val_idx)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), **val})
        writer.add_scalar("loss/train_nll", float(np.mean(losses)), epoch)
        writer.add_scalar("loss/val_nll", val["nll"], epoch)
        for name, value in zip(joint_names, val["action_rmse"]):
            writer.add_scalar(f"val_action_rmse/{name}" + ("_pct" if name == "gripper" else "_deg"), value, epoch)
        writer.flush()
        print(f"epoch {epoch:3d}  train nll {np.mean(losses):7.3f}  val nll {val['nll']:7.3f}  "
              f"val rmse deg {val['action_rmse'][:5]} grip% {val['action_rmse'][5]}  "
              f"({time.time() - start:.0f}s)", flush=True)
        # Validation frames are clean, so with noise-augmented training their loss rewards staying
        # close to clean behavior, not recovering: it picked the epoch-1 model (0% success) over
        # epoch 20 (12%). Those runs keep the final epoch instead.
        if args.perturbed or val["nll"] < best:
            best = min(best, val["nll"])
            model.save(run / "bc_model.zip")
    model.save(run / "bc_model_last.zip")

    best_model = PPO.load(run / "bc_model.zip", device=device)
    train_seconds = round(time.time() - start, 1)
    closed_loop = full_eval(best_model, demos, run, seed=10_000)
    for name, r in closed_loop.items():
        for metric in ("success_rate", "lifted_rate", "in_box_rate"):
            writer.add_scalar(f"closed_loop/{name}/{metric}", r[metric], args.epochs)
    writer.close()
    result = {
        "seed": args.seed, "epochs": args.epochs, "train_seconds": train_seconds,
        "best_val_nll": best, "history": history, "closed_loop": closed_loop,
    }
    (run / "bc_eval.json").write_text(json.dumps(result, indent=2))
    print(f"saved {run / 'bc_model.zip'} (best val nll {best:.3f}); rollout video {run / 'rollout_val0.mp4'}")


if __name__ == "__main__":
    main()
