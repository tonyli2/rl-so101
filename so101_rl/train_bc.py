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
from envs import So101PickPlaceEnv, prepare_mounted

HERE = Path(__file__).parent
DATA = HERE / "outputs" / "bc_data" / "sim-baseline.npz"
OUT = HERE / "outputs" / "bc"
HISTORY = 3
VAL_EPISODES = set(range(45, 50))


def env_kwargs(**extra) -> dict:
    """The env configuration BC is trained for (and PPO must use): fixed cube, absolute actions."""
    return {"action_mode": "absolute", "image": True, "history": HISTORY, "cube_xy_range": (0.0, 0.0),
            "include_targets": False, "wrist": True, **extra}


# make_bc_dataset.py stores [qpos_n(6), qvel(6), targets(6)] per frame; BC drops the targets.
STATE_COLUMNS = slice(0, 12)


class Demos:
    """Single-frame demo arrays on the GPU, stacked into env-style 3-step histories on demand."""

    def __init__(self, path: Path, device: torch.device):
        d = np.load(path)
        self.episode = d["episode"].astype(np.int64)
        frame = d["frame"].astype(np.int64)
        # History indices [t-2, t-1, t], repeating the first frame at episode start like env.reset.
        idx = np.arange(len(frame))
        self.hist = np.stack([idx - np.minimum(frame, k) for k in range(HISTORY - 1, -1, -1)], axis=1)
        self.states = torch.as_tensor(d["states"][:, STATE_COLUMNS], device=device)
        # Camera key -> N,3,H,W uint8 frames on the device.
        self.cameras = {"front": torch.as_tensor(d["images"], device=device).permute(0, 3, 1, 2).contiguous()}
        if "wrist_images" in d.files:
            self.cameras["wrist"] = torch.as_tensor(d["wrist_images"], device=device).permute(0, 3, 1, 2).contiguous()
        else:
            raise SystemExit("Dataset has no wrist images: re-run make_bc_dataset.py")
        self.actions = torch.as_tensor(d["actions"], device=device)
        self.starts = d["states"][frame == 0]  # first observation of each episode (normalized joints first)
        self.start_episode = self.episode[frame == 0]
        val = np.isin(self.episode, list(VAL_EPISODES))
        self.train_idx, self.val_idx = np.flatnonzero(~val), np.flatnonzero(val)
        self.device = device

    def batch(self, idx: np.ndarray, shift: int = 0) -> tuple[dict, torch.Tensor]:
        h = torch.as_tensor(self.hist[idx], device=self.device)                 # B,3
        obs = {"state": self.states[h].reshape(len(idx), -1)}                   # B,3*S
        for key, frames in self.cameras.items():
            image = frames[h].reshape(len(idx), -1, *frames.shape[2:])          # B,9,H,W
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


def evaluate_offline(model: PPO, demos: Demos, idx: np.ndarray, span: np.ndarray, batch: int = 512) -> dict:
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
    rmse = rmse_norm / 2 * span                      # normalized -> actuator radians
    units = np.rad2deg(rmse)
    units[5] = rmse_norm[5] / 2 * 100                 # gripper: percent of its range
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


def start_poses(demos: Demos, world, episodes: set[int], noise_deg: float, repeats: int, seed: int):
    """Start joints of the given demo episodes, optionally repeated with small random offsets."""
    rng = np.random.default_rng(seed)
    out = []
    for s, ep in zip(demos.starts, demos.start_episode):
        if ep not in episodes:
            continue
        q = (s[:6] + 1) / 2 * (world.hi - world.lo) + world.lo  # first 6 state entries: normalized joints
        for r in range(repeats):
            jitter = np.zeros(6)
            if noise_deg and r:
                jitter[:5] = np.deg2rad(rng.uniform(-noise_deg, noise_deg, 5))
            out.append(q + jitter)
    return out


def full_eval(model: PPO, demos: Demos, run: Path | None, seed: int) -> dict:
    world = So101PickPlaceEnv(**env_kwargs()).world
    val_starts = start_poses(demos, world, VAL_EPISODES, 0.0, 1, seed)
    train_starts = start_poses(demos, world, set(range(45)), 3.0, 1, seed)[::3]  # 15 training starts
    jittered = start_poses(demos, world, VAL_EPISODES, 3.0, 4, seed)            # val starts + 3 jitters
    world.close()
    video = run / "rollout_val0.mp4" if run else None
    result = {
        "val_starts": rollout_eval(model, val_starts, seed, video=video),
        "val_starts_jitter3deg": rollout_eval(model, jittered, seed + 100),
        "train_starts": rollout_eval(model, train_starts, seed + 200),
    }
    for name, r in result.items():
        print(f"{name:22s} success {r['success_rate']:.0%}  lifted {r['lifted_rate']:.0%}  "
              f"in box {r['in_box_rate']:.0%}  ({r['episodes']} episodes)")
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
    parser.add_argument("--eval", type=Path, help="evaluate a saved BC model")
    parser.add_argument("--view", type=Path, help="watch a saved BC model from the validation starts")
    args = parser.parse_args()
    if not DATA.is_file():
        sys.exit(f"{DATA} missing: run make_bc_dataset.py first")

    prepare_mounted(rebuild=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demos = Demos(DATA, device)
    print(f"{len(demos.train_idx)} train / {len(demos.val_idx)} val frames on {device}")

    if args.view or args.eval:
        model = PPO.load(args.view or args.eval, device=device)
        if args.view:
            view(model, demos)
        else:
            print(json.dumps(full_eval(model, demos, None, seed=10_000), indent=2)[:2000])
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = OUT / (args.run_name or f"{args.encoder}_seed{args.seed}")
    run.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(So101PickPlaceEnv, n_envs=1, seed=args.seed, env_kwargs=env_kwargs())
    policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -1.0}
    if args.encoder == "resnet18":
        policy_kwargs.update(features_extractor_class=ResNetExtractor,
                             features_extractor_kwargs={"freeze": True})
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
        val = evaluate_offline(model, demos, demos.val_idx, span)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), **val})
        writer.add_scalar("loss/train_nll", float(np.mean(losses)), epoch)
        writer.add_scalar("loss/val_nll", val["nll"], epoch)
        for name, value in zip(joint_names, val["action_rmse"]):
            writer.add_scalar(f"val_action_rmse/{name}" + ("_pct" if name == "gripper" else "_deg"), value, epoch)
        writer.flush()
        print(f"epoch {epoch:3d}  train nll {np.mean(losses):7.3f}  val nll {val['nll']:7.3f}  "
              f"val rmse deg {val['action_rmse'][:5]} grip% {val['action_rmse'][5]}  "
              f"({time.time() - start:.0f}s)", flush=True)
        if val["nll"] < best:
            best = val["nll"]
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
