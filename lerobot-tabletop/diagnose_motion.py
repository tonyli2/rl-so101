"""Instrument native LeRobot ACT rollouts without changing checkpoints or the evaluator."""

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from learning.motion_metrics import analyze_motion  # noqa: E402
from learning.tabletop import TabletopConfig  # noqa: E402
from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTConfig, ACTPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import rollout  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from recording.archive import load_episode_archive, restore_state  # noqa: E402
from recording.schema import JOINTS, applied_action  # noqa: E402


def measured(world):
    dofs = np.array([world.model.joint(name).dofadr[0] for name in JOINTS])
    velocity = np.rad2deg(world.data.qvel[dofs])
    velocity[5] = world.data.qvel[dofs[5]] / (world.hi[5] - world.lo[5]) * 100
    return world.joint_state().copy(), velocity.copy()


def save_trace(output, label, targets, positions, velocities, queries, extra):
    targets, positions, velocities = map(np.asarray, (targets, positions, velocities))
    starts = np.zeros(len(targets), dtype=bool)
    starts[np.asarray(queries, dtype=int)] = True
    report = analyze_motion(targets, positions, velocities, starts)
    report.update(extra)
    np.savez_compressed(
        output / f"{label}.npz",
        targets=targets,
        positions=positions,
        velocities=velocities,
        chunk_starts=starts,
    )
    (output / f"{label}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"run": label, "steps": len(targets), **extra}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", type=int, default=[45, 46, 47, 48, 49])
    p.add_argument("--checkpoints", nargs="+", default=["010000", "100000"])
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "execute10", "ensemble", "fp32"],
        default=["baseline", "execute10", "ensemble"],
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reports = {}
    import mujoco

    metadata = {
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "seed": 1000,
        "fps": 30,
        "threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "episodes": args.episodes,
        "checkpoints": args.checkpoints,
        "variants": args.variants,
        "checkpoint_root": str(args.run.resolve()),
        "dataset": str(args.dataset.resolve()),
    }
    (args.output / "execution.json").write_text(json.dumps(metadata, indent=2))
    # Reuse the exact native evaluator loop and processors. Hooks only record traces.
    for checkpoint in args.checkpoints:
        path = args.run / "checkpoints" / checkpoint / "pretrained_model"
        for variant in (
            args.variants if checkpoint == args.checkpoints[-1] else ["baseline"]
        ):
            config = ACTConfig.from_pretrained(path)
            if variant == "execute10":
                config.n_action_steps = 10
            elif variant == "ensemble":
                config.n_action_steps = 1
                config.temporal_ensemble_coeff = 0.01
            set_seed(1000)
            policy = ACTPolicy.from_pretrained(path, config=config).cuda().eval()
            pre, post = make_pre_post_processors(config, pretrained_path=path)
            selected = (
                args.episodes
                if variant == "baseline" and checkpoint == args.checkpoints[-1]
                else args.episodes[:1]
            )
            for episode in selected:
                cfg = TabletopConfig(
                    reset_dataset_root=str(args.dataset.resolve()),
                    reset_episode=episode,
                )
                vector = cfg.create_envs(1)[cfg.type][0]
                sim = vector.envs[0]
                env_pre, env_post = make_env_pre_post_processors(cfg, config)
                targets, positions, velocities, queries = [], [], [], []
                requested, forces, contacts = [], [], []
                step_original = sim.step
                predict_original = policy.predict_action_chunk

                def predict(batch):
                    queries.append(len(targets))
                    return predict_original(batch)

                def step(action):
                    world = sim._ensure_world()
                    if not positions:
                        q, v = measured(world)
                        positions.append(q)
                        velocities.append(v)
                    obs, reward, term, trunc, info = step_original(action)
                    q, v = measured(world)
                    positions.append(q)
                    velocities.append(v)
                    requested.append(np.asarray(action).copy())
                    targets.append(applied_action(world)[0])
                    forces.append(world.data.actuator_force[world.actuators].copy())
                    contacts.append(world.data.ncon)
                    return obs, reward, term, trunc, info

                policy.predict_action_chunk = predict
                sim.step = step
                try:
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            "cuda", dtype=torch.float16, enabled=variant != "fp32"
                        ),
                    ):
                        result = rollout(
                            vector, policy, env_pre, env_post, pre, post, seeds=[1000]
                        )
                    label = f"{checkpoint}_{variant}_ep{episode}"
                    # Ensemble queries every step are not independent chunk handoffs.
                    boundaries = queries if variant != "ensemble" else [0]
                    report = save_trace(
                        args.output,
                        label,
                        targets,
                        positions,
                        velocities,
                        boundaries,
                        {
                            "checkpoint": checkpoint,
                            "variant": variant,
                            "episode": episode,
                            "success": bool(result["success"].any()),
                            "query_steps": queries,
                            "clipped_values": int(
                                np.sum(np.abs(np.asarray(requested) - targets) > 1e-3)
                            ),
                            "max_abs_actuator_force": np.max(
                                np.abs(forces), axis=0
                            ).tolist(),
                            "max_contacts": int(max(contacts)),
                        },
                    )
                    reports[label] = report
                finally:
                    policy.predict_action_chunk = predict_original
                    vector.close()
            del policy
            torch.cuda.empty_cache()
    # Recorded commands and exact saved states give a human baseline without new collection.
    for episode in args.episodes:
        manifest, arrays, model = load_episode_archive(args.dataset, episode)
        import mujoco

        data = mujoco.MjData(model)
        targets = arrays["applied_actions"]
        positions, velocities = [], []
        for state in arrays["states"]:
            restore_state(model, data, state)
            joint_ids = [model.joint(name).id for name in JOINTS]
            q = np.rad2deg(data.qpos[model.jnt_qposadr[joint_ids]])
            v = np.rad2deg(data.qvel[model.jnt_dofadr[joint_ids]])
            lo, hi = model.actuator_ctrlrange[model.actuator("gripper").id]
            q[5] = (data.qpos[model.jnt_qposadr[joint_ids[5]]] - lo) / (hi - lo) * 100
            v[5] = data.qvel[model.jnt_dofadr[joint_ids[5]]] / (hi - lo) * 100
            positions.append(q)
            velocities.append(v)
        label = f"demo_ep{episode}"
        reports[label] = save_trace(
            args.output,
            label,
            targets,
            positions,
            velocities,
            [0],
            {"variant": "demonstration", "episode": episode},
        )
    (args.output / "summary.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
