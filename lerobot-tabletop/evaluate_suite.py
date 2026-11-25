"""Evaluate fixed scenarios with native LeRobot ACT inference and frozen resets."""

import argparse
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["HF_HUB_OFFLINE"] = "1"

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from learning.robustness import ScenarioEnv  # noqa: E402
from learning.tabletop import TabletopConfig  # noqa: E402
from lerobot.envs import make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import FreezeAfterEpisodeEnd  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTConfig, ACTPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import rollout  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402


def summarize(bundle, checkpoint, records):
    primary = [r for r in records if r["primary"]]
    result = {
        "checkpoint": checkpoint,
        "completed_cases": len(records),
        "primary_cases": len(primary),
    }
    for label, rows in [("all_primary", primary)] + [
        (f, [r for r in primary if r["family"] == f])
        for f in sorted({r["family"] for r in primary})
    ]:
        result[label] = {
            "cases": len(rows),
            "task_successes": sum(r["task_success"] for r in rows),
            "collision_free_successes": sum(r["collision_free_success"] for r in rows),
            "cases_with_self_contact": sum(
                r["first_contact"] is not None for r in rows
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Separate results for another policy run, reusing this exact suite",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[f"{i:06d}" for i in range(10000, 100001, 10000)],
    )
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    manifest_path = args.suite / "manifest.json"
    suite = json.loads(manifest_path.read_text())
    assert (
        hashlib.sha256((args.suite / "states.npz").read_bytes()).hexdigest()
        == suite["states_sha256"]
    )
    assert (
        hashlib.sha256((args.suite / "model.mjb").read_bytes()).hexdigest()
        == suite["model_sha256"]
    )
    if not suite.get("bin_reachability_checked"):
        raise ValueError("Complete bin reachability preflight first")
    suite_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_output = args.output or args.suite
    run_output.mkdir(parents=True, exist_ok=True)
    if run_output.resolve() != args.suite.resolve():
        copied_manifest = run_output / "manifest.json"
        if (
            copied_manifest.exists()
            and copied_manifest.read_bytes() != manifest_path.read_bytes()
        ):
            raise ValueError("Output directory belongs to another scenario suite")
        copied_manifest.write_bytes(manifest_path.read_bytes())
        (run_output / "suite_source.txt").write_text(str(args.suite.resolve()) + "\n")
    output = run_output / "results"
    output.mkdir(exist_ok=True)
    conditions = {
        "suite_sha256": suite_hash,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "precision": "fp16",
        "tf32": True,
        "temporal_ensemble_coeff": 0.01,
        "n_action_steps": 1,
        "seed": 1000,
        "threads": 4,
    }
    conditions_path = run_output / "execution.json"
    if (
        conditions_path.exists()
        and json.loads(conditions_path.read_text()) != conditions
    ):
        raise ValueError("Existing suite run uses different inference conditions")
    conditions_path.write_text(json.dumps(conditions, indent=2))
    all_summaries = []
    for checkpoint in args.checkpoints:
        path = args.run / "checkpoints" / checkpoint / "pretrained_model"
        cfg = ACTConfig.from_pretrained(path)
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = 0.01
        config_hash = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
        weights_hash = hashlib.sha256(
            (path / "model.safetensors").read_bytes()
        ).hexdigest()
        set_seed(1000)
        policy = ACTPolicy.from_pretrained(path, config=cfg).cuda().eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=path)
        env_pre, env_post = make_env_pre_post_processors(TabletopConfig(), cfg)
        checkpoint_dir = output / checkpoint
        checkpoint_dir.mkdir(exist_ok=True)
        records = []
        start = time.monotonic()
        for first in range(0, len(suite["cases"]), args.batch_size):
            indices = list(
                range(first, min(first + args.batch_size, len(suite["cases"])))
            )
            # Resume only complete same-suite groups, never silently mix conditions.
            saved = [
                checkpoint_dir / (suite["cases"][i]["id"] + ".json") for i in indices
            ]
            if all(p.exists() for p in saved):
                group = [json.loads(p.read_text()) for p in saved]
                if all(
                    r["suite_sha256"] == suite_hash
                    and r["batch_size"] == args.batch_size
                    and r["weights_sha256"] == weights_hash
                    and r["config_sha256"] == config_hash
                    for r in group
                ):
                    records.extend(group)
                    continue
                raise ValueError("Existing result uses different suite or batch size")
            factories = [partial(ScenarioEnv, args.suite, i) for i in indices]
            vector = gym.vector.SyncVectorEnv(
                [partial(FreezeAfterEpisodeEnd, fn()) for fn in factories],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            )
            try:
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.float16),
                ):
                    rollout(
                        vector,
                        policy,
                        env_pre,
                        env_post,
                        pre,
                        post,
                        seeds=[1000] * len(indices),
                    )
                for i, wrapped in zip(indices, vector.envs):
                    env = wrapped.env
                    case = suite["cases"][i]
                    record = {
                        "checkpoint": checkpoint,
                        "scenario_id": case["id"],
                        "family": case["family"],
                        "primary": case["primary"],
                        "suite_sha256": suite_hash,
                        "batch_size": args.batch_size,
                        "group_start": first,
                        "group_size": len(indices),
                        "config_sha256": config_hash,
                        "weights_sha256": weights_hash,
                        **env.outcome,
                    }
                    if not env.outcome:
                        raise RuntimeError("Rollout did not reach a terminal outcome")
                    target = np.asarray(env.targets)
                    position = np.asarray(env.positions)
                    error = target - position[1:]
                    record["tracking_error_rms_per_joint"] = np.sqrt(
                        np.mean(error**2, axis=0)
                    ).tolist()
                    record["tracking_error_max_per_joint"] = np.max(
                        np.abs(error), axis=0
                    ).tolist()
                    record["target_acceleration_rms_per_joint"] = (
                        np.sqrt(np.mean(np.diff(target, n=2, axis=0) ** 2, axis=0))
                        * 900
                    ).tolist()
                    record["state_motion_acceleration_rms_per_joint"] = (
                        np.sqrt(np.mean(np.diff(position, n=2, axis=0) ** 2, axis=0))
                        * 900
                    ).tolist()
                    np.savez_compressed(
                        checkpoint_dir / (case["id"] + ".npz"),
                        targets=target,
                        positions=position,
                        velocities=env.velocities,
                    )
                    record_path = checkpoint_dir / (case["id"] + ".json")
                    temporary = record_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
                    os.replace(temporary, record_path)
                    records.append(record)
                    print(
                        "CASE "
                        + json.dumps(
                            {
                                k: record[k]
                                for k in (
                                    "checkpoint",
                                    "scenario_id",
                                    "task_success",
                                    "collision_free_success",
                                    "sim_seconds",
                                    "contact_control_ticks",
                                )
                            }
                        ),
                        flush=True,
                    )
            finally:
                vector.close()
            summary = summarize(args.suite, checkpoint, records)
            summary["elapsed_seconds"] = time.monotonic() - start
            (checkpoint_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries.append(summarize(args.suite, checkpoint, records))
        print("CHECKPOINT " + json.dumps(all_summaries[-1]), flush=True)
        del policy
        torch.cuda.empty_cache()
        (run_output / "checkpoint_summary.json").write_text(
            json.dumps(all_summaries, indent=2)
        )
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == suite_hash
    print("COMPLETE paired benchmark", flush=True)


if __name__ == "__main__":
    main()
