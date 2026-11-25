"""Run the main experiment (proposal sec. 3): nominal vs randomized PPO from one BC checkpoint.

Six train_ppo.py runs back to back, interleaved (nominal s1, randomized s1, nominal s2, ...) so a
complete pair exists if the night is cut short. The plan lives in outputs/ppo/main_plan.json and is
re-read before every run, so the remaining runs can be changed while it works (e.g. "steps", or
"skip": true on a run). Finished runs (model_final.zip present) are skipped, so re-running resumes.

Encoder setting ("encoder": "auto"): wait for the --train-encoder pilot, then train the encoder heads
only if that pilot is at least as good as the frozen-encoder pilot at the same steps (mean of the
40-episode evaluations from 200k on, within 5 points) and never collapses (no evaluation below 60%
after 200k). The decision and its numbers are written into the plan.

Every run saves an evaluated checkpoint every "eval_every" steps; main_summary.json collects each
run's success-vs-steps curve.

    python run_main_experiment.py
    python run_main_experiment.py --plan outputs/ppo/scene2_plan.json   # another experiment

Optional plan keys: "randomize_ranges" (train_ppo.py --randomize-ranges file) and "extra_args"
(list of further train_ppo.py arguments, e.g. ["--quick-eval"]).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "ppo"
PLAN = OUT / "main_plan.json"
SUMMARY = OUT / "main_summary.json"

DEFAULT_PLAN = {
    "bc": "outputs/bc/resnet18_h1_dart_rel_e20_seed0_last/bc_model.zip",
    "steps": 500_000,
    "warmup_steps": 50_000,
    "eval_every": 100_000,
    "encoder": "auto",
    "encoder_pilot": "pilot_nominal_trainenc_seed0",
    "frozen_pilot": "pilot_nominal_seed0",
    "runs": [{"name": f"main_{cond}_s{seed}", "seed": seed, "randomize": cond == "randomized"}
             for seed in (1, 2, 3) for cond in ("nominal", "randomized")],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def load_plan() -> dict:
    if not PLAN.is_file():
        PLAN.write_text(json.dumps(DEFAULT_PLAN, indent=2))
    return json.loads(PLAN.read_text())


def save_plan(plan: dict) -> None:
    PLAN.write_text(json.dumps(plan, indent=2))


def history(run: str) -> list:
    path = OUT / run / "eval_history.json"
    return json.loads(path.read_text()) if path.is_file() else []


def decide_encoder(plan: dict) -> str:
    pilot, frozen = plan["encoder_pilot"], plan["frozen_pilot"]
    log(f"waiting for the encoder pilot {pilot} to finish")
    while not (OUT / pilot / "model_final.zip").is_file():
        time.sleep(60)
    a = {r["step"]: r["success_rate"] for r in history(pilot)}
    b = {r["step"]: r["success_rate"] for r in history(frozen)}
    steps = sorted(s for s in a if s >= 200_000 and s in b)
    mean_a = sum(a[s] for s in steps) / len(steps)
    mean_b = sum(b[s] for s in steps) / len(steps)
    lowest = min(a[s] for s in a if s >= 200_000)
    choice = "train" if mean_a >= mean_b - 0.05 and lowest >= 0.60 else "frozen"
    plan["encoder"] = choice
    plan["encoder_decision"] = {"steps_compared": steps, "train_encoder_mean": mean_a, "frozen_mean": mean_b,
                                "train_encoder_lowest_after_200k": lowest, "rule": "mean >= frozen - 0.05 and lowest >= 0.60"}
    save_plan(plan)
    log(f"encoder: {choice}  (trainable mean {mean_a:.1%}, frozen mean {mean_b:.1%}, lowest {lowest:.1%})")
    return choice


def main() -> None:
    global PLAN, SUMMARY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=PLAN, help="plan json (created with defaults if missing)")
    args = parser.parse_args()
    PLAN = args.plan
    SUMMARY = PLAN.with_name(PLAN.stem.replace("_plan", "") + "_summary.json") if PLAN != OUT / "main_plan.json" else SUMMARY
    OUT.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        plan = load_plan()  # re-read: lets the remaining runs be changed overnight
        if i >= len(plan["runs"]):
            break
        run = plan["runs"][i]
        i += 1
        name = run["name"]
        if run.get("skip") or (OUT / name / "model_final.zip").is_file():
            log(f"skip {name}")
            continue
        encoder = plan["encoder"] if plan["encoder"] != "auto" else decide_encoder(plan)
        cmd = [sys.executable, "-u", "train_ppo.py", "--bc", plan["bc"], "--run-name", name,
               "--seed", str(run["seed"]), "--steps", str(plan["steps"]),
               "--warmup-steps", str(plan["warmup_steps"]), "--eval-every", str(plan["eval_every"])]
        if encoder == "train":
            cmd.append("--train-encoder")
        if run["randomize"]:
            cmd.append("--randomize")
            if plan.get("randomize_ranges"):
                cmd += ["--randomize-ranges", plan["randomize_ranges"]]
        cmd += [str(a) for a in plan.get("extra_args", [])]
        log(f"start {name}: {' '.join(cmd[2:])}")
        start = time.time()
        with open(OUT / f"{name}.log", "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        curve = [(r["step"], round(r["success_rate"], 3)) for r in history(name)]
        log(f"done {name} (exit {code}, {(time.time() - start) / 3600:.2f} h): success by step {curve}")
        summary = json.loads(SUMMARY.read_text()) if SUMMARY.is_file() else {}
        summary[name] = {"exit_code": code, "hours": round((time.time() - start) / 3600, 2),
                         "encoder": encoder, "randomize": run["randomize"], "seed": run["seed"],
                         "steps": plan["steps"], "curve": curve}
        SUMMARY.write_text(json.dumps(summary, indent=2))
    log("all runs finished")


if __name__ == "__main__":
    main()
