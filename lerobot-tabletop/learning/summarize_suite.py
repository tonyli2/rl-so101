"""Summarize a frozen, paired ACT scenario suite without treating missing cases as failures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CHECKPOINTS = [f"{step:06d}" for step in range(10000, 100001, 10000)]
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *[
                "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |"
                for row in rows
            ],
        ]
    )


def status(record):
    if record is None:
        return "—"
    if record["collision_free_success"]:
        return "S"
    return "T" if record["task_success"] else "F"


def load_results(bundle, checkpoints):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    records = {}
    cases = {case["id"] for case in manifest["cases"]}
    for checkpoint in checkpoints:
        records[checkpoint] = {}
        for case_id in cases:
            path = bundle / "results" / checkpoint / f"{case_id}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["suite_sha256"] != expected_hash:
                raise ValueError(f"Result belongs to a different suite: {path}")
            if record["scenario_id"] != case_id or record["checkpoint"] != checkpoint:
                raise ValueError(f"Result identity differs from its path: {path}")
            records[checkpoint][case_id] = record
    return manifest, records


def checkpoint_rows(cases, records):
    primary = [c for c in cases if c["primary"]]
    rows = []
    for checkpoint, completed in records.items():
        selected = [completed[c["id"]] for c in primary if c["id"] in completed]
        row = {
            "checkpoint": checkpoint,
            "expected_primary": len(primary),
            "completed_primary": len(selected),
            "missing_primary": len(primary) - len(selected),
            "task_successes": sum(r["task_success"] for r in selected),
            "collision_free_successes": sum(
                r["collision_free_success"] for r in selected
            ),
            "cases_with_self_contact": sum(
                r["first_contact"] is not None for r in selected
            ),
        }
        rows.append(row)
    return rows


def paired(cases, records, left, right):
    common = [
        c["id"]
        for c in cases
        if c["primary"] and c["id"] in records[left] and c["id"] in records[right]
    ]
    result = {"left": left, "right": right, "common_primary": len(common)}
    for key in ("task_success", "collision_free_success"):
        result[f"{key}_gained"] = [
            c for c in common if not records[left][c][key] and records[right][c][key]
        ]
        result[f"{key}_lost"] = [
            c for c in common if records[left][c][key] and not records[right][c][key]
        ]
    return result


def write_csv(bundle, cases, records, rows):
    fields = [
        "checkpoint",
        "scenario_id",
        "family",
        "primary",
        "completed",
        "task_success",
        "collision_free_success",
        "self_contact",
        "steps",
        "sim_seconds",
        "contact_control_ticks",
        "first_contact_step",
        "first_contact_bodies",
        "failure",
        "robot_delta_deg",
        "cube_x_cm",
        "cube_y_cm",
        "bin_away_cm",
        "bin_side_cm",
    ]
    fields += [
        f"{metric}_{joint}"
        for metric in (
            "tracking_error_rms",
            "tracking_error_max",
            "target_acceleration_rms",
            "state_motion_acceleration_rms",
        )
        for joint in JOINTS
    ]
    with (bundle / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for checkpoint, results in records.items():
            for case in cases:
                r = results.get(case["id"])
                row = {
                    "checkpoint": checkpoint,
                    "scenario_id": case["id"],
                    "family": case["family"],
                    "primary": case["primary"],
                    "completed": r is not None,
                    "robot_delta_deg": json.dumps(case["robot_delta_deg"]),
                    "cube_x_cm": 100 * case["cube_delta_m"][0],
                    "cube_y_cm": 100 * case["cube_delta_m"][1],
                    "bin_away_cm": 100 * case["bin_away_m"],
                    "bin_side_cm": 100 * case["bin_side_m"],
                }
                if r is not None:
                    row.update(
                        {
                            key: r.get(key, "")
                            for key in (
                                "task_success",
                                "collision_free_success",
                                "steps",
                                "sim_seconds",
                                "contact_control_ticks",
                                "failure",
                            )
                        }
                    )
                    contact = r["first_contact"]
                    row["self_contact"] = contact is not None
                    row["first_contact_step"] = (
                        contact.get("step", "") if contact else ""
                    )
                    row["first_contact_bodies"] = (
                        json.dumps(contact["bodies"]) if contact else ""
                    )
                    for metric in (
                        "tracking_error_rms",
                        "tracking_error_max",
                        "target_acceleration_rms",
                        "state_motion_acceleration_rms",
                    ):
                        for joint, value in zip(
                            JOINTS, r.get(metric + "_per_joint", [])
                        ):
                            row[f"{metric}_{joint}"] = value
                writer.writerow(row)
    with (bundle / "checkpoint_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_plot(bundle, cases, records, key, filename, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    selected = [c for c in cases if c["primary"]]
    checkpoints = list(records)
    data = np.full((len(selected), len(checkpoints)), np.nan)
    for y, case in enumerate(selected):
        for x, checkpoint in enumerate(checkpoints):
            record = records[checkpoint].get(case["id"])
            if record is not None:
                data[y, x] = float(record[key])
    colors = ListedColormap(["#ce6262", "#4da97c"])
    colors.set_bad("#cbd0d6")
    fig, ax = plt.subplots(
        figsize=(max(10, len(checkpoints) * 0.9), max(8, len(selected) * 0.32)),
        constrained_layout=True,
    )
    ax.imshow(
        np.ma.masked_invalid(data),
        cmap=colors,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for y in range(len(selected)):
        for x in range(len(checkpoints)):
            ax.text(
                x,
                y,
                "—" if np.isnan(data[y, x]) else str(int(data[y, x])),
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_xticks(
        np.arange(len(checkpoints)), [f"{int(c) // 1000}k" for c in checkpoints]
    )
    ax.set_yticks(np.arange(len(selected)), [c["id"] for c in selected], fontsize=8)
    for y in range(1, len(selected)):
        if selected[y]["family"] != selected[y - 1]["family"]:
            ax.axhline(y - 0.5, color="white", linewidth=2)
    ax.set(
        xlabel="Training checkpoint",
        title=title
        + "\nFixed primary cases; historical colliding-start control excluded",
    )
    ax.legend(
        handles=[
            Patch(color="#4da97c", label="Pass"),
            Patch(color="#ce6262", label="Fail"),
            Patch(color="#cbd0d6", label="Missing"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
    )
    fig.savefig(bundle / filename, dpi=160)
    plt.close(fig)


def axis_tables(cases, records):
    chunks = [
        "## Sampled perturbation boundaries",
        "",
        "Cells: **S** = task completed without self-contact; **T** = task completed with self-contact; **F** = task not completed; **—** = missing. These are tested points, not interpolated thresholds. Non-monotonic results must not be converted into a single guaranteed operating radius.",
        "",
    ]
    for axis in (0, 1):
        for sign, sign_name in ((-1, "minus"), (1, "plus")):
            selected = sorted(
                [
                    c
                    for c in cases
                    if c["family"] == "cube"
                    and c["cube_delta_m"][axis] * sign > 0
                    and c["cube_delta_m"][1 - axis] == 0
                ],
                key=lambda c: abs(c["cube_delta_m"][axis]),
            )
            rows = []
            for checkpoint, completed in records.items():
                labels = [status(completed.get(c["id"])) for c in selected]
                passed = [
                    abs(c["cube_delta_m"][axis]) * 100
                    for c, value in zip(selected, labels)
                    if value == "S"
                ]
                failed = [
                    abs(c["cube_delta_m"][axis]) * 100
                    for c, value in zip(selected, labels)
                    if value in ("T", "F")
                ]
                rows.append(
                    [
                        checkpoint,
                        *labels,
                        f"{max(passed):g}" if passed else "none observed",
                        f"{min(failed):g}" if failed else "none observed",
                    ]
                )
            chunks += [
                f"### Cube {'xy'[axis]} {sign_name}",
                "",
                table(
                    [
                        "Checkpoint",
                        *[
                            f"{abs(c['cube_delta_m'][axis]) * 100:g} cm"
                            for c in selected
                        ],
                        "Largest tested S (cm)",
                        "Smallest tested non-S (cm)",
                    ],
                    rows,
                ),
                "",
            ]
    bins = [c for c in cases if c["family"] == "bin"]
    chunks += [
        "### Bin away/sideways translation",
        "",
        "Sideways uses the frozen perpendicular-to-separation axis, not the world's arbitrary x/y axis. Each column isolates its specified combination; a failed shifted-bin task can reflect perception or placement adaptation rather than geometric unreachability.",
        "",
        table(
            [
                "Checkpoint",
                *[
                    f"away {100 * c['bin_away_m']:g}, side {100 * c['bin_side_m']:g} cm"
                    for c in bins
                ],
            ],
            [
                [checkpoint, *[status(completed.get(c["id"])) for c in bins]]
                for checkpoint, completed in records.items()
            ],
        ),
        "",
    ]
    for family in ("robot", "combined"):
        selected = [c for c in cases if c["family"] == family]
        chunks += [
            f"### {family.title()} cases",
            "",
            table(
                ["Checkpoint", *[c["id"] for c in selected]],
                [
                    [checkpoint, *[status(completed.get(c["id"])) for c in selected]]
                    for checkpoint, completed in records.items()
                ],
            ),
            "",
        ]
    return chunks


def write_report(bundle, manifest, records, rows):
    cases = manifest["cases"]
    primary = [c for c in cases if c["primary"]]
    complete = [r for r in rows if r["missing_primary"] == 0]
    best = (
        sorted(
            complete,
            key=lambda r: (
                -r["collision_free_successes"],
                -r["task_successes"],
                int(r["checkpoint"]),
            ),
        )[0]
        if complete
        else None
    )
    plateau = None
    if best and best["collision_free_successes"] > 0:
        eligible = [
            r
            for r in complete
            if r["collision_free_successes"] >= best["collision_free_successes"] - 1
            and r["task_successes"] >= best["task_successes"] - 1
        ]
        plateau = (
            min(eligible, key=lambda r: int(r["checkpoint"])) if eligible else None
        )
    parts = [
        "# Paired ACT robustness evaluation",
        "",
        f"This frozen suite contains **{len(primary)} primary scenarios** plus a separately labeled historical control. Current completion: **{sum(r['completed_primary'] for r in rows)}/{len(primary) * len(rows)} primary checkpoint/case pairs**. Missing results are not failures.",
        "",
        "Every checkpoint uses the same saved scenario states, original model, 30 Hz control, 900-step deadline, ACT temporal ensembling coefficient 0.01 and one action executed per prediction. The execution JSON records precision, seed, batching and hashes. These hand-designed cases are a checkpoint-selection development suite, not independent random trials or a population success-probability estimate.",
        "",
        "## Primary checkpoint comparison",
        "",
        "Task success means released, settled placement in the bin. Collision-free success additionally requires no detected robot self-contact throughout the rollout. Historical colliding-start cases are excluded from both primary denominators.",
        "",
        table(
            [
                "Checkpoint",
                "Completed/expected",
                "Task success",
                "Collision-free task success",
                "Cases with self-contact",
                "Missing",
            ],
            [
                [
                    r["checkpoint"],
                    f"{r['completed_primary']}/{r['expected_primary']}",
                    f"{r['task_successes']}/{r['completed_primary']}"
                    if r["completed_primary"]
                    else "—",
                    f"{r['collision_free_successes']}/{r['completed_primary']}"
                    if r["completed_primary"]
                    else "—",
                    r["cases_with_self_contact"],
                    r["missing_primary"],
                ]
                for r in rows
            ],
        ),
        "",
    ]
    if best:
        parts += [
            f"Among checkpoints with all primary cases complete, the descriptive best is **{best['checkpoint']}**: {best['collision_free_successes']}/{len(primary)} collision-free successes and {best['task_successes']}/{len(primary)} task successes. Ranking prioritizes collision-free count, then task count, then the earlier checkpoint for exact ties.",
            "",
        ]
        if plateau:
            parts += [
                f"The earliest fully evaluated checkpoint within **one collision-free case and one task-success case** of that best is **{plateau['checkpoint']}**. This is an observed near-best budget on this fixed grid, not proof of convergence or equivalent results after data cleaning.",
                "",
            ]
        else:
            parts += [
                "No checkpoint demonstrates a positive collision-free plateau on this suite; a task-only plateau must not be described as safe policy convergence.",
                "",
            ]
        if len(complete) < len(rows):
            parts += [
                f"**Interim result:** only {len(complete)}/{len(rows)} checkpoints are complete; the best/near-best selection can change as later results arrive.",
                "",
            ]
    else:
        parts += [
            "No checkpoint has all primary cases complete yet. Best-checkpoint/plateau selection is withheld until at least one complete checkpoint exists.",
            "",
        ]
    parts += ["## Family-specific performance", ""]
    family_rows = []
    for checkpoint, completed in records.items():
        for family in sorted({c["family"] for c in primary}):
            selected = [c for c in primary if c["family"] == family]
            available = [completed[c["id"]] for c in selected if c["id"] in completed]
            family_rows.append(
                [
                    checkpoint,
                    family,
                    f"{len(available)}/{len(selected)}",
                    sum(r["task_success"] for r in available) if available else "—",
                    sum(r["collision_free_success"] for r in available)
                    if available
                    else "—",
                ]
            )
    parts += [
        table(
            [
                "Checkpoint",
                "Family",
                "Completed/expected",
                "Task successes",
                "Collision-free successes",
            ],
            family_rows,
        ),
        "",
    ]
    parts += axis_tables(cases, records)
    comparisons = [
        paired(cases, records, a, b)
        for a, b in zip(list(records)[:-1], list(records)[1:])
    ]
    parts += [
        "## Paired adjacent-checkpoint changes",
        "",
        "Gains/losses compare the same completed cases. They are descriptive case transitions, not significance tests or evidence from independent random samples.",
        "",
        table(
            [
                "Earlier → later",
                "Common primary cases",
                "Task gained/lost",
                "Collision-free gained/lost",
            ],
            [
                [
                    f"{p['left']} → {p['right']}",
                    p["common_primary"],
                    f"{len(p['task_success_gained'])}/{len(p['task_success_lost'])}",
                    f"{len(p['collision_free_success_gained'])}/{len(p['collision_free_success_lost'])}",
                ]
                for p in comparisons
            ],
        ),
        "",
    ]
    legacy = [c for c in cases if not c["primary"]]
    parts += [
        "## Historical control (excluded from primary scores)",
        "",
        table(
            ["Checkpoint", *[c["id"] for c in legacy]],
            [
                [checkpoint, *[status(completed.get(c["id"])) for c in legacy]]
                for checkpoint, completed in records.items()
            ],
        ),
        "",
    ]
    parts += [
        "## Interpretation and clean-data retraining",
        "",
        "The existing checkpoints were trained on 50 original demonstrations, including the 18 now quarantined for self-contact. The retained active dataset has 32 demonstrations / 17,281 frames. These old-checkpoint comparisons can show whether more optimization improved their robustness, but cannot establish the step budget for retraining on the filtered dataset.",
        "",
        "Use a fresh clean-data training run, save checkpoints every 5k steps, and apply this unchanged development suite at the saved checkpoints. Begin with a shorter monitored budget rather than assuming 100k is necessary; extend only when the clean learning curve still improves the desired collision-free outcomes. A fresh held-out scenario set is required for an unbiased final generalization assessment after using this suite to choose settings.",
        "",
        "The original demonstrations kept cube/bin positions fixed. Failing displaced-object cases can reflect missing task variation in the training data; simply training longer on identical fixed poses may not solve that limitation. Perturbations also remain bounded by the frozen reachability preflight: its IK witnesses establish valid endpoints, not a guaranteed collision-free grasp-and-place trajectory.",
        "",
        "## Artifacts",
        "",
        "- `results.csv`: all expected checkpoint/case rows; missing outcome fields stay blank.",
        "- `checkpoint_table.csv`: primary counts and missing counts.",
        "- `task_success_matrix.png`, `collision_free_success_matrix.png`: primary case matrices.",
        "- `analysis.json`: ranking, paired gained/lost case IDs, and counts.",
        "- `results/<checkpoint>/<case>.json` and `.npz`: original measured outcomes and joint traces.",
        "",
    ]
    (bundle / "report.md").write_text("\n".join(parts))
    analysis = {
        "primary_cases": len(primary),
        "checkpoint_rows": rows,
        "best_complete_checkpoint": best,
        "earliest_near_best_complete_checkpoint": plateau,
        "paired_adjacent": comparisons,
        "all_primary_complete": len(complete) == len(rows),
    }
    (bundle / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False)
    )
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    args = parser.parse_args()
    manifest, records = load_results(args.bundle, args.checkpoints)
    rows = checkpoint_rows(manifest["cases"], records)
    write_csv(args.bundle, manifest["cases"], records, rows)
    analysis = write_report(args.bundle, manifest, records, rows)
    matrix_plot(
        args.bundle,
        manifest["cases"],
        records,
        "task_success",
        "task_success_matrix.png",
        "Task success",
    )
    matrix_plot(
        args.bundle,
        manifest["cases"],
        records,
        "collision_free_success",
        "collision_free_success_matrix.png",
        "Collision-free task success",
    )
    print(
        json.dumps(
            {
                "completed_primary_pairs": sum(r["completed_primary"] for r in rows),
                "expected_primary_pairs": analysis["primary_cases"] * len(rows),
                "best_complete_checkpoint": analysis["best_complete_checkpoint"],
                "earliest_near_best_complete_checkpoint": analysis[
                    "earliest_near_best_complete_checkpoint"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
