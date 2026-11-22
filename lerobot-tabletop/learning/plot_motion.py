"""Plot saved motion traces and report boundary/interior measurements.

Run from lerobot-tabletop: python -m learning.plot_motion OUTPUT_DIRECTORY
Uses only saved traces; never loads a policy, simulator, or CUDA context.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from recording.schema import JOINTS


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def write_report(output, reports, baseline_label):
    baseline = reports[baseline_label]
    boundaries = baseline["boundary_steps"]
    parts = [
        "# ACT motion diagnostics",
        "",
        "Measurements below describe the saved traces; they do not automatically establish a root cause.",
        "Targets are applied/clipped joint position commands. Actual motion comes from simulator states.",
        "Angles use degrees; gripper opening uses percentage points (pp). No statistic pools these units.",
        "",
        f"Baseline: `{baseline_label}`, {baseline['steps']} executed steps, "
        f"{len(boundaries)} noninitial chunk handoffs at steps {boundaries}.",
        "",
        "At 30 Hz, a handoff at step 100 is the command difference a[100] − a[99], at 3.333 s.",
        "The initial reset-to-command jump and terminal autoreset are excluded. Higher-order derivatives",
        "are classified as boundary derivatives whenever their finite-difference stencil crosses a handoff.",
        "",
        "## Applied target increments: boundary versus interior",
        "",
        "RMS/max increments are in degrees per control tick (or pp/tick for the gripper).",
        "Squared-increment energy share indicates how much of the total squared target variation occurs",
        "at handoffs. A small number of boundary samples can have large per-event changes without",
        "accounting for most total variation. Compare both quantities.",
        "",
    ]
    rows = []
    for name in JOINTS:
        data = baseline["joints"][name]["target_delta"]
        b, i = data["boundary"], data["interior"]
        share = "—" if b["energy_share"] is None else f"{100 * b['energy_share']:.1f}%"
        rows.append(
            [
                name,
                b["count"],
                i["count"],
                number(b["rms"]),
                number(i["rms"]),
                number(b["max"]),
                number(i["max"]),
                share,
            ]
        )
    parts += [
        table(
            [
                "Joint",
                "Boundary n",
                "Interior n",
                "Boundary RMS",
                "Interior RMS",
                "Boundary max",
                "Interior max",
                "Boundary energy share",
            ],
            rows,
        ),
        "",
    ]
    parts += [
        "## Within-chunk target acceleration and measured state acceleration",
        "",
        "Target acceleration is the second finite difference of position targets divided by dt².",
        "Measured acceleration is the first difference of simulator joint velocity divided by dt.",
        "These describe different signals: one is requested motion, the other is the dynamics response.",
        "Values below are RMS deg/s² or pp/s², per joint. Boundary-crossing target stencils are excluded.",
        "For measured motion, ±3-tick handoff windows and the first three startup ticks are excluded.",
        "Demonstrations have no artificial chunk boundaries, so their interior column uses all eligible samples.",
        "Full-task comparisons can also reflect different movement phases, speeds, contacts, or durations.",
        "",
    ]
    selected = [
        label
        for label in reports
        if label == baseline_label
        or label.startswith("demo_")
        or "execute10_ep45" in label
        or "ensemble_ep45" in label
    ]
    for metric, title in (
        ("target_acceleration", "Target acceleration"),
        ("state_acceleration_boundary_window", "Measured state acceleration"),
    ):
        rows = []
        for label in selected:
            report = reports[label]
            rows.append(
                [
                    label,
                    report["steps"],
                    *[
                        number(report["joints"][name][metric]["interior"]["rms"])
                        for name in JOINTS
                    ],
                ]
            )
        parts += [f"### {title}", "", table(["Trace", "Steps", *JOINTS], rows), ""]
    parts += [
        "## Rapid target direction reversals inside chunks",
        "",
        "A reversal requires consecutive target increments of opposite sign, both larger than 0.1 degree (or gripper percentage point). Centers in the first four ticks or within +/-3 ticks of a handoff are excluded. Counts measure rapid back-and-forth commands, not ordinary smooth one-way motion.",
        "",
    ]
    parts.append(
        table(
            ["Trace", *JOINTS],
            [
                [
                    label,
                    *[
                        report["joints"][name]["interior_reversals"]["count"]
                        for name in JOINTS
                    ],
                ]
                for label, report in reports.items()
            ],
        )
    )
    parts += [
        "",
        "Actual joint-position reversals with the same threshold/exclusions:",
        "",
    ]
    parts.append(
        table(
            ["Trace", *JOINTS],
            [
                [
                    label,
                    *[
                        report["joints"][name]["state_interior_reversals"]["count"]
                        for name in JOINTS
                    ],
                ]
                for label, report in reports.items()
            ],
        )
    )
    parts += ["", "## Runs and outcomes", ""]
    parts.append(
        table(
            ["Trace", "Steps", "Handoffs", "Success", "Clipped values"],
            [
                [
                    label,
                    report["steps"],
                    len(report["boundary_steps"]),
                    report.get("success", "demonstration"),
                    report.get("clipped_values", "—"),
                ]
                for label, report in reports.items()
            ],
        )
    )
    parts += [
        "",
        "## Interpretation limits",
        "",
        "- Large boundary target increments support chunk handoff discontinuities; substantial interior variation supports within-chunk roughness. Both can coexist.",
        "- Rough targets with rough states differ from smooth targets with oscillating states; controller/contact effects need target/state comparison.",
        "- More frequent feedback and temporal ensembling change the executed trajectory, so their outcomes alone do not isolate every cause.",
        "- Only a few handoffs occur in each short episode. Reported measurements are descriptive, not independent statistical trials.",
        "- The report does not compare wall-clock compute latency: the simulator advances by fixed control periods.",
        "",
        "## Figures",
        "",
        "- `baseline_targets_states.png`: commanded targets versus actual position for the five angular joints.",
        "- `baseline_gripper.png`: gripper target/actual, in its own units.",
        "- `baseline_target_increments.png`: absolute per-tick target increments with handoff markers.",
        "- `acceleration_comparison.png`: per-joint target and measured acceleration RMS, keeping each joint on a separate axis.",
        "",
    ]
    (output / "report.md").write_text("\n".join(parts))


def make_plots(output, reports, baseline_label):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "report.md was written; PNG plotting requires matplotlib in this interpreter"
        ) from error

    with np.load(output / f"{baseline_label}.npz", allow_pickle=False) as archive:
        targets, positions = archive["targets"], archive["positions"]
    boundaries = reports[baseline_label]["boundary_steps"]
    t = np.arange(len(targets)) / 30
    measured_t = np.arange(len(positions)) / 30

    def mark(ax):
        for b in boundaries:
            ax.axvline(
                b / 30, color="tab:red", linestyle="--", alpha=0.55, linewidth=0.8
            )
        ax.grid(alpha=0.2)

    fig, axes = plt.subplots(
        5, 1, figsize=(13, 12), sharex=True, constrained_layout=True
    )
    for j, ax in enumerate(axes):
        ax.step(t, targets[:, j], where="post", label="Applied target", linewidth=1)
        ax.plot(measured_t, positions[:, j], label="Actual position", linewidth=1)
        ax.set_ylabel(f"{JOINTS[j]}\n(deg)")
        mark(ax)
    axes[0].legend(loc="best")
    axes[0].set_title(
        f"{baseline_label}: targets versus states; dashed red = chunk handoff"
    )
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(output / "baseline_targets_states.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3), constrained_layout=True)
    ax.step(t, targets[:, 5], where="post", label="Applied target")
    ax.plot(measured_t, positions[:, 5], label="Actual opening")
    ax.set(xlabel="Simulation time (s)", ylabel="Gripper opening (pp)")
    ax.legend()
    mark(ax)
    fig.savefig(output / "baseline_gripper.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        6, 1, figsize=(13, 13), sharex=True, constrained_layout=True
    )
    for j, ax in enumerate(axes):
        delta = np.abs(np.diff(targets[:, j]))
        ax.plot(t[1:], delta, linewidth=0.8)
        valid = np.asarray([b for b in boundaries if b < len(targets)], dtype=int)
        if len(valid):
            ax.scatter(t[valid], delta[valid - 1], color="tab:red", s=20, zorder=3)
        ax.set_ylabel(f"{JOINTS[j]}\n({'pp' if j == 5 else 'deg'}/tick)")
        mark(ax)
    axes[0].set_title(
        "Absolute applied-target increments; red dots identify handoff increments"
    )
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(output / "baseline_target_increments.png", dpi=150)
    plt.close(fig)

    selected = [
        label
        for label in reports
        if label == baseline_label
        or "execute10_ep45" in label
        or "ensemble_ep45" in label
        or "fp32_ep45" in label
        or label.startswith("demo_")
    ]
    short = [
        label.replace("100000_", "").replace("demonstration", "demo")
        for label in selected
    ]
    fig, axes = plt.subplots(2, 6, figsize=(20, 9), constrained_layout=True)
    for row, (metric, title) in enumerate(
        (
            ("target_acceleration", "Target interior acceleration"),
            ("state_acceleration_boundary_window", "Measured interior acceleration"),
        )
    ):
        for j, name in enumerate(JOINTS):
            ax = axes[row, j]
            values = [
                reports[label]["joints"][name][metric]["interior"]["rms"]
                for label in selected
            ]
            ax.bar(
                np.arange(len(selected)),
                [v if v is not None else np.nan for v in values],
            )
            ax.set_xticks(
                np.arange(len(selected)), short, rotation=80, ha="right", fontsize=7
            )
            ax.set_title(name)
            ax.set_ylabel(f"RMS {'pp' if j == 5 else 'deg'}/s²")
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.text(-0.5, 1.15, title, transform=ax.transAxes, fontsize=12)
    fig.savefig(output / "acceleration_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", default="100000_baseline_ep45")
    args = parser.parse_args()
    reports = json.loads((args.output / "summary.json").read_text())
    if args.baseline not in reports:
        parser.error(f"Baseline {args.baseline!r} not found in summary.json")
    write_report(args.output, reports, args.baseline)
    make_plots(args.output, reports, args.baseline)
    print(f"Wrote report.md and four figures to {args.output}")


if __name__ == "__main__":
    main()
