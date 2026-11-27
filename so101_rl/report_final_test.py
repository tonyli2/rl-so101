"""Final-test report as declared in outputs/suite_eval/PROTOCOL.md step 2.

Task success and collision-free task success, separately for the modest and stress families, per
run and pooled per condition (3 seeds x 5 cases per family and delay), with Wilson 95% intervals.

    python report_final_test.py --results outputs/suite_eval/physA_final_test/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="markdown output (default: next to the results)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    delays = sorted({r["delay"] for r in rows})
    families = ("modest", "stress")
    lines = ["# Final test (reserved, run once)", "",
             "Cells: successes/cases (rate, Wilson 95% interval).", ""]

    def table(title: str, key, groups):
        lines.extend([f"## {title}", "", "| " + " | ".join(["group", "delay"] + [f"{f} task" for f in families]
                                                      + [f"{f} collision-free" for f in families]) + " |",
                      "|" + "---|" * (2 + 2 * len(families))])
        for g in groups:
            for d in delays:
                sel = [r for r in rows if key(r) == g and r["delay"] == d]
                parts = []
                for metric in ("task_success", "collision_free_success"):
                    for f in families:
                        fs = [r for r in sel if r["family"] == f]
                        parts.append(cell(sum(r[metric] for r in fs), len(fs)))
                lines.append(f"| {g} | {d} | " + " | ".join(parts) + " |")
        lines.append("")

    table("Pooled per condition (3 seeds)", lambda r: r["condition"], ("nominal", "randomized"))
    runs = sorted({(r["condition"], r["run"], r["step"]) for r in rows})
    table("Per run (selected checkpoint)", lambda r: f"{r['run']} @ {r['step']}", [f"{run} @ {step}" for _, run, step in runs])

    # Pairs: the same physical start with a modest and a stress appearance.
    lines.extend(["## Paired cases (task success, pooled over seeds; modest -> stress)", "",
                  "| condition | delay | " + " | ".join(f"pair {p}" for p in range(1, 6)) + " |",
                  "|---|---|" + "---|" * 5])
    factor = {r["pair"]: r["stress_factor"] for r in rows}
    for cond in ("nominal", "randomized"):
        for d in delays:
            parts = []
            for p in range(1, 6):
                m = [r for r in rows if r["condition"] == cond and r["delay"] == d and r["pair"] == p]
                a = sum(r["task_success"] for r in m if r["family"] == "modest")
                b = sum(r["task_success"] for r in m if r["family"] == "stress")
                parts.append(f"{a} -> {b}")
            lines.append(f"| {cond} | {d} | " + " | ".join(parts) + " |")
    lines.append("")
    lines.append("Pairs: " + ", ".join(f"{p} = {factor.get(p)}" for p in range(1, 6)) + ".")
    contacts = defaultdict(int)
    for r in rows:
        if r.get("first_contact"):
            contacts[r["first_contact"].get("kind", "self")] += 1
    lines.extend(["", f"Prohibited contacts: {dict(contacts) or 'none'} in {len(rows)} episodes.", "",
                  "Scope: 10 correlated cases at one layout, appearance stress only. Not a workspace-wide "
                  "or real-world estimate (PROTOCOL.md)."])
    out = args.out or args.results.parent / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
