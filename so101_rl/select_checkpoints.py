"""Pick each run's checkpoint from validation-suite results (outputs/suite_eval/PROTOCOL.md, step 1).

Rule: highest task success summed over the delay conditions (all cases); ties go to higher
collision-free success, then to the later step. Writes {run: step} for eval_suite.py --checkpoints.

    python select_checkpoints.py --results outputs/suite_eval/physA_validation/results.jsonl \
        --out outputs/suite_eval/physA_selected.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", help="only these runs (default: all in the results)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    delays = sorted({r["delay"] for r in rows})
    cases = sorted({r["case"] for r in rows})
    score: dict = defaultdict(lambda: [0, 0, 0])  # (run, step) -> success, collision-free, episodes
    for r in rows:
        s = score[(r["run"], r["step"])]
        s[0] += r["task_success"]
        s[1] += r["collision_free_success"]
        s[2] += 1
    expected = len(delays) * len(cases)
    selected, report = {}, []
    for run in sorted({run for run, _ in score}):
        if args.runs and run not in args.runs:
            continue
        steps = [(step, s) for (r, step), s in score.items() if r == run]
        incomplete = [step for step, s in steps if s[2] != expected]
        if incomplete:
            raise SystemExit(f"{run}: incomplete validation for steps {incomplete} (need {expected} episodes each)")
        best = max(steps, key=lambda x: (x[1][0], x[1][1], int(x[0])))
        selected[run] = int(best[0])
        report.append((run, sorted(steps, key=lambda x: int(x[0])), best[0]))
    args.out.write_text(json.dumps(selected, indent=2) + "\n")
    for run, steps, best in report:
        print(f"{run} (task / collision-free successes of {expected}; delays {', '.join(delays)}):")
        for step, (s, c, n) in steps:
            print(f"  {int(step):>8}: {s:3d} / {c:3d}{'   <- selected' if step == best else ''}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
