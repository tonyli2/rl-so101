# Diagnosing ACT motion

`lerobot-tabletop/diagnose_motion.py` loads saved policy weights and processors,
then instruments LeRobot's native rollout loop. It records applied targets,
actual positions/velocities and actual policy query indices. It does not alter
checkpoints, the baseline evaluator or demonstrations. Demonstration references
come from their saved commands and exact MuJoCo states.

Run from `lerobot-tabletop/` in the lerobot Conda environment:

```bash
python diagnose_motion.py \
  --run ../outputs/train/act-sim-baseline-100k \
  --dataset recordings/sim-baseline \
  --output ../outputs/motion-diagnostics-new
python -m learning.plot_motion ../outputs/motion-diagnostics-new
python -m unittest discover -s tests -p test_motion_diagnostics.py -v
```

Use a fresh output directory. Plotting requires `python -m pip install matplotlib`.
Default experiments are the 10k checkpoint from start 45, the final checkpoint
from starts 45–49, and shorter execution / temporal ensembling from start 45.
A full-precision control is available via `--checkpoints 100000 --episodes 45
--variants fp32`. All learned variants use the same weights; no retraining occurs.
The recorded settings in `execution.json` allow numerical reruns. The original
in-training videos were not instrumented, so their actions cannot be recovered
exactly from the MP4s. Backend numerical differences can change closed-loop
trajectory length; the diagnostic baseline took 389 steps versus 406 actions
in the original final video. These results identify reproducible mechanisms,
not a bitwise reconstruction of that original rollout.

## Measurements

A chunk handoff is the difference `action[100]-action[99]`, then steps 200,300,
etc. The initial command-versus-reset discrepancy is not a handoff. The native
same-step vector autoreset is excluded from saved position/velocity traces.
Higher-order derivative stencils touching a handoff are classified as boundary
stencils. Physical response is also measured in +/-3-control-tick windows.

Rapid within-chunk reversal counts require both neighboring increments to
exceed 0.1 degrees (gripper: percentage points) with opposite signs. Centers in
the first four ticks or within +/-3 ticks of a handoff are excluded. Angular
joints and gripper units are never pooled. These counts help distinguish
back-and-forth motion from acceleration along a smooth one-way trajectory.

## Findings from the 100k run

Both mechanisms are present:

- **Handoffs:** for start 45, wrist-roll target jumps reached 3.17 degrees at
  handoffs, versus a maximum 0.44 degrees within chunks. Three handoffs
  accounted for 73% of squared wrist-roll target increments. Other initial
  states showed different proportions, so this is not universal dominance.
- **Within chunks:** wrist-flex targets reversed sharply 24 times, and actual
  wrist-flex position reversed 26 times, versus zero for both in demonstration
  45. This excludes handoff windows and confirms genuine internal oscillation.
- **Ensembling:** setting `n_action_steps=1` and
  `temporal_ensemble_coeff=0.01` reduced those counts to 1 and 4, respectively,
  and completed the task from that start. Elbow target acceleration RMS fell
  from 478 to 158 deg/s². These are single-start comparisons, not guarantees
  across all starts or physical hardware.
- **Precision control:** FP32 still produced 17 wrist-flex target reversals
  and 21 actual-position reversals. Elbow target acceleration remained
  458 deg/s². Thus FP16 is not the primary cause, although it affects results.
- **Short execution:** executing only 10 actions per prediction failed the
  task from start 45. Its low aggregate movement statistics partly reflect
  getting stuck; they must not be interpreted as a successful smooth policy.
- **Training duration:** the 10k policy had substantially more within-chunk
  reversals than 100k, so training already improved smoothness.

Baseline final-policy rollouts succeeded from all five tested starts. The
ensemble and FP32 controls were only tested from start 45. Target roughness
and actual motion both show oscillations, so this is not solely a video artifact
or a controller oscillating under perfectly smooth commands. Dynamics and
contacts can still amplify the input. Exact attribution of perceived jerk to
individual causes requires further controlled experiments.

Raw per-joint traces, numeric summaries, a detailed report and figures are in
`outputs/motion-diagnostics/`; the separate FP32 execution metadata is in
`outputs/motion-diagnostics-fp32/`. Ten synthetic metric tests cover boundary
indexing, derivative stencils, initial-transient exclusion, smooth-motion
controls, oscillating measured states and reversal thresholds/windows.

## Remaining large unfolding jerk with TE

A subsequent instrumented comparison of 80k, 90k and 100k found a separate
contact-induced effect. The fixed gripper collision box contacts the shoulder
while unfolding in 80k/100k; the motor reaches its torque limit and falls behind
the smoothly moving target. After contact clears, it catches up abruptly.
90k avoids this contact. Replaying identical commands while suppressing only
that body-pair contact reduces the largest early elbow change from 7.06 to
1.77 degrees/tick (80k), and 6.44 to 1.34 (100k); 90k is unchanged at 2.13.
Normal replays reproduce logged positions to within about 0.000004 degrees.

See `outputs/te-early-jerk/report.md` and its comparison figure for the full
analysis. `learning/contact_replay.py` reproduces this controlled intervention.
No production contacts were disabled. This establishes a collision/dynamics
cause for the singular unfolding jerk, distinct from the within-chunk command
jitter and handoff jumps analyzed above.
