# Collision-aware collection and paired checkpoint evaluation

## Plan and implementation status

1. Reject nonadjacent robot self-contact before recording starts and after each
   physics substep. Automatically discard affected active attempts; retain clear
   feedback and allow retry. Expected gripper/object and object/table contacts
   remain permitted.
2. Audit every archived transition at all seven physics substeps with exact
   recorded controls. Quarantine every affected episode in a valid native
   dataset, preserve an untouched original backup, and verify copied videos,
   numeric data and archive mappings.
3. Freeze a nearby collision-free initial posture, scenario states and compiled
   model. Build separate robot, cube, bin and combined cases with conservative
   endpoint-reachability checks. Review the design independently.
4. Evaluate every 10k..100k checkpoint using TE=0.01, identical case ordering,
   seeds, precision, batching, camera geometry and 900-step deadlines. Save
   resumable per-case results and traces. Continue after contact so both task
   success and collision-free success are available.
5. Compare checkpoint curves by factor and offset. Use the result to propose
   an initial clean-data retraining budget, not to claim that old contaminated
   checkpoints determine clean-data convergence.

The old-data sweep was stopped at the user's request after complete10k–50k
checkpoints and a partial60k checkpoint. Completed results are preserved; missing
cases are not failures. The collection gate, data quarantine and reusable
evaluation pipeline are complete. The clean-data baseline is configured separately; this benchmark does not start training.

## Frozen development suite

Bundle: `outputs/eval/act-robustness-v1/`.

| Family | Cases | Changes |
| --- | ---: | --- |
| Historical control | 1 | Exact original episode45 reset; known initial contact; excluded from primary score |
| Anchor control | 1 | Original pose plus shoulder lift +2 degrees and elbow -2 degrees |
| Robot pose | 4 | Two +/-2-degree pan offsets and two coordinated small joint offsets |
| Cube | 16 | World X/Y shifts of +/-1, +/-2, +/-4 and +/-6 cm |
| Bin | 6 | 2/4 cm away; 2 cm away plus +/-2/4 cm sideways |
| Combined | 4 | Robot offsets, diagonal 2 cm cube shifts, bin away/sideways 2 cm |

There are 31 primary cases plus the historical control. The bin's radial and
sideways axes are defined from the original cube-to-bin vector once, rather
than recalculated after moving the cube. Bin offsets are deliberately capped:
the largest combined displacement is approximately 4.47 cm, and every modified
bin has a joint-limited, self-contact-free release endpoint witness within 5 mm
of its requested position and 15 degrees of its reference tool axis. All shifted
cube approach endpoints also have witnesses. These establish feasible endpoints,
not collision-free paths or successful grasps. Cube+Y6cm leaves little initial
bin clearance and is labeled as a stress case by magnitude.

All primary starts are within joint limits and initially free of self-contact,
robot/object overlap, cube/bin overlap and table-edge violations. Initial
velocities and warm starts are standardized; archived historical control keeps
its original state. `manifest.json`, `states.npz` and `model.mjb` are independent
of subsequent dataset reindexing and are hashed. Do not edit them once results
exist; create a new version for new cases.

## Run or resume the benchmark

```bash
conda activate lerobot
cd ~/code/enph576/lerobot-tabletop
python evaluate_suite.py \
  --run ../outputs/train/act-sim-baseline-100k \
  --suite ../outputs/eval/act-robustness-v1 --batch-size 3
```

The runner reuses LeRobot's native rollout, saved ACT processors and checkpoint
weights. It executes the default ten checkpoints; `--checkpoints 010000 020000`
selects a subset. The exact batch size/order is fixed because GPU numerical
kernels can differ with batch shape. Native `FreezeAfterEpisodeEnd` prevents
finished cases from resetting and contaminating contact/motion statistics while
other cases finish. Complete groups resume only when suite/checkpoint hashes
and inference conditions match. No training, Hub uploads or hardware control
occur.

Outputs:

- `results/<step>/<scenario>.json`: task success, collision-free success,
  completion/deadline, contacts, tracking error, per-joint motion diagnostics and
  checkpoint/scenario identity.
- `results/<step>/<scenario>.npz`: targets and actual joint positions/velocities.
- `checkpoint_summary.json`: overall and per-factor counts.
- `execution.json`: precision, TE, seeds, batch size and library version.

The baseline evaluation videos remain unchanged. This expanded suite is for
checkpoint selection and debugging. A deterministic fraction of these selected
cases is not an independent real-world success probability. Keep a separate,
previously unused test suite for final claims after tuning against this one.

## Data and retraining interpretation

The full audit found 18 affected episodes, including the original 11 shoulder/gripper
cases and 7 additional body-pair cases. The migration initially retained 32 episodes / 17,281 frames; 18 clean
replacements have since restored the active baseline to 50 episodes. Quarantine
contains the 18 original contact episodes / 8,725 frames. See
[dataset quarantine](dataset-quarantine.md) for mappings and preserved backups.
The audit replayed 26,006 transitions and reproduced every next qpos exactly.

For a model trained only on accepted data, retraining from the pretrained visual
backbone is the clearest comparison. Continuing old weights may improve them,
but does not make them a model trained only on the filtered data. Removing bad
examples alone does not guarantee collision avoidance: ACT can still predict a
new colliding path, and a single-position dataset may not teach object-location
generalization. New collision-checked, varied demonstrations may be more useful
than extra updates if the spatial-offset cases fail across all checkpoints.

Start the clean comparison with periodic checkpoints/evaluation and extend only
while held-out outcomes improve. Since fewer frames remain, the same number of
updates revisits the retained demonstrations more often. Old-checkpoint results
can suggest which initial budget to try but cannot prove how many steps a new
clean-data policy will need. No retraining is started by this benchmark.

## Generate a new version

Use the untouched historical backup as the source for this version's original
episode45 reference (the retained dataset was reindexed and excludes it):

```bash
python -m learning.robustness \
  --dataset ../outputs/dataset-backups/sim-baseline-original-50 \
  --output ../outputs/eval/act-robustness-v2
```

Never regenerate an existing bundle. The reference archive and compiled model
are copied into the bundle so evaluation does not depend on the active dataset.
A second independent generation of version1 reproduced every frozen state and
bin position exactly. After changing detector, evaluator, or success-check code,
use a fresh result directory/version rather than mixing old and new outcomes.

Raw success uses the existing released-and-settled cube-in-bin criterion; it
does not independently certify a grasp/lift strategy. In particular, near-bin
cube cases should be visually inspected for alternative strategies. The
2 cm bin shifts can also fit within a fixed old release position's geometric
tolerance; the 4 cm cases are more discriminating. See the
[independent review](evaluation-design-review.md).

Summarize completed or partial runs with:

```bash
python -m learning.summarize_suite ../outputs/eval/act-robustness-v1
```

The resulting report, CSV tables and heat maps distinguish missing evaluations
from failures. Only complete checkpoints are ranked. The historical control is
shown separately. Near-best step counts are descriptive model-selection hints,
not evidence that a newly trained clean-data model has converged.

To compare a newly trained policy without mixing its checkpoints with the old
run, reuse the same suite and select a separate result directory:

```bash
python evaluate_suite.py \
  --run ../outputs/train/act-clean-50demos-30k \
  --suite ../outputs/eval/act-robustness-v1 \
  --output ../outputs/eval/act-clean-50demos-30k-robustness \
  --checkpoints 010000 020000 030000 \
  --batch-size 3
python -m learning.summarize_suite \
  ../outputs/eval/act-clean-50demos-30k-robustness \
  --checkpoints 010000 020000 030000
```

The output directory receives an identical manifest copy for reporting; the
simulator still uses the original frozen model/states. Weight fingerprints
prevent accidentally reusing old policy results at the same step number.
