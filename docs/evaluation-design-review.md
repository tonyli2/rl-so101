# Independent review of the ACT robustness evaluation

This review concerns the fixed development suite used to compare the existing
10k–100k ACT checkpoints. It is separate from demonstration collection's
self-collision rejection. The evaluation should continue after a policy
collides so it can report both eventual task success and collision-free success.

## What the previous evaluation measured

Five recorded demonstrations were held out for validation loss. The saved
checkpoint evaluation videos and their 100% success values used **one** fixed
simulation initial state, archived episode 45. These are different measurements.
Repeated seeds do not create independent trials when the reset and inference
are deterministic. A fixed-case pass fraction is a benchmark score, not a
well-estimated probability of success in the real world.

## Reviewed scenario design

The initial narrow perturbation plan was expanded at the user's request. Use an
immutable manifest containing the following families (35 cases if four arm
perturbations are retained, or 33 with two):

- One legacy control restoring the original full integration state unchanged.
  This is initially self-colliding under the broader detector and must be
  labeled as an invalid-start historical control, excluded from primary
  collision-free score denominators.
- One standardized, contact-free anchor.
- Two or four small, validated arm-pose perturbations about that anchor.
- Sixteen cube offsets: ±1, ±2, ±4 and ±6 cm in world X/Y.
- Nine bin offsets: 2/4/6 cm farther from the cube, plus 2 cm farther combined
  with ±2/4/6 cm lateral displacement.
- Four combined cases to expose interactions between pose and object variation.

The larger offsets are stress/OOD tiers. Report their results separately from
small local perturbations rather than interpreting one pooled average as the
quality of an ordinary pick-and-place policy.

Define the bin's away/lateral axes once from the baseline cube-to-bin vector,
not independently after each cube perturbation. For the mounted baseline,
these unit vectors are approximately `[-0.07975, 0.99682]` and
`[-0.99682, -0.07975]`, respectively. Keep bin yaw and all heights unchanged.
Only robot joint pose is perturbed; moving the robot's physical base would
change the camera/robot setup and is outside this experiment.

Do not blindly use symmetric noise on all joints. Archived episode 45 starts
shoulder lift at -99.971° near its -100° limit and elbow at 94.781° near its
upper limit. The broader self-collision detector also finds a shoulder–lower-arm
penetration of 0.306 mm in this legacy initial state, which the earlier
shoulder–gripper-specific audit did not examine.

Independent CPU checks found that an elbow-only offset of -0.25° clears the
initial contact. For margin, the proposed standardized anchor uses shoulder
lift +2° and elbow -2° relative to episode 45. This anchor and the following
four offsets relative to it are within limits and initially self-contact free:

```text
shoulder_pan +2° only
shoulder_pan -2° only
[pan +2°, lift +1°, elbow +1°, wrist_flex +2°, wrist_roll +2°, gripper 0]
[pan -2°, lift -1°, elbow -1°, wrist_flex -2°, wrist_roll -2°, gripper 0]
```

Gripper opening remains unchanged. These are valid starting poses, not a claim
that the policy will stay collision-free afterward. The case builder should
repeat validity checks against its frozen model rather than relying only on
this review's original model.

## Required validity checks

Build and validate each case once before evaluating any checkpoint. Reject an
invalid case instead of changing it for a model that fails it. Check:

- Robot joint limits, finite state, and initial unwanted self-contact.
- Cube supported by the table, appropriate height/orientation and no initial
  cube/bin/robot overlap.
- Bin wholly supported by the table and separated from the cube.
- Plausible grasp and release reachability. Small radial bounds alone do not
  prove a collision-free IK solution; document the actual reachability check.
- Identical camera settings, 30 Hz action timing, 900-step deadline, and task
  success criterion across all models.

At the +6 cm world-Y cube offset, the cube edge is only about 1 mm from the
bin exterior wall. That is initially non-overlapping but can constrain gripper
approach, so it should be treated as a grasp-clearance stress case.

Archive-based perturbations must be applied **after** restoring the archived
state. The cube is a free body and moves through its qpos. The bin is static and
moves through `model.body_pos`. Call `mj_forward` after edits. Standardized
variants should reset velocities, position-hold controls and solver warm-start
state consistently. Restore the baseline model parameters between cases so bin
moves cannot accumulate. Preserve the legacy reset as a separately labeled
control rather than silently changing it.

## Fair checkpoint comparison and batching

Use TE coefficient 0.01, one executed action per prediction, and the existing
100-action model horizon for every checkpoint. Freeze case IDs, model/state
payloads, case ordering, seeds, batch grouping, numerical precision and backend
settings. Save hashes and software/config metadata alongside results. A smaller
batch can be necessary for memory; the grouping itself should not vary by model.

Batched evaluation requires particular care with completed cases. Gymnasium's
same-step autoreset can start another episode while other cases finish. Reuse
LeRobot's `FreezeAfterEpisodeEnd` wrapper or otherwise stop logging a case after
its first done, and slice all custom metrics through that done step inclusive.
Otherwise collision/roughness counts can include a second episode even though
LeRobot's reward metric masks it out. Preserve final pre-reset state for videos
and tracking diagnostics.

Report a per-case table as well as summaries by perturbation axis and size:

- Raw placement success and collision-free placement success.
- Self-contact pairs, duration/step counts and force summaries where available.
- Completion time, deadline failure and other failure categories.
- Target–position tracking error and per-joint motion roughness, with the
  gripper's percentage units separated from angular joints.

A motionless failed policy can have excellent aggregate smoothness, so do not
rank smoothness without outcome and progress. The legacy and standardized
controls are related controls, not two independent randomized samples.

## What this can say about retraining

These checkpoints were trained on the original data, including demonstrations
now flagged for self-contact. Removing recordings does not remove their
influence from existing weights. Comparing these checkpoints can reveal an
old-run performance plateau, suggest a sensible initial budget and expose
failures that the fixed evaluation missed. It cannot establish the necessary
training duration on the cleaned data or promise that any checkpoint is ready
for sim-to-real transfer.

Quarantine membership and clean-data counts must come from the completed broad
self-contact/substep audit, not the original 11-example shoulder–gripper audit.
Episode45 itself already demonstrates why that narrow audit was insufficient.
At batch12, effective data passes are `12 * training_steps / training_frames`;
a given update count revisits a smaller retained set more often. Train a fresh
clean-data pilot with intermediate checkpoints, evaluate it on the frozen suite,
and extend if the harder-case results are still improving. Keep a later,
untouched test set for final reporting because this suite is being used for
model selection.

## Targeted tests to preserve

1. Scenario hashes and first observations are identical for a case across model
   loads; different case IDs produce the intended pose/object offsets.
2. Away/lateral bin offsets are orthogonal and have the declared magnitudes.
3. Invalid joint-limit, initial-contact and overlapping-object cases are rejected.
4. Model/bin state is restored between cases and case order cannot change resets.
5. Expected cube/table and gripper/cube contacts are not self-collision failures.
6. A policy collision is logged without aborting later task-success evaluation.
7. A finished vector element does not generate extra trace/contact samples while
   another element continues.
8. A timed-out, motionless policy is not ranked above successful policies on a
   smoothness-only aggregate.

## Implementation review

The first implementation of `learning/robustness.py` and `evaluate_suite.py`
was reviewed independently. The runner reuses native LeRobot rollout/inference,
validates frozen state/model hashes, records batch grouping, and wraps every
case in `FreezeAfterEpisodeEnd` before same-step vector autoreset. This protects
custom contact, tracking and roughness traces from completed cases restarting
while another vector element continues. Historical invalid-start cases are
excluded from primary summaries.

Review feedback requested explicit initial robot/object contact rejection in
addition to self-contact and cube/bin overlap, integration of endpoint witnesses
before freezing moved-bin cases, and checkpoint/config/software identity in
resume provenance. The final frozen manifest remains the authority for accepted
case counts and preflight evidence; the proposed grid above is a candidate grid,
not a promise that every extreme offset is a valid benchmark task.

The implementation reduced candidate bin displacement to at most4 cm away and
4 cm sideways before reachability preflight, rather than keeping a requested
6 cm bin displacement without verifying it. The cube stress grid retains the
larger user-requested offsets. This keeps the limitations of numerical endpoint
checks explicit rather than calling every unverified local solve reachable.

## Endpoint reachability helper

`learning/reachability.py` now provides a damped least-squares endpoint witness
using the five arm joints, with fixed gripper opening. It solves tool position
plus a soft demonstrated tool-axis direction rather than asking a five-DOF arm
to meet all six dimensions of an arbitrary pose. Accepted witnesses respect
joint limits, have no robot self-contact or robot table/bin contact, lie within
5 mm of the target, and differ by no more than15 degrees in tool-axis direction.
A failed local solve is labeled unverified, not physically impossible.
Endpoint acceptance must not be relabeled a proved collision-free trajectory.

The original episode45 clear elevated bin-side seed is state253, with tool
site approximately `[0.21321,0.123806,0.106409]` meters. Independent CPU checks
found witnesses for bin shifts of2/4 cm away and2 cm away plus±2/±4 cm sideways.
A4 cm away/+4 cm sideways endpoint passed; the mirrored-4 cm sideways endpoint
was locally unverified at5.59 mm residual. Case generation should omit or reduce
unverified shifts before freezing the manifest, and retain the witness metadata.
This test does not prove a collision-free path, stable release, or grasp success.

The reviewer also checked grasp endpoint witnesses from demonstrated state125
(tool site approximately`[0.232590,0.014169,0.044175]`m). All16 cube shifts
±1/2/4/6 cm in worldX/Y passed the endpoint test. The largest residual was
4.90 mm for+6 cmX, with6.9 degrees tool-axis error. These witness checks do not
remove the approach-clearance caveat for the+6 cmY case near the bin wall.

## Outcome interpretation during the frozen 32-case benchmark

The running benchmark has32 frozen cases, including one historical invalid-start
control and31 primary cases. Do not change cases, thresholds or success rules
partway through the checkpoint comparison. Any additional scoring criteria
should be introduced as a separately versioned future benchmark.

The current success predicate establishes that the whole cube is released,
settled and inside the bin for0.5 seconds. It does **not** establish that the
policy explicitly grasped and lifted the cube before transporting it. Describe
this outcome as **released-and-settled placement success**, alongside collision-
free placement success, rather than treating the raw percentage as proof of
high-quality pick-and-place behavior. In particular, the+6 cm cube-Y case
places the cube close to the bin wall; a push, roll or other shortcut could
potentially meet the same final-state predicate. Watching representative
success videos is necessary before claiming the intended grasp strategy.

Bin displacement also has a tolerance issue. The interior is90 mm wide and the
cube is30 mm wide, leaving roughly30 mm center-position tolerance per axis for
an axis-aligned cube. A2 cm bin shift can therefore still contain a release
near the old bin center, without the policy adapting its destination at all.
Small bin-shift success is a useful tolerance check, but weak evidence that the
policy visually localizes the moved bin. A4 cm shift is more discriminating,
although the actual old release offset, cube orientation and trajectory must
still be considered. Report the magnitude-specific bin case table rather than
pooling2 cm and4 cm successes into a claim of object-location generalization.

For a future benchmark version, log cube height, gripper–cube contacts and
release timing through the episode. A separate intended-strategy score can
then require a verified lift/transport/release sequence while preserving the
current final-placement score. Those logs/criteria should be planned before
execution; the present robot-only motion traces and final cube position are
not sufficient to reconstruct every successful grasp retrospectively. Existing
checkpoint rankings remain useful as paired comparisons under the fixed
predicate, with these limitations stated explicitly.
