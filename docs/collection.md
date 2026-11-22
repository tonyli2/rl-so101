# Demonstration collection

## Randomized cube/bin collection (default for real-leader recording)

```bash
conda activate lerobot
cd ~/code/enph576/lerobot-tabletop
python record_demonstrations.py --root recordings/sim-collision-free \
  --leader-port /dev/ttyACM0 --calibration-file calibration/leader.json \
  --episodes 50
```

New real-leader sessions randomize task positions on initial preview and each
reset, save, or discard. The cube is sampled within +/-4 cm in world X/Y.
The bin is sampled 0–4 cm away and +/-4 cm sideways relative to the original
cube-to-bin direction. Positions remain fixed during the episode, except for
normal cube physics. You choose the arm starting pose before pressing S.
Rejection sampling excludes overlapping/nearby cube/bin layouts (1 cm clearance
beyond conservative footprints), table-edge violations, and layouts without
verified approach/release IK endpoints. Endpoint checks do not prove a feasible
collision-free path. Rejection changes the accepted distribution from uniform.

The overhead camera must see at least 16 segmented cube pixels with a 2-pixel
frame margin. This is checked during sampling and again at S after the arm has
moved; moving the arm out of the way can clear an occlusion. If preview motion
pushes the cube more than 2 mm from its sampled XY position, reset before starting.
Robot contact with cube/bin at S also blocks recording. The usual self-collision
gate still checks every physics step throughout recording.

The sampler reuses existing endpoint IK and demonstrated reference poses from
episode 0 of `recordings/sim-baseline`; override with `--layout-reference PATH`.
Camera parameters, action units, and control frequency are unchanged. Each saved
episode includes layout parameters, positions, RNG state, and endpoint witnesses;
its scene snapshot contains the moved fixed bin and its state archive contains
the cube pose. `--seed` selects a reproducible sampler stream; resumed sessions
incorporate their existing episode count, rather than restarting at layout zero.

Use a separate dataset for randomized collection. Resume with the same command
plus `--resume`. Mixing fixed and randomized episodes is refused by the CLI.
For legacy fixed-layout collection, explicitly pass `--fixed-layout`, including
when appending to `recordings/sim-baseline`. Synthetic tests remain fixed unless
using the session API to explicitly supply a sampler.


The collection format is independent of the later policy. It stores ordinary
single-step observations/actions in a native LeRobot dataset, plus a numerical
MuJoCo archive for replay and camera re-rendering. ACT chunks, BC normalization,
visual embeddings and PPO rollouts are produced later; they are not baked into
the source demonstrations.

## Before recording

```bash
conda activate lerobot
cd ~/code/enph576/lerobot-tabletop
```

Use the same physical leader calibration as the viewer. The physical follower
does not need to be connected for simulated collection. New datasets remain
local; collecting does not upload anything to Hugging Face.

Start with a short hardware-free smoke test:

```bash
python record_demonstrations.py --root recordings/smoke \
  --synthetic --headless --steps 6 --episodes 2 --width 160 --height 120
```

Synthetic episodes exercise the writer and archive. They are labeled synthetic,
not successful human demonstrations, and must not be included in an ordinary BC
training set. The lower image size above is a test setting; collect real pilot
demonstrations at the intended dataset resolution.

## Collect with the real leader

```bash
python record_demonstrations.py --fixed-layout --root recordings/sim-baseline \
  --leader-port /dev/ttyACM0 --calibration-file calibration/leader.json \
  --episodes 3
```

The default is two 640×480 cameras at 30 Hz. No graphics environment variables
need setting. The normal command opens a viewer; `--headless` records without
one. The existing calibration must match the leader; the script does not run
calibration or connect to a physical follower.

The desktop default is one maximized four-panel collection window: front,
overhead/top, side, and elevated diagonal. Use its buttons or keyboard shortcuts while that
window is focused; there is no need to return to the terminal. Closing the window
requests a graceful quit. Desktop previews require Tk.

The display runs in a separate process with a **30 Hz target**, independently of 30 Hz
recording. A bounded queue skips display updates when the window is busy; display
rendering does not run inside the control loop. Only overhead and wrist images are
recorded; preview zoom, aim and cropping do not change those dataset cameras.
`--preview-fps 15` or `--preview-fps 5` reduces display GPU/CPU work. Shared machine
load can still affect recording timing. `--single-view` restores the original
orbitable MuJoCo viewer with terminal controls; `--multi-view` remains accepted
for existing commands.

| Key | Action |
|---|---|
| `s` | Begin recording from the current preview pose |
| `f` | Finish, record a one-second held-target settling tail, then pause for review |
| `k` | Keep the reviewed attempt as a human-confirmed successful demonstration |
| `d` | Discard the current attempt and reset the scene |
| `r` | Reset, discarding any active attempt |
| `q` / Escape | Quit; an unaccepted attempt is discarded |

Position the arm before pressing `s`; avoid moving or grasping the cube during
preview. A successful demonstration should include pickup, release, withdrawal
and settling. Press `f` after withdrawing the gripper, then `k` only if the cube
remains in the bin. Do not mark an unsuccessful attempt as successful just to
retain it. Current pilot collection keeps accepted demonstrations and discards
rejected attempts; a failed-attempt archive is not implemented.

The default maximum is 1,800 recorded steps (60 simulated seconds) before
settling/review. `--steps`, `--settle-seconds`, and `--episodes` adjust these
limits. Completed episodes are finalized individually. Graceful Ctrl+C discards
the active unaccepted attempt and preserves already finalized episodes.

To append more episodes, repeat the same command with `--resume`:

```bash
python record_demonstrations.py --fixed-layout --root recordings/sim-baseline --resume \
  --leader-port /dev/ttyACM0 --calibration-file calibration/leader.json \
  --episodes 3
```

Resume checks completion markers, committed archive metadata/counts, archive
presence, and matching image dimensions/FPS. It does not re-decode old episodes.
Each new episode is fully validated when Keep saves it, before it is marked
committed. Use `replay_demonstrations.py --root <directory> --validate --headless`
for an explicit full audit, including old recordings that may have been changed
or damaged since collection. `--episodes` counts episodes accepted during this run. Use a
new root for a different image size or collection protocol.

An active attempt creates `COLLECTION_INCOMPLETE.json`. Graceful discard removes
it. If it remains after a crash or failed save, validation/resume stops rather
than silently treating partial data as complete. Inspect the saved files before
removing any marker; automatic crash repair is not provided.

## What is recorded

The learning dataset uses these fields:

| Field | Meaning |
|---|---|
| `observation.images.overhead` | RGB overhead image before applying the action |
| `observation.images.wrist` | RGB wrist image before applying the action |
| `observation.state` | Actual simulated joint positions |
| `action` | The joint targets actually applied after simulation limits |

Joint order is `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`. Arm positions/targets use degrees and the gripper uses
0–100 opening. The raw leader request is retained in the simulator archive so
that clipping and mapping differences can be inspected.

Each stored row represents `(observation_t, action_t)`. An episode of T actions
has T dataset rows and T+1 simulator-state snapshots, including the final state
before any reset. LeRobot generates its standard timestamp from frame index and
FPS; actual simulator and wall-clock timing are stored separately.

The archive stores the applied MuJoCo controls, raw requests, outcome and
provenance, numerical integration state, and the scene/configuration needed to
restore it. It is associated with the learning dataset by episode ID and frame
index. Large assets are shared through scene snapshots rather than copied into
every episode.

## Validate and replay

After the recorder has finalized the dataset:

```bash
python replay_demonstrations.py --root recordings/smoke --validate --headless
python replay_demonstrations.py --root recordings/smoke --replay commands --headless
python replay_demonstrations.py --root recordings/smoke --replay states --headless
```

State replay restores the recorded trajectory directly. Command replay starts
from the recorded initial state and integrates the applied controls again; it
checks whether the pinned simulator configuration reproduces the trajectory.

Camera-only changes can generate another version of the learning dataset:

```bash
python replay_demonstrations.py --root recordings/smoke \
  --rerender recordings/smoke-rerendered --headless
```

Use the re-render tool's `--camera-overrides` option for calibrated camera changes.
The JSON keys are MuJoCo camera names (`overhead` and `wrist_cam`), with local
`pos`, `quat` (w,x,y,z) and/or `fovy` values, for example:

```json
{
  "overhead": {"fovy": 65},
  "wrist_cam": {"fovy": 68}
}
```

```bash
python replay_demonstrations.py --root recordings/smoke \
  --rerender recordings/smoke-calibrated --headless \
  --camera-overrides camera-overrides.json
```

Add both `--width` and `--height` to re-render at a different output resolution.
This makes a new dataset with a new image schema; it does not stretch previously
recorded video frames or modify the source dataset.

The source recording is preserved. Camera changes do not prove that the same
motion remains successful under different cube sizes, friction or actuators:
those require physics replay and outcome review.

## Acceptance and limitations

- Keep successful, failed, interrupted and synthetic outcomes distinct. Select
  complete, explicitly successful human episodes for the initial BC dataset.
- Leave a release/withdrawal/settling tail before accepting a pick-and-place.
  Merely holding a cube over or inside the bin is not task success.
- Split data by original episode, not individual frames. Re-rendered copies of
  an episode must remain in the same training/evaluation split.
- The recorder uses LeRobot's bounded streaming-video writer. Images are
  compressed; decoded MP4 frames are not expected to match raw renders exactly.
- Graceful shutdown finalizes native files. Power-loss or forced-kill recovery
  is not guaranteed; treat incomplete attempts as unusable until validated.
- Control-timestep snapshots support camera pose/projection re-rendering, not
  exact reconstruction of future rolling-shutter or motion-blur effects.
- Privileged object poses and simulator state are archive/evaluation data, not
  default policy inputs. A deployable MLP policy needs real-world-available
  visual features or estimates to respond to object motion.
- Human demonstrations can initialize BC or provide an auxiliary imitation
  loss. PPO still needs fresh policy rollouts and policy action probabilities.

The focused hardware-free tests can be run with:

```bash
python -m unittest discover -s tests -p 'test_recording.py' -v
```

Run these checks sequentially on this desktop, rather than starting multiple
rendering/encoding test processes at once. Whole-run cadence includes preview
as well as recording; isolated warnings are not the achieved rate of an entire
episode. Inspect the cadence summary and validation results together.

## Local baseline and backups

The baseline archives require MuJoCo **3.17.0**, which is pinned in
`lerobot-tabletop/requirements.txt`. The collection environment uses LeRobot
**0.6.2**. Historical setup snapshots describe earlier installations, not
necessarily the runtime used for these demonstrations.

The initial `recordings/sim-baseline` collection contains 50 accepted episodes
and 26,006 frames at 30 Hz (640×480 overhead and wrist images). This is a local
snapshot of collection progress, not a training/evaluation split or a claim of
policy success. Training has not been implemented here.

Recordings are ignored by Git. A code commit or GitHub push does **not** back up
the dataset. Preserve the entire dataset directory, including `sim_archive/`,
when making a separate data backup. Existing scene snapshots retain the model
and source code used at collection time even if the working code changes.

## Self-collision rejection

Collection now checks MuJoCo contacts between nonadjacent robot bodies at every
physics substep, including the initial pose and the final refreshed state. A
colliding pose cannot start recording. If contact occurs while recording, the
attempt is automatically discarded, its pending marker/buffer is cleared, and
the preview reports REJECTED with the contacting body names. Move the leader
clear or reset, then start another attempt. Expected gripper-cube and cube-table
contacts are allowed. A successful placement alone cannot override this gate.

After quarantine, 32 accepted episodes remained; 18 clean replacements have
since restored the fixed-layout baseline to 50. See
[dataset quarantine](dataset-quarantine.md). Reindexed episode numbers differ
from historical videos. Do not resume collection into the quarantine folder.
