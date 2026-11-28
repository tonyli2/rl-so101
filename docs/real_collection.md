# Real leader/follower demonstrations

`lerobot-tabletop/real_record_gui.py` is a desktop window for recording real pick-and-place
demonstrations: the leader drives the physical follower, and both cameras and the follower's
joints are saved as a native LeRobot dataset. It is a separate dataset from the sim recordings;
do not mix them. The recording logic lives in `real_record.py`, which also has a browser front
end (`real_record.py`, http://localhost:8002; `--cameras browser` lets Chrome capture the cameras
when Python cannot get camera permission, at the cost of syncing and hidden-tab pauses).

## Setup (once, on the Mac with the arms and cameras)

LeRobot at the team's commit (`9e5ee9859480e425a97a6836bb3fee479539beb5`) and Qt in a venv:

```bash
python3.13 -m venv .venv-lerobot
.venv-lerobot/bin/pip install "lerobot[feetech,dataset,hardware,core-scripts] @ git+https://github.com/huggingface/lerobot.git@9e5ee9859480e425a97a6836bb3fee479539beb5"
.venv-lerobot/bin/pip install --only-binary=:all: pyside6-essentials
```

## Run

Start it from the macOS **Terminal app** or Finder, not from inside Claude: macOS then asks
Terminal for camera access (allow it, or turn on Terminal in System Settings > Privacy &
Security > Camera).

```bash
cd ~/Downloads/EECE576/rl-so101/lerobot-tabletop
../../.venv-lerobot/bin/python real_record_gui.py
```

Python reads both cameras with LeRobot's OpenCV cameras; each control tick takes the newest frame
of each, so the views are at most one frame apart. Occasional Feetech read timeouts skip a tick;
10 failures in a row stop a recording.

1. **Setup.** Click which thumbnail is the overhead camera and which is the wrist camera. The
   leader and follower are found automatically from the calibration offsets stored in their
   motors (they must match `calibration/leader.json` and `follower.json`). Hold the leader near
   the follower's pose and click **Connect**: the follower then copies the leader.
2. **Once.** Fold the arm into a rest pose and click **Set rest pose**. Put the cube on the home
   mark and click **Cube is on the mark**: placement offsets are then measured from the mark.
3. **Each episode.**
   - Return the arm to rest (all joints green, within 5 deg).
   - Place the cube on the yellow outline. The window also gives the move in numbers; a cross
     turns green within 1.5 cm. Close is enough: the detected position is logged at Start.
   - Hands out of view, **Start** (Space), do the pick and place, **Save** (Space). **Discard**
     (D) redoes the same placement.
4. **Pause teleop** (between episodes) holds the follower still and ignores the leader;
   **Resume** eases the follower to the leader over 1.5 s.
5. **Stop & finish dataset** when done, and wait for the window to close. Until then the
   dataset's parquet files are not finalised and cannot be read; do not force-quit or unplug.
   Stopping turns the follower's torque off, so leave it in a pose that can go limp. Running
   the same command again appends to the dataset.

Only Start to Save is recorded, so resets never enter the data. Episodes stop at 60 s.

## Placements (domain randomisation)

`<root>_plan.json` fixes 50 placements: a 5x5 grid over +-3 cm forward/back and left/right
around the cube's home spot (`mounts/config.json` `task.cube_position`), each spot twice with a
different rotation from -20, -10, 0, +10, +20 deg, in a fixed shuffled order. The bin does not
move. Keep lighting, the bin and the rest pose constant. `<root>_episodes.jsonl` records, per
saved episode, the planned placement and the cube position detected at Start (cm from the mark).

The outline uses the measured overhead camera model (`mounts/config.json` `overhead_camera`
and its intrinsics), anchored at the mark (`<root>_home.json`).

## Output

- `recordings/real-pickplace-1/`: LeRobot dataset, 30 Hz, `observation.images.overhead` and
  `observation.images.wrist` (640x480 raw, not undistorted), `observation.state` (follower) and
  `action` (leader), degrees with gripper 0-100 in the real LeRobot convention (not the sim
  convention; see `calibration/follower_sim_offsets.json`).
- `recordings/real-pickplace-1_frames/{overhead,wrist}/`: PNG snapshots every 2 s for the whole
  session, including resets, for sim/real image comparison (`--snapshot-every`).

First dataset, 2026-11-27: 50 episodes, 39,660 frames (12.7-60 s, median 24.8 s). Episode 2 hit
the 60 s cap; check it before training.

## Train (on the desktop GPU)

Copy `recordings/real-pickplace-1` to the desktop and train ACT on it like the sim baseline in
[training.md](training.md), but without the sim environment evaluation:

```bash
python train_act.py \
  --dataset.repo_id=local/real-pickplace-1 \
  --dataset.root="$PWD/recordings/real-pickplace-1" \
  --dataset.video_backend=pyav --dataset.eval_split=0.1 \
  --policy.type=act --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01 \
  --policy.device=cuda --policy.use_amp=true --accelerator.mixed_precision=fp16 \
  --policy.push_to_hub=false --batch_size=12 --num_workers=2 --prefetch_factor=1 \
  --steps=30000 --save_freq=10000 --log_freq=100 \
  --output_dir=../outputs/train/act-real-50demos-30k
```

Evaluate the 10k/20k/30k checkpoints on the real arm at placements inside and just outside the
+-3 cm grid.

## Run a trained ACT checkpoint on the real arm

`lerobot-tabletop/real_run_act.py` opens a window that runs a LeRobot ACT checkpoint on the
follower (the leader is not used). Use a separate venv with PyTorch 2.7, which still supports the
Mac GPU (MPS) on macOS 13 (PyTorch 2.11 needs macOS 14 for MPS):

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv-act
.venv-act/bin/pip install "torch==2.7.1" "torchvision==0.22.1"
.venv-act/bin/pip install "lerobot[feetech,dataset,hardware] @ git+https://github.com/huggingface/lerobot.git@9e5ee9859480e425a97a6836bb3fee479539beb5" "torch==2.7.1" "torchvision==0.22.1"
.venv-act/bin/pip install --only-binary=:all: pyside6-essentials
```

Start it from Terminal or Finder (camera permission):

```bash
cd ~/Downloads/EECE576/rl-so101/lerobot-tabletop
../../.venv-act/bin/python real_run_act.py --policy checkpoints/act-real-49demos-30k/020000/pretrained_model
```

- Inputs match the recordings: raw 640x480 RGB from both cameras, follower joints in the real
  convention, no undistortion or sim offsets.
- ACT takes ~85-150 ms per chunk on the M1 Pro GPU (~0.6 s on CPU), so the window predicts full
  chunks continuously in a worker thread, anchors each at the tick its observation was taken,
  and sends ACT's temporal ensemble (coefficient from the checkpoint, 0.01) at 30 Hz. Each
  command is limited to `--max-step` degrees per tick (default 4, i.e. 120 deg/s).
- Run policy / STOP (Space; Esc also stops, holding the arm where it is), Return to rest (3 s
  ease to the recorder's rest pose), Success / Fail log each trial with the cube position seen at
  Run to `<checkpoint run>/real_trials.jsonl`.
- First runs: keep a hand near the follower's power, start with the cube at the home mark, and
  compare checkpoints (10k/20k/30k) on the same placements, inside and just outside the +-3 cm grid.
