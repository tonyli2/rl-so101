# Local ACT training and evaluation

Use the `lerobot` Conda environment. Training extras are installed with:

```bash
python -m pip install -e '/home/tonyli2/code/lebrobot[training]'
```

`lerobot-tabletop/train_act.py` registers our local environment, then calls the
native LeRobot trainer. It reuses LeRobot's dataset reader, ACT policy,
normalization, train/validation split, optimizer, checkpointing, rollout evaluator,
video encoding and Trackio logger. It never opens a leader serial port.
`evaluate_act.py` registers the same task for LeRobot's standalone evaluator.
No upstream LeRobot files are modified.

## Clean 50-demonstration baseline

```bash
conda activate lerobot
cd ~/code/enph576/lerobot-tabletop
# Keep the run alive if your terminal connection drops.
tmux new -s act

# Prevent automatic sleep for the duration of training.
systemd-inhibit --what=sleep --mode=block --why="ACT training" \
python train_act.py \
  --dataset.repo_id=local/sim-baseline \
  --dataset.root="$PWD/recordings/sim-baseline" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.1 \
  --policy.type=act \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --accelerator.mixed_precision=fp16 \
  --accelerator.grad_scaler.init_scale=128 \
  --policy.push_to_hub=false \
  --batch_size=12 \
  --num_workers=2 \
  --prefetch_factor=1 \
  --steps=30000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_steps=2000 \
  --max_eval_samples=600 \
  --env.type=so101_tabletop \
  --env.episode_length=900 \
  --env.reset_dataset_root="$PWD/recordings/sim-baseline" \
  --env.reset_episode=0 \
  --env_eval_freq=10000 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --trackio.enable=true \
  --trackio.project=enph576-act \
  --trackio.run_name=act-clean-50demos-30k \
  --output_dir=../outputs/train/act-clean-50demos-30k
```

This 30k budget is an initial pilot, not a proven convergence requirement. Save
intermediate checkpoints and evaluate them with the frozen robustness suite;
extend training only if useful outcomes improve. The one-start in-training
evaluation below is a reference check, not a robustness score.

Use a fresh output directory and run name for a new experiment. This command
reads the local dataset and saves everything locally. The pretrained ResNet18
backbone downloads once if uncached. The recordings are not uploaded. Close
VocalLinux before training. The sleep inhibitor releases automatically when
training exits. In tmux, detach with Ctrl-b then d; reconnect with
`tmux attach -t act`. Trackio is enabled locally and will create the project
and run automatically.

The native trainer's actual AMP control is `accelerator.mixed_precision`;
`policy.use_amp` alone does not select accelerator AMP in this installed version.
The lower initial loss scale avoids wasting early updates on FP16 overflow.

The active dataset has 50 clean episodes: 32 retained episodes plus 18 newly
collected replacements. A 10% split holds out the last 5 complete episodes,
leaving 45 for training. Historical frame counts below describe the original run.
LeRobot currently uses the dataset's stored normalization statistics (computed
on all episodes), so validation is a practical diagnostic rather than a strict
unseen-data benchmark. `max_eval_samples=600` caps validation cost; native
subsampling chooses the first frames per task, so it does not cover all phases
uniformly. Set it to `0` for the full validation set.

## What evaluation measures

The evaluation scene matches collection: fixed scene, overhead and
wrist RGB at 640x480, 30 Hz, five joint angles in degrees plus gripper opening
in 0–100 units. ACT uses saved normalization and its default 100-action chunks.
The overnight command restores the full archived starting state of demonstrated
episode 0 (zero-based, in the active dataset), including arm pose, cube pose, velocity and controls.
It checks scene compatibility before restoring. This matters because the
recorded starts vary with the leader pose during preview. Without
`env.reset_dataset_root`, evaluation uses the XML home reset. Change
`env.reset_episode` to assess other demonstrated starts.

Each episode has a 900-step / 30-second deadline, exceeding the longest recorded
demonstration. Rendering and rollout run headlessly through EGL.

Success requires the entire red cube inside the bin, below the rim, with low
linear/angular velocity, no robot contact, continuously for 0.5 simulation
seconds. This criterion only affects evaluation; collection is unchanged.
The geometric tolerances permit 2 mm of contact penetration.

Evaluation is deterministic with the current fixed reset and ACT inference.
One episode per checkpoint is sufficient for this baseline; repeating 10 seeds
would repeat the same initial condition, not estimate generalization. Add
multiple randomized episodes only when reset variation has been explicitly
introduced and measured against the baseline. A high fixed-scene success rate
does not establish sim-to-real readiness.

## Review results

```bash
trackio show --project enph576-act
```

Select the run to inspect `train/loss`, `train/l1_loss`, `train/kld_loss`,
`eval/eval_loss`, `eval/pc_success`, GPU/system metrics and `eval/video`.
ACT training loss includes a KL term; compare the L1 component with validation
loss rather than interpreting the total training/validation gap as overfitting.
Nested per-task summaries may produce an upstream logger warning; the overall
success scalar and videos are logged separately.

Checkpoints, optimizer state, processors, normalization and configuration are
under `outputs/train/<run>/checkpoints/`. `last` points at the newest checkpoint.
Rollout videos are under `eval/videos_step_<step>/`. These files are ignored by
Git. Trackio's local database is under `~/.cache/huggingface/trackio`.

For an existing checkpoint, from `lerobot-tabletop/`:

```bash
python evaluate_act.py \
  --policy.path=../outputs/train/act-sim-baseline-100k/checkpoints/last/pretrained_model \
  --env.type=so101_tabletop \
  --env.episode_length=900 \
  --env.reset_dataset_root="$PWD/recordings/sim-baseline" \
  --env.reset_episode=0 \
  --eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false \
  --output_dir=../outputs/eval/act-sim-baseline-100k
```

To resume interrupted training, use the same registration launcher and the
checkpoint's saved configuration:

```bash
python train_act.py --resume=true \
  --config_path=../outputs/train/act-sim-baseline-100k/checkpoints/last/pretrained_model/train_config.json
```

## Readiness verification (2026-11-22)

Native FP16 training at batch 12, held-out validation, periodic/final saving,
checkpoint resume with optimizer/scaler state, multi-chunk inference, a full
900-step rollout, archived-state reset, video encoding and local Trackio logging
have passed. All 153 saved optimizer parameter states reached the expected
update count. Four evaluator tests cover units/mapping, the placement hold/reset,
wall/height rejection, robot contact and cube motion.

The measured training window was approximately 0.40 seconds per step including
data loading, with 4.97 GiB peak allocated GPU memory. Allow roughly 11 hours
plus validation, evaluation and checkpoint overhead for 100k steps; this is an
estimate from a short run. Checkpoints every 10k steps allow recovery after an
interruption. These checks validate execution, not eventual task success.

The final data audit also passed: all 26,006 rows had finite six-value state and
action vectors; episode indices/timestamps were consistent; all 100 videos
(52,012 camera frames) decoded at 640x480/30 Hz; and native LeRobot loading
returned correct image shapes, 100-action targets and padding masks at both
ends of all 50 episodes. The exact documented 100k configuration parses and
validates, its output directory is unused, and `pip check` found no broken
requirements. Approximately 1.7 TB of disk is free, well above the roughly
12–15 GB required for twenty full training checkpoints plus videos/logs.
The system sleep inhibitor was exercised successfully. Temporary run outputs
and tracking records have been removed; the reusable evaluator tests remain.

For post-training motion analysis, see [ACT motion diagnostics](motion-diagnostics.md).

After collision quarantine, original episode45 is not in the retained dataset.
Historical checkpoint reproduction must explicitly use
`--env.reset_dataset_root=../outputs/dataset-backups/sim-baseline-original-50`
and `--env.reset_episode=45` from the tabletop directory. New training should
use the active 50-episode clean dataset and choose its budget from the robustness
results rather than automatically repeating100k. See
[the expanded evaluation pipeline](robustness-evaluation.md).
