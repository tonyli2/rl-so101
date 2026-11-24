# so101_rl: EECE 576 domain-randomization project (MuJoCo)

The project code, separate from the scene in [../lerobot-tabletop](../lerobot-tabletop/).

## Model variants

| `--variant` | Model |
|---|---|
| `mounted` (default) | The stock-mount assembly from [lerobot-tabletop/mounts](../lerobot-tabletop/mounts/) (see [docs/simulation.md](../docs/simulation.md)): arm on the overhead stand's base adapter, hex-nut wrist camera mount, `overhead` camera. Regenerated from `mounts/config.json` at the start of every check or training run. |
| `legacy` | `scene_pick_place.xml` with Menagerie's unmodified arm at the world origin and the `front` camera. Kept to compare against the first results. |

Reach targets are sampled relative to the arm base, so they follow the mount if it moves. The
policy's task camera is `overhead` (mounted) or `front` (legacy). Results go in
`outputs/<check>/<variant>/`.

| File | Content |
|---|---|
| `scene_pick_place.xml` | legacy scene: one red 3 cm cube, the green box, a reach-target marker |
| `envs.py` | `So101World` (model, IK, friction knob, physics record), `So101ReachEnv` (stage 1), `So101PickPlaceEnv` (stage 2) |
| `check_milestone1.py` | servo step responses, scripted grasp (lift/hold/carry/release), friction sweep → `outputs/milestone1/<variant>/` |
| `train_reach.py` | stage 1 PPO reaching (Stable-Baselines3), evaluation, viewer → `outputs/reach/<variant>/seed*/` |
| `make_bc_dataset.py` | replays Tony's demonstrations into env observations (both cameras, 96×128) → `outputs/bc_data/` |
| `encoders.py` | `ResNetExtractor`: frozen ImageNet ResNet18 per camera, safe from SB3's re-initialization |
| `train_bc.py` | behavior cloning as the SB3 PPO policy, closed-loop evaluation, viewer, TensorBoard → `outputs/bc/<run>/` |

## Behavior cloning

Demonstrations: the `sim-baseline-clean-50-v1` GitHub release (50 episodes, 25,666 frames),
extracted to `lerobot-tabletop/recordings/sim-baseline` as its release notes describe. Split:
episodes 0-44 train, 45-49 validation. `make_bc_dataset.py` restores each frame's archived MuJoCo
state into the env and records what the policy would observe; the archived scene is identical to
the mounted model (checked at load).

```powershell
python make_bc_dataset.py                 # ~4 min, ~1.9 GB in outputs\bc_data
python make_bc_dataset.py --perturb 3     # noise augmentation: ~13 min, ~5.2 GB
# Current best recipe: current frame only, relative actions, noise copies, idle starts trimmed
python train_bc.py --epochs 60 --history 1 --no-velocities --perturbed --action relative --trim-idle --run-name my_run
python train_bc.py --eval outputs\bc\my_run\bc_model.zip    # 115 distinct episodes, 8 in parallel, ~8 min
python train_bc.py --view outputs\bc\my_run\bc_model.zip
tensorboard --logdir outputs\bc\tb
```

The saved `bc_model.zip` is a PPO checkpoint, so PPO fine-tuning can start from it directly.
Each run saves `bc_config.json` (observation and action settings, relative scales), which
`--eval`, `--view` and `camera_ablation.py` use to rebuild the matching env.

Closed-loop results (115 episodes, cube and box as in the demonstrations):

| Change (cumulative) | Success | Lifted |
|---|---|---|
| ResNet18, balanced state branch, current frame, absolute actions | 1.7% | 12.2% |
| + noise augmentation (3 copies, arm nudged by σ = 2.5°) | 4.3% | 18.3% |
| + relative actions + idle starts trimmed (20 epochs) | 12.2% | 58.3% |

Lessons:
- Leave commanded targets out of the observations (`include_targets=False`), or BC copies its
  last command.
- Use pretrained encoders: a from-scratch CNN overfits the wrist camera on 45 demonstrations.
- Predict motion (`--action relative`), not absolute targets: 87% of demonstrated commands are
  within 1° of the measured arm, smaller than the policy's error on absolute targets.
- Noise augmentation teaches recovery; select its final epoch (clean validation loss picks the
  least-trained model).
- Offline action error does not predict closed-loop success; judge runs by `--eval`.
- With the cube and box fixed (as in this dataset) the policy relies on joint state, not vision;
  demonstrations with varied cube positions are needed for a vision-based policy.
- Remaining failure: the gripper closes ~1 cm too shallow on the cube, which slips out while
  being carried.

## PPO fine-tuning (proposal: BC checkpoint -> nominal vs randomized PPO)

```powershell
python train_ppo.py --bc outputs\bc\resnet18_h1_dart_rel_e20_seed0_last\bc_model.zip --run-name nominal_s1 --seed 1 --steps 500000
python train_ppo.py --bc ...\bc_model.zip --run-name randomized_s1 --seed 1 --steps 500000 --randomize
python train_bc.py --eval outputs\ppo\<run>\model_300000.zip   # (copy bc_config.json next to it) 115-episode eval
tensorboard --logdir outputs\ppo\tb
```

- Starts from a BC checkpoint (same network); episodes start from the demonstrations' start poses.
- Critic warm-up (actor frozen) for 50k steps, then small learning rate, clip 0.1, `target_kl` 0.02.
  Encoder heads stay at BC unless `--train-encoder`; the ResNet backbones are always frozen.
- Reward: potential-based progress (reach, lift, carry toward the box, cube in the box) plus a
  success bonus; stalling earns nothing.
- 40-episode deterministic evaluation before training and every `--eval-every` steps; each
  evaluated checkpoint is saved (`model_<step>.zip`, `eval_history.json`).
- `--randomize [texture lighting camera friction]`: resample those factors every training episode
  (`randomization.py`; 6 training textures, 3 held out for evaluation). Evaluation sweeps use the
  env's `fixed_factors`.

Nominal pilot (from the 20-epoch BC model, 1M steps, 1.7 h): 12.5% -> 100% success on the
40-episode evaluation by 300k steps, then ~85-92%. The 300k checkpoint scores **96.5%** on the
115-episode evaluation (BC: 12.2%); successful grasps hold the cube at -0.35 cm along the fingers
(BC's slipping grasps: +1.2-1.4 cm) and finish in 4.5 s (demonstrations: ~17 s).

`run_main_experiment.py` runs the six main runs (nominal / randomized x seeds 1-3) back to back
from `outputs/ppo/main_plan.json`, which it re-reads before each run.

### Randomization ranges: curriculum and calibration

Randomized PPO at the full `randomization.Ranges` defaults collapses: the BC starting policy cannot
succeed there (lighting alone: 0% lifts), so PPO gets no successes to learn from. Without a
curriculum it fell from 22.5% to 0% by 200k steps; with a curriculum it reached 67.5% at 300k and
then fell to 5% as strength reached the full ranges. Two fixes, both now defaults:

- **Curriculum** (`train_ppo.py --randomize-curriculum 0.6`): randomization strength is 0 during
  the critic warm-up and ramps linearly to 1 at 60% of the steps.
- **Calibrated ranges** (`calibrate_ranges.py`, written to `randomization_ranges.json`, read by
  `train_ppo.py --randomize`): each factor is tested alone on the BC policy at 25/50/75% of its full
  range; its cap is the largest strength keeping >= 2/3 of the nominal lift rate.

| Lifted (16 episodes; nominal 44%) | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| texture | 38% | 31% | 25% | 25% |
| lighting | 25% | 31% | 25% | 0% |
| camera pose | 50% | 38% | 44% | 50% |
| friction | 50% | 44% | 50% | 38% |

Caps: texture 0.5, lighting 0.25 (the rule's 0.5 passed by one episode), camera 0.75, friction
0.75; effective ranges at full strength: friction 0.55-1.15, light brightness x0.9-1.1, camera
field of view +-2.2 deg. All four together at the caps: 19% lifted (factors compound).
With these, `main_randomized_cal_s1` reached 95% at 300k and 100% at 400k on the nominal
evaluation while training under full (calibrated) randomization.

### Actuator factors and control latency

`factor_sweep.py` also tests servo strength (`torque_scale`), stiffness (`kp_scale`) and control
latency (`latency_steps`: commands take effect N 33 ms control steps late). Torque x0.5 and
stiffness x0.5/x1.5 barely matter (94-98% for the 500k PPO policies). Latency does: 90-94% at 1
step, then 0-1% at 2 steps. The real arm measures ~76-110 ms (camera 26 ms + frame age 0-33 ms +
servo dead time ~50 ms), i.e. 2.3-3.3 steps, so this sweep setting matches the real robot.

The cause is the action base. Relative actions add the policy's motion to the *measured* joints,
and under latency those are stale, so the arm chases old positions. Basing the motion on the
*previous command* instead (`envs relative_base = "command"`, `factor_sweep.py --relative-base
command`) fixes it with no retraining (`outputs/sweeps/step500000_cmdbase`, 40 episodes per cell):

| setting | nominal PPO, measured base | nominal PPO, command base | randomized PPO, measured base | randomized PPO, command base |
|---|---|---|---|---|
| nominal | 97% | 96% | 99% | 97% |
| latency 2 steps | 1% | 94% | 0% | 97% |
| latency 3 steps (measured real) | - | 95% | - | 97% |
| friction 0.30 | 74% | 85% | 97% | 93% |
| combined_moderate | 71% | 63% | 89% | 96% |

Training with latency randomized over 0-3 steps (`train_ppo.py --randomize ... actuator`) did not
help with the measured base: 0% at 2 and 3 steps after 200k steps (pilot stopped). Deploy with the
command base.

## Run

```powershell
conda activate lerobot
cd ~\rl-so101\so101_rl
python check_milestone1.py
python train_reach.py --seed 0                         # 1M steps, ~6 min on CPU
python train_reach.py --view outputs\reach\mounted\seed0\model.zip
tensorboard --logdir outputs\reach
```

## Interfaces (proposal section 3)

- **Actions:** 6 values in [-1, 1], five arm joint-position targets and a gripper command, tracked by
  the position servos. The default `action_mode="delta"` moves the targets by at most 0.05 rad per
  step; `"absolute"` maps onto the actuator ranges.
- **Observations:** joint positions (normalized), joint velocities and, optionally
  (`include_targets`), current targets, stacked over the last 3 control steps. The pick-place env
  adds the overhead RGB camera and optionally (`wrist=True`) the wrist camera (96×128, 3 frames
  stacked on channels). Object poses go into rewards, never observations. The reach env also gets
  the target position, its only way to know where the target is.
- **Reset:** `reset(options={"joints": q})` starts from given joints (e.g. a demonstration's start).
- **Control:** 30 Hz, 7 physics substeps (dt = 4.76 ms), Menagerie's tuned STS3215 servos.

## Contact and friction

The jaw collision geoms have `priority="1"`, so **finger friction governs finger–cube contacts**.
`So101World.set_object_friction(mu)` therefore sets the fingers (and the cube, for consistency);
cube–table contacts keep μ = 1. The fixed physics settings are recorded in `report.json`
(`physics_settings`).
