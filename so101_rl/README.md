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
- **Observations:** joint positions (normalized), joint velocities, current targets (including the
  gripper), stacked over the last 3 control steps. The pick-place env adds the front RGB camera
  (96×128, 3 frames stacked on channels). Object poses go into rewards, never observations. The
  reach env also gets the target position, its only way to know where the target is.
- **Control:** 30 Hz, 7 physics substeps (dt = 4.76 ms), Menagerie's tuned STS3215 servos.

## Contact and friction

The jaw collision geoms have `priority="1"`, so **finger friction governs finger–cube contacts**.
`So101World.set_object_friction(mu)` therefore sets the fingers (and the cube, for consistency);
cube–table contacts keep μ = 1. The fixed physics settings are recorded in `report.json`
(`physics_settings`).
