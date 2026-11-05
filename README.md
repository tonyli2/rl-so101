# SO-101 simulation project

First milestone: control a MuJoCo SO-101 follower with the real, calibrated
SO-101 leader through LeRobot.

See [sim_setup.md](sim_setup.md) for project goals, hardware, camera mounts,
and later demonstration plans.

## Environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/code/enph576
```

LeRobot is installed in editable mode from `../lebrobot`, with `core_scripts`
and Feetech support. MuJoCo is installed in the same environment.

Open `robotics.code-workspace` in VS Code to work with both folders.

## Setup records

Installation notes, package snapshots, the verification script, and the prior
MuJoCo warning log are preserved in [archive/setup](archive/setup/).

To rerun the installation check from an activated environment:

```bash
MUJOCO_GL=egl python archive/setup/verify_installation.py
```

Activate `lerobot` first: selecting its Python executable alone does not add
the environment's FFmpeg executable to the shell's search path.

The SO-101 simulation model and leader-to-simulation bridge are still to be built.
