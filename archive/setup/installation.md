# SO-101 simulation environment

Installed and verified on 2026-11-04 following the LeRobot main installation guide:
https://huggingface.co/docs/lerobot/main/en/installation

- Conda environment: `lerobot` (`/home/tonyli2/miniforge3/envs/lerobot`)
- Editable LeRobot checkout: `/home/tonyli2/code/lebrobot`
- LeRobot commit: `9e5ee9859480e425a97a6836bb3fee479539beb5`
- Extras installed: `core_scripts,feetech`
- Python 3.12.16; LeRobot 0.6.2; PyTorch 2.15.0 (CUDA 13.2)
- torchvision 0.29.2; TorchCodec 0.17.0; rustypot 1.13.1
- MuJoCo 3.17.0; Conda FFmpeg 7.1.1
- NVIDIA RTX 3070, 8 GB; driver 595.99.02

## Use

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/code/enph576
MUJOCO_GL=egl python archive/setup/verify_installation.py
```

`MUJOCO_GL=egl` selects offscreen rendering for the verification script. Leave it
unset when using MuJoCo's interactive desktop viewer unless the application
specifically needs EGL.

## Verified

- Shell networking: `curl -I https://conda.anaconda.org` returned HTTP 302.
- `pip check`: no broken requirements.
- Imports: LeRobot, torchvision, PyAV, datasets, pyserial, rustypot, Rerun,
  Foxglove, serial motor bus, SO leader and SO follower modules.
- `lerobot-teleoperate`, `lerobot-record`, `lerobot-replay`, and
  `lerobot-calibrate`: `--help` exited successfully.
- PyTorch CUDA matrix computation on the RTX 3070.
- Generated a five-frame MP4 with FFmpeg and decoded it with TorchCodec.
- MuJoCo cube physics and 320×240 offscreen rendering; OpenGL renderer reported
  `NVIDIA GeForce RTX 3070/PCIe/SSE2`.
- LeRobot checkout remained clean.

The executor sandbox hides NVIDIA device nodes. GPU checks passed when run
outside that sandbox; the installed driver did not need changes.

`evdev` was installed from conda-forge because the pip source build required
GCC, which was absent. FFmpeg was pinned to the guide's compatibility version.

Full installed versions are saved in `requirements-installed.txt` and
`environment-installed.yml`. These are environment snapshots, including local
paths and platform-specific packages.

## Next milestone

Use the real calibrated SO-101 leader to control a MuJoCo SO-101 follower.
No arms were connected, moved, recalibrated, or assigned new servo IDs during
installation. Hardware communications and an SO-101 simulation model have not
yet been verified.

The next implementation needs an SO-101 MuJoCo model, the leader's serial port
and existing calibration file/ID, and an explicit joint/gripper mapping.
The installed LeRobot SO leader defaults to body joint angles in degrees;
MuJoCo joint coordinates use radians. Validate joint signs, zero offsets,
limits, and gripper units before live control. Keep the existing calibration.

Project goals, camera mounts, and later demonstration plans are in
`../../sim_setup.md`. Camera intrinsics and exact modules remain unconfirmed.
