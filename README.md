# SO-101 simulation and demonstration collection

Control a simulated SO-101 follower with a calibrated real leader, then collect
pick-and-place demonstrations for imitation learning. The mounted task uses one
red cube, a black bin, and overhead/wrist cameras. Policy training is next.

## Run

```bash
conda activate lerobot
cd ~/code/enph576/lerobot-tabletop

# Explore the scene; add --leader-port /dev/ttyACM0 for leader control.
python launch_tabletop.py

# Append to the existing local baseline (omit --resume for a new dataset).
python record_demonstrations.py --root recordings/sim-baseline --resume \
  --leader-port /dev/ttyACM0 --episodes 3
```

The recorder opens one four-view window with buttons and keyboard controls:
`s` start, `f` finish, `k` keep, `d` discard, `r` reset, `q` quit. Only overhead and
wrist images are recorded. Do not run two programs against the same leader port.

## Guides

- [Collection, dataset format, validation and replay](docs/collection.md)
- [Mounted scene, configuration, camera assumptions and asset sources](docs/simulation.md)
- [Project goals, hardware and remaining sim-to-real work](docs/project.md)
- [Original upstream tabletop reference](docs/upstream-tabletop.md)
- [Installation history and environment snapshots](archive/setup/installation.md)

## Repository layout

| Path | Purpose |
|---|---|
| `lerobot-tabletop/` | Simulation and three command-line entry points |
| `lerobot-tabletop/recording/` | LeRobot writer, MuJoCo archive, replay and preview |
| `lerobot-tabletop/mounts/` | Mounted scene builder, editable poses and stock STL assets |
| `lerobot-tabletop/tests/` | Hardware-free collection/replay checks |
| `lerobot-tabletop/robotstudio_so101/` | Vendored upstream robot and its license |
| `docs/` | Current project guides |
| `archive/setup/` | Historical installation records |

The local baseline contains 50 episodes / 26,006 frames. Recordings, generated
MJCF, caches and verification images stay outside Git. **Committing this code
does not back up demonstrations**; preserve the full dataset directory separately.

## Environment and checks

LeRobot is installed editable from `../lebrobot` with `core_scripts` and Feetech
support. Activate Conda so FFmpeg is on PATH. `robotics.code-workspace` opens both
checkouts in VS Code. No shell graphics variables need to be configured.

From the repository root:

```bash
python archive/setup/verify_installation.py
cd lerobot-tabletop
python launch_tabletop.py --check
python -m unittest discover -s tests -v
```

Run rendering/encoding checks one at a time on this desktop. The full tests use
temporary synthetic datasets; they do not connect hardware or modify recordings.
