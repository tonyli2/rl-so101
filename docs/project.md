# Project context

**First sim-to-real check:** [rigorously validate both camera pipelines and
physical alignment](sim2real.md). Wrist extrinsics remain CAD estimates; paired
real/sim validation is still open.

The goal is SO-101 pick-and-place learning in MuJoCo, followed by sim-to-real
transfer. A calibrated physical SO-101 leader controls the simulated follower
through LeRobot. The physical follower need not be connected for collection.

The current task is a single red PLA cube placed into a grey PLA bin, with fixed
starting positions. Both printed mounts and the follower are black. The follower
occupies the right base slot when viewed from the front; the left adapter is
empty. The stock overhead and hex-nut wrist mounts use unmodified STL designs.
See [scene assumptions and sources](simulation.md).

Demonstrations store joint states/targets and two cameras in native LeRobot
format, with MuJoCo states for replay and camera re-rendering. The initial local
baseline has 50 accepted episodes, now archived at
`lerobot-tabletop/recordings/archive/sim-baseline` for historical reproduction,
not training against the updated scene. Next comes imitation learning; ACT, BC with a
Gaussian policy, and later PPO/residual PPO remain possible consumers. Local ACT
training and collision-aware evaluation are implemented; the RL training/reward
pipeline remains future work.

Measured camera intrinsics and overhead pose are applied, with per-camera
distortion values retained for real-image preprocessing. Wrist extrinsics, paired
image validation, timing, and actuator/physics matching remain open before
sim-to-real claims. Matching real cube
and bin geometry to the scene reduces task differences but does not establish
physical equivalence.

The desktop has an i7-10700K, RTX 3070 (8 GB VRAM), and about 16 GB RAM. The voice
application uses about 1.8 GB VRAM. Linux and swap currently live on a rotational
system drive; concurrent simulation/encoding workloads have caused substantial
storage stalls. Run verification jobs sequentially and monitor memory pressure.

The Conda environment is `lerobot`; its editable LeRobot source checkout is
`~/code/lebrobot`. Installation history and package snapshots are retained under
[archive/setup](../archive/setup/).
