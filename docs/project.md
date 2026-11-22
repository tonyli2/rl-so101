# Project context

The goal is SO-101 pick-and-place learning in MuJoCo, followed by sim-to-real
transfer. A calibrated physical SO-101 leader controls the simulated follower
through LeRobot. The physical follower need not be connected for collection.

The current task is a single red cube placed into a black bin, with fixed
starting positions. Both printed mounts and the follower are black. The follower
occupies the left base slot when viewed from the front; the right adapter is
empty. The stock overhead and hex-nut wrist mounts use unmodified STL designs.
See [scene assumptions and sources](simulation.md).

Demonstrations store joint states/targets and two cameras in native LeRobot
format, with MuJoCo states for replay and camera re-rendering. The initial local
baseline has 50 accepted episodes. Next comes imitation learning; ACT, BC with a
Gaussian policy, and later PPO/residual PPO remain possible consumers. No policy
training or RL reward pipeline has been implemented yet.

Real camera intrinsics/extrinsics, distortion, timing, and actuator/physics
matching still need measurement before sim-to-real claims. Matching real cube
and bin geometry to the scene reduces task differences but does not establish
physical equivalence.

The desktop has an i7-10700K, RTX 3070 (8 GB VRAM), and about 16 GB RAM. The voice
application uses about 1.8 GB VRAM. Linux and swap currently live on a rotational
system drive; concurrent simulation/encoding workloads have caused substantial
storage stalls. Run verification jobs sequentially and monitor memory pressure.

The Conda environment is `lerobot`; its editable LeRobot source checkout is
`~/code/lebrobot`. Installation history and package snapshots are retained under
[archive/setup](../archive/setup/).
