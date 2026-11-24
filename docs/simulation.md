# Stock printed camera mounts

This variant reuses the upstream tabletop environment's physics, control,
reset, reward, and rendering implementation. The upstream world constructor has
one optional cube-list parameter, and reward normalization uses that list's size.
Original scene XML and robot XML remain unchanged. [mounts/environment.py](../lerobot-tabletop/mounts/environment.py) generates scene/arm variants
under ignored `generated/` using the original XML plus the stock printed meshes.

## Run

From `lerobot-tabletop` in the activated `lerobot` Conda environment:

```bash
python launch_tabletop.py
python launch_tabletop.py --leader-port /dev/ttyACM0
python launch_tabletop.py --check
```

The mounted scene is now the default. `--scene upstream` selects the original.
The launcher automatically selects desktop rendering for a window and offscreen
rendering for `--check`; no shell graphics configuration is needed.
Checks write overhead, wrist and overview PNGs to `verification/mounted/`.

## Assembly and assumptions

- Both printed mounts use the servo housing's black colour and highlight settings.
- The follower's visible parts use that same material, giving the mounted setup
  a consistent black finish. The original upstream scene retains its colours.
- Meshes are the unmodified stock designs at millimetre-to-metre scale.
- Task objects use a red PLA cube and medium-grey PLA bin with subtle layer-band
  appearance (`textures/pla_layers.png`). The cube retains sharp 30 mm edges and
  has measured mass 10.21 g; MuJoCo derives its inertia from the box geometry.
  Bin geometry is unchanged. The finish is visual; collision shapes and friction
  parameters remain unchanged. Colours and cube mass are under `task` in config.
- The overhead stand is fixed to the table. Its upright mating geometry gives
  approximately 56 cm height. The follower is supported by the front-view right
  base adapter (rear-view left).
- A second stock base adapter occupies the front-view left slot and remains empty.
  The adapter mesh is symmetric across its width, so its opposite-end tabs connect
  to the other side of the stand without mirroring or changing the stock mesh.
- World +X points forward into the task workspace; +Z points upward. Looking
  from the front (+X toward -X), the follower is right (+Y) of the stand.
- Robot-base registration on the adapter is estimated from the footprint,
  accounting for the upstream base mesh origin. It is not measured bolt-hole
  registration. Wrist camera extrinsics remain geometric estimates; measured
  overhead extrinsics require the frame conversion documented in sim2real.md.
- The exact hex-nut adapter replaces the old wrist mount; it follows the fixed
  gripper/wrist-roll link rather than the moving jaw.
- Stand STL meshes remain visual-only, with one separate box for the straight
  upright (`overhead_upright_collision`). It follows the STL core's 25.4 × 36.625 mm
  cross-section from 8.7 to 511.3 mm above the tabletop. Flat bases, adapters,
  and the angled camera head remain non-colliding. The box is hidden in normal
  views (group 3); enable that group in the viewer to inspect it.
- The recorder rejects arm–upright contact at the initial pose and each physics
  substep, in addition to its existing self-collision rule. No PPO reward or
  environment contact flag is added. Historical frozen benchmark models keep
  their original geometry and self-contact metric.
  Wrist collision/payload approximations are described in `wrist.py`.
- The mounted task has one red cube, with fixed cube/box positions, camera poses
  and initial arm targets on every reset. `task.cube_position` and
  `task.box_position` in [mounts/config.json](../lerobot-tabletop/mounts/config.json) set the starting positions in metres.
  The original four-cube randomized task is available with `--scene upstream`.
- Reward is 0 or 1 using the existing cube-centre-in-box test; it is a basic
  evaluation signal, not a verified release/stability detector or a shaped PPO
  reward. ACT behavior cloning trains on observations and expert actions.

## White paper enclosure

The measured default object centres in the pole-front tabletop frame are cube
(-8.5, 27.5) cm and bin (7.3, 24.9) cm, with edges aligned to the axes.
These come from the cube's near-right corner (-7, 26) cm and the bin's outside
near-left corner (2.4, 20) cm, using the 3 cm cube and 9.8 cm outer bin width.
Their MuJoCo world positions are respectively (0.2550305, 0.20103212, 0.015) m
and (0.2290305, 0.04303212, 0) m; the bin body origin is at its bottom.

Three simple 2 mm thick, finite box panels represent the paper curtains. Their
**interior faces**, measured from the midpoint of the upright's front face at
tabletop height, are:

| Panel | Measurement coordinates | Height |
| --- | --- | --- |
| Left | X=-0.53 m; Y=0..0.455 m | Z=0..0.41 m |
| Right | X=+0.53 m; Y=0..0.455 m | Z=0..0.41 m |
| Front/far | Y=0.455 m; X=-0.53..+0.53 m | Z=0..0.41 m |

Measurement +X points toward the overhead image's right (world -Y), +Y points
forward into the workspace (world +X), and +Z is up. This is a physical tabletop
frame, not a pixel coordinate system. The origin is derived from the pole's
front face: currently world [-0.0199695, 0.11603212, 0] m. In this assembly:
`world = origin + [measurement_y, -measurement_x, measurement_z]`.

Paper is matte white. No seams, overlaps, wrinkles, or paper dynamics are modeled.
The existing supporting table is whitened and its footprint is enlarged only as
needed to contain the measured enclosure (currently 0.80 x 1.06 m); its surface
height stays Z=0. The table's outer edges are a support-model assumption, not
additional physical measurements. Robot mounting elevation is unchanged.

Curtains are in render group 5. Recorded/policy overhead and wrist views include
fully opaque curtains; the inspection window and orbit views use 15% opacity.
Only the renderer's copied scene changes: inspection cannot change recorded
images, the shared model, or collision detection. Replay/re-rendering and cube
visibility checks use the same opaque sensor rendering.

The recorder rejects arm contact with left/right/front curtains at Start or any
physics substep, including transient contacts between camera frames. Such an
attempt cannot be saved, and the rejection message identifies the curtain.
The single upright check and self-contact check remain active. No PPO reward or
RL contact flag is introduced. Cube–paper support contact is allowed.

## Adjusting the scene

Edit [mounts/config.json](../lerobot-tabletop/mounts/config.json), then restart the launcher. Positions use metres and
quaternions use w,x,y,z. `stand_position` places the stand root; each
`overhead_parts` entry places a stock STL within it. `follower_position` is a
world position, so moving the stand also requires moving this position.

`overhead_camera` places the measured optical frame within the stand; PCB/lens
visual proxy poses remain nominal CAD positions. Wrist transforms are passed to `configure_wrist` in `wrist.py`; its
documented configuration keys can override the defaults, including camera roll.
The wrist adapter is physically flipped around its flange screw axis, and
`camera_plate_side: -1` places the camera on the opposite face of the printed
plate. This changes the lens position and viewing direction; `camera_roll_degrees`
only rotates the image around that viewing direction.

Output cameras are `overhead` and `wrist`, still 640x480 RGB at nominal 30 Hz.
Both use the measured full intrinsic matrices (including principal-point offsets)
from `intrinsics_640x480.camera_matrix`; those values take precedence over the
readable `fovy` metadata in config. The projection is applied through native MuJoCo `focalpixel`/`principalpixel`, with vertical FOV equivalents
47.1455 degrees overhead and 47.4988 degrees wrist. The overhead optical pose is
converted from the calibration's arm-reference world into this scene and stored
stand-relative in config. Wrist optical position/tilt remain CAD estimates;
roll=0 retains the expected lower-image gripper position.

Use each camera's own distortion parameters to undistort real frames while
preserving its output K. No per-frame image warping is added to sim. Visible
PCB/lens proxies remain nominal assembly geometry. Numerical projection checks
do not establish physical image agreement: **see [the first sim-to-real
check](sim2real.md)** before transfer debugging or readiness claims.

## Sources

Stock designs from TheRobotStudio/SO-ARM100 at commit
`141aea7bcac2744d92998e30268c413202a56f96`:

- [Overhead mount guide](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Optional/Overhead_Cam_Mount_32x32_UVC_Module)
- [Hex-nut wrist mount guide](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module)

[asset provenance](../lerobot-tabletop/mounts/assets/OVERHEAD_SOURCES.json) records source paths and SHA256 hashes for all five
meshes, including the wrist adapter. Upstream Apache-2.0 license is in
[asset license](../lerobot-tabletop/mounts/assets/LICENSE).
