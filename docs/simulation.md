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
- The overhead stand is fixed to the table. Its upright mating geometry gives
  approximately 56 cm height. The follower is supported by the left base adapter.
- A second stock base adapter occupies the right slot and remains empty. The
  adapter mesh is symmetric across its width, so its opposite-end tabs connect
  to the other side of the stand without mirroring or changing the stock mesh.
- World +X points forward into the task workspace; +Z points upward. Looking
  from the front (+X toward -X), the follower is left (-Y) of the stand.
- Robot-base registration on the adapter is estimated from the footprint,
  accounting for the upstream base mesh origin. It is not measured bolt-hole
  registration. Nominal camera poses are geometric estimates, not calibration.
- The exact hex-nut adapter replaces the old wrist mount; it follows the fixed
  gripper/wrist-roll link rather than the moving jaw.
- New overhead plastic meshes are visual-only: no exact collision model is
  inferred from concave STL geometry. Do not rely on stand contact for planning.
  Wrist collision/payload approximations are described in `wrist.py`.
- The mounted task has one red cube, with fixed cube/box positions, camera poses
  and initial arm targets on every reset. `task.cube_position` and
  `task.box_position` in [mounts/config.json](../lerobot-tabletop/mounts/config.json) set the starting positions in metres.
  The original four-cube randomized task is available with `--scene upstream`.
- Reward is 0 or 1 using the existing cube-centre-in-box test; it is a basic
  evaluation signal, not a verified release/stability detector or a shaped PPO
  reward. ACT behavior cloning trains on observations and expert actions.

## Adjusting the scene

Edit [mounts/config.json](../lerobot-tabletop/mounts/config.json), then restart the launcher. Positions use metres and
quaternions use w,x,y,z. `stand_position` places the stand root; each
`overhead_parts` entry places a stock STL within it. `follower_position` is a
world position, so moving the stand also requires moving this position.

`overhead_camera` places the optical frame, PCB and lens proxies within the
stand. Wrist transforms are passed to `configure_wrist` in `wrist.py`; its
documented configuration keys can override the defaults, including camera roll.
The wrist adapter is physically flipped around its flange screw axis, and
`camera_plate_side: -1` places the camera on the opposite face of the printed
plate. This changes the lens position and viewing direction; `camera_roll_degrees`
only rotates the image around that viewing direction.

Output cameras are `overhead` and `wrist`, still 640x480 RGB at nominal 30 Hz.
Vertical FOV remains 60 degrees overhead and 61 degrees wrist. Lens intrinsics,
distortion, focus, shutter, latency and exposure have NOT been calibrated to the
real cameras. Camera placement and image orientation should be visually checked
before collecting demonstrations, and calibrated before claiming sim-to-real
matching. Rotating the wrist image later is straightforward, but data collected
with the old orientation will need consistent correction or recollection.

## Sources

Stock designs from TheRobotStudio/SO-ARM100 at commit
`141aea7bcac2744d92998e30268c413202a56f96`:

- [Overhead mount guide](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Optional/Overhead_Cam_Mount_32x32_UVC_Module)
- [Hex-nut wrist mount guide](https://github.com/TheRobotStudio/SO-ARM100/tree/141aea7bcac2744d92998e30268c413202a56f96/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module)

[asset provenance](../lerobot-tabletop/mounts/assets/OVERHEAD_SOURCES.json) records source paths and SHA256 hashes for all five
meshes, including the wrist adapter. Upstream Apache-2.0 license is in
[asset license](../lerobot-tabletop/mounts/assets/LICENSE).
