# Camera calibration (real cameras -> sim)

Measures the real cameras so `mounts/config.json` can replace its nominal camera values.
`calib.py` needs only OpenCV (with `aruco`) and NumPy. On the Jetson:

```
PY=~/lerobot-env/bin/python
$PY calib.py selftest          # checks the maths on synthetic images, no camera needed
```

## What the sim needs

| Sim value | Where | Measured by |
|---|---|---|
| `overhead_camera.fovy` | `mounts/config.json` | `intrinsics` |
| `overhead_camera.pos`, `xyaxes` (world frame) | `mounts/config.json` | `extrinsics` |
| `wrist.fovy` | `mounts/config.json` | `intrinsics` |
| wrist camera pose in the gripper frame | `mounts/wrist.py` | not automated yet (needs arm joint readings + forward kinematics) |

The sim renders 640x480. On the Innomaker U20CAM, 1280x960 is exactly 2x that mode with the
same field of view, while the 16:9 modes (1280x720, 1920x1080) see a wider view. So calibrate
at 1280x960 (the default); the tool also writes the values scaled to 640x480.

MuJoCo cameras have no lens distortion. Undistort the real images with the saved
`camera_matrix` and `distortion` before comparing them with sim renders.

## Steps

1. **Print the board.** `$PY calib.py board` writes `board.pdf`. Print it at 100% scale and
   tape it flat to something rigid. Measure the 100 mm line; if it is off, measure one square
   and pass `--square-mm <value>` to every command below.
2. **Intrinsics, once per camera.** Stop anything else using the camera, then:
   ```
   $PY calib.py capture --name overhead --device /dev/video0
   $PY calib.py intrinsics --name overhead
   ```
   Watch the preview at http://localhost:8000 (over SSH: `ssh -L 8000:localhost:8000 ...`).
   Move the board to the centre, every corner of the image, near and far, and tilted each way.
   A view is saved each time the board is held still in a new pose. Aim for an RMS error under
   0.5 px and coverage over 60%. Repeat with `--name wrist` for the wrist camera.
3. **Overhead extrinsics.** Lay the board flat on the table in view of the overhead camera,
   text upright as seen from behind the arm, long edge square to the arm's forward direction.
   Measure where the board's `origin` dot is in the sim world frame, then:
   ```
   $PY calib.py extrinsics --name overhead --device /dev/video0 --board-in-world X Y 0 YAW
   ```
   It prints an `overhead_camera` block (`pos`, `xyaxes`, `fovy`) for `mounts/config.json`, and
   saves `results/overhead_extrinsics_check.jpg` with the board axes drawn for a visual check.

## Frames

- **Sim world:** +X forward into the workspace, +Y left (seen from behind the arm), +Z up, table
  top at z = 0. With the mounted scene's `follower_position`, the world origin is on the table
  11.1 mm behind the shoulder-pan axis. So a point `d` metres in front of the pan axis is at
  world x = `d + 0.0111`.
- **Board:** origin at the printed `origin` dot (bottom-left corner), +x along the long edge,
  +y along the short edge, +z up out of the table. `YAW` is the angle from world +X to board +x,
  in degrees, counter-clockwise seen from above. With the board's long edge running left to
  right as seen from behind the arm, board +x points along world -Y, so `YAW` is -90.
- **Board thickness:** if the board sits on a backing, pass its thickness as Z.

Results are written to `results/` and raw views to `captures/` (both ignored by git).
