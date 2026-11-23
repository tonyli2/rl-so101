# Camera results for sim matching

Date: 2026-11-22. Camera: the overhead (bird's-eye) camera on the Jetson.
Full values for every camera mode are in
[`lerobot-tabletop/camera_calibration/provisional_intrinsics.json`](lerobot-tabletop/camera_calibration/provisional_intrinsics.json).

## Status

| Item | Status |
|---|---|
| Overhead lens distortion | Measured from straight scene edges, about 1 px residual |
| Overhead focal length | Estimated, about +/- 5% (not measured directly) |
| Overhead principal point | Assumed at the image centre |
| Overhead extrinsics (position and angle) | Not measured |
| Wrist camera | Not measured (same model, was unplugged) |

These are provisional. A board calibration with `calib.py` replaces them.

## The camera

- Innomaker U20CAM-1080p-S1, a 32 x 32 mm USB module (USB ID 0c45:6366).
- Fixed-focus wide-angle lens. The closest vendor listing gives 130 degrees, which fits the
  measurements as the diagonal of the full 16:9 frame.
- No focus control, so the lens values do not drift.
- Auto exposure and auto white balance are on.
- MJPEG up to 1920x1080 at 30 fps. The vendor does not publish the sensor model.
- The 4:3 modes (640x480, 800x600, 1280x960) show the middle 75% of the 16:9 frame's width at
  full height. 640x480 is the 1080p frame cropped to 1440x1080 and scaled by 1/2.25.

## Lens values at 640x480 (the sim's resolution)

| Value | Estimate |
|---|---|
| fx = fy | 419.4 px |
| cx, cy | 320, 240 |
| Distortion `[k1, k2, p1, p2, k3]` (OpenCV) | `[-0.3126, 0.1274, 0, 0, -0.0255]` |
| Raw image field of view | 87.0 deg horizontal, 65.5 deg vertical, 107.9 deg diagonal |
| MuJoCo `fovy` after undistorting | 59.6 deg |

The distortion coefficients do not depend on resolution. For other modes, scale fx, fy, cx, cy
(the JSON file lists them).

## What the sim should do

MuJoCo renders an ideal pinhole and cannot reproduce this lens's barrel distortion. Two options:

1. **Undistort the real frames (simplest).** Run each real 640x480 frame through
   `cv2.undistort(frame, K, D)` with the values above. The result matches a MuJoCo camera with
   `fovy` 59.6, so the sim's current 60 is nearly right. The undistorted frame shows about
   74.7 x 59.6 degrees, so the outer part of the raw view is lost.
2. **Distort the sim renders.** Render a wider pinhole image in MuJoCo and warp it with the same
   model so it looks like the raw camera. This keeps the full 87 x 65.5 degree view, at the cost
   of an extra warp in the sim pipeline.

Raw frames compared directly with sim renders will not match under either `fovy`.

## How the values were obtained

- **Distortion:** four straight edges in the camera's view (both edges of a rail, a power strip
  and the table edge) were traced at 1920x1080 and a one-parameter radial model was fitted to
  make them straight. The error dropped from 12 px to 0.9 px. Leaving any one edge out gave
  nearly the same answer. On a separately captured 1280x960 frame, `cv2.undistortPoints` with
  the final OpenCV coefficients straightened each edge to between 0.3 and 1.2 px.
- **Focal length:** two estimates at 1920x1080 scale agree within 1%: 948 px from the 130 degree
  diagonal figure, and 939 px assuming an equidistant wide-angle lens. The adopted value is
  944 px, which is 419.4 px at 640x480.
- **Principal point:** assumed at the centre. The edge fit places the distortion centre within
  about 40 px of it at 1920x1080 scale.

## Overhead extrinsics: how to measure them

The sim needs the camera's `pos` and `xyaxes` in its world frame (`mounts/config.json`,
`overhead_camera`). World frame: +X forward from the arm, +Y to the left seen from behind the
arm, +Z up, table top at z = 0. The origin is on the table, 11.1 mm behind the follower's
shoulder-pan axis.

1. Show or print `lerobot-tabletop/camera_calibration/board.pdf`. For extrinsics the size
   matters, so measure one square with a ruler.
2. Lay the board flat on the table in the overhead camera's view, in front of the follower arm,
   with its text upright as seen from behind the arm and its long edge square to the arm's
   forward direction.
3. Measure two distances from the follower's shoulder-pan axis to the board's `origin` dot:
   `forward` (along the arm's forward direction) and `left` (positive to the left).
4. Run, with distances in metres:
   ```
   python lerobot-tabletop/camera_calibration/calib.py extrinsics --name overhead \
       --device /dev/video0 --square-mm <measured> \
       --board-in-world <forward + 0.0111> <left> 0 -90
   ```
   It prints the `overhead_camera` block for `config.json` and saves a check image with the
   board's axes drawn on it.

`calib.py extrinsics` reads `results/overhead_intrinsics.json`, so run `capture` and `intrinsics`
first (see the README in that folder). That also replaces the provisional lens values. A 5%
focal length error becomes roughly a 5% error in camera height, so the board intrinsics should
come first.

The follower arm's real base position relative to the camera differs from the sim's nominal
layout, so expect `pos` and `xyaxes` to change noticeably from the current values.
