# Camera results for sim matching

## Board calibration, 2026-11-23 (supersedes the provisional values below)

Overhead (bird's-eye) camera, Innomaker U20CAM-1080p-S1 at **640x480**, calibrated with the
ChArUco board on a Mac (`camera_calibration/browser_capture.py` + `calib.py`, steps in
`camera_calibration/RUN_ON_MAC.md`). Files: `lerobot-tabletop/camera_calibration/results/`.

### Lens (intrinsics)

| Value | Result |
|---|---|
| fx, fy | 549.8, 550.1 px |
| cx, cy | 334.5, 269.4 px (14 px right of and 29 px below the image centre) |
| Distortion `[k1, k2, p1, p2, k3]` | `[-0.437, 0.1936, -0.0009, -0.0001, 0]` (k3 held at 0) |
| RMS reprojection error | 0.295 px over 61 views |
| MuJoCo `fovy` after undistorting | 47.15 deg |

- Board coverage reached 67% of the image; each image corner is only about 10% covered. With k3
  free the distortion curve folded back before the corners, so `calib.py` now keeps k3 = 0 by
  default (0.295 vs 0.289 px, well behaved to the corners). `--free-k3` restores it.
- The provisional estimate further down (fx 419.4, fovy 59.6) was wrong by about 30%.

### Pose (extrinsics), sim world frame

Board flat in front of the arm, origin dot 205 mm forward of and 20 mm left of the shoulder-pan
axis, board surface 5 mm above the table (`--board-in-world 0.2161 0.020 0.005 -90`).
15 frames, 0.18 px reprojection, spread under 0.3 mm.

| | Measured | Sim `overhead_camera` (nominal) |
|---|---|---|
| pos (m) | [0.0536, -0.0460, 0.5356] | [0.0331, 0.0, 0.5439] |
| Tilt from straight down | 24.5 deg | 24.3 deg |
| Yaw | 2.8 deg | 0 |
| fovy | 47.15 deg | 60 deg |

Height and tilt (from the board alone) match the designed mount within 1 cm and 0.2 deg. The
real camera sits 46 mm to the right, 21 mm further forward, and sees a much narrower view.

Tape checks at the very edges of the picture agreed within about 5% (720 mm across the image
at 50 mm forward vs 755 mm computed; 147 mm from the origin dot to the bottom edge vs 140 mm),
consistent with the less certain lens model at the image edges.

The measured camera is in `mounts/config.json` as `overhead_camera_measured`.
`environment.py` still uses `overhead_camera`, so existing demos and policies are unaffected.
Real frames must be undistorted with the lens values before comparing them with sim renders.

Not done: the wrist camera.

---

# Provisional results, 2026-11-22 (superseded)

Camera: the overhead (bird's-eye) camera on the Jetson.
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

These were provisional and are replaced by the board calibration above.

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
