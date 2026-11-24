# Running the calibration on the Mac (bird's-eye camera, 640x480)

Everything runs at **640x480**, the sim's resolution. This camera's 4:3 modes show the middle 75%
of the sensor's width, so lens values from another mode would not match.

On the Mac the camera is opened by **Chrome** (which has camera permission), not by Python.
`browser_capture.py` serves a page that opens the camera at its native 640x480 mode and sends
frames to Python, which runs the same board detection and view selection as `calib.py capture`.
The page warns if the browser would rescale instead of using the native mode.

All commands are run from `rl-so101/` with the environment in `EECE576/.venv-calib`:

```bash
cd ~/Downloads/EECE576/rl-so101
```
```bash
source ../.venv-calib/bin/activate
```

## 0. Before starting

- Check that the 100 mm line on the printed board measures 100 mm. If it does not, measure one
  square and pass that size as `--square-mm` to every command below. The default is 35.
- Mount the board on something flat and stiff.
- Fix the camera in its final position first. The lens values do not depend on where the camera
  is, but the position step does: if the camera moves afterwards, redo step 3.

## 1. Start the capture page

```bash
python lerobot-tabletop/camera_calibration/browser_capture.py --name overhead
```

Open http://localhost:8001 in Chrome and allow the camera. The top bar should read
`Innomaker-U20CAM-1080p-S1 ... 640x480 resize=none` with no red warning. If it shows another
camera, pick the Innomaker from the dropdown.

## 2. Lens values (intrinsics)

With **Intrinsics** selected, hold the board in front of the camera and move it slowly: centre,
all four corners of the image, near and far, tilted about 30-45 degrees each way. Hold it still
for a moment in each pose. A view is saved each time (green flash), up to 30. Get the green
coverage out to the image edges and corners, because that is where this lens distorts most.

```bash
python lerobot-tabletop/camera_calibration/calib.py intrinsics --name overhead
```

Check the printed reprojection error: under about 0.5 px is good, above 1 px means redo the
capture (delete `captures/overhead/` first). The fit keeps k3 = 0 by default because the board
rarely reaches the image corners; add `--free-k3` only if it did (the command warns if the
distortion model folds back before the corners).

## 3. Camera position (extrinsics)

Lay the board flat in front of the arm, text upright as seen from behind the arm, long edge
square to the arm's forward direction, the whole pattern visible and nothing covering it. The
`origin` dot is the bottom-left corner of the pattern: the corner nearest the arm, on the left.

On the page select **Extrinsics**; the status should say `24/24 corners seen`. Once intrinsics
exist, the page also draws live rulers in mm from the origin dot (cyan: forward, toward the arm;
orange: left/right), so a tape measure on the table can be lined up and checked in the feed. Click
**Save extrinsics frames** (it saves 15 frames to `captures/overhead_extrinsics/`, and only
when all 24 corners are visible).

Measure in mm from the base's rotation (shoulder-pan) axis to the origin dot:
- forward (along the arm's forward direction)
- left (positive to the left, negative to the right)
- board thickness (0 for bare paper)

Convert to metres and add 0.0111 to forward (the sim's origin is 11.1 mm behind the pan axis):

```bash
python lerobot-tabletop/camera_calibration/calib.py extrinsics --name overhead --image lerobot-tabletop/camera_calibration/captures/overhead_extrinsics/*.png --board-in-world <forward/1000+0.0111> <left/1000> <thickness/1000> -90
```

Example: forward 120 mm, left 122 mm, bare paper gives `--board-in-world 0.1311 0.122 0 -90`.

Check `results/overhead_extrinsics_check.jpg`. The axes are drawn in OpenCV's board frame, at
the pattern's top-left corner (the far corner on the left, away from the arm, not the origin dot):
the red axis should point to the arm's right and the green one toward the arm. The printed
position spread should be a few mm at most. The printed `overhead_camera` block goes into
`mounts/config.json`.

## Without Chrome

`calib.py capture --device <index>` opens the camera directly (AVFoundation on macOS). It needs
camera permission for the app running Python; `calib.py cameras` shows which index is which.
