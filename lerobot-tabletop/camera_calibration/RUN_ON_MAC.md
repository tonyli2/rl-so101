# Running the calibration on the Mac (bird's-eye camera, 640x480)

Everything runs at **640x480**, the sim's resolution. This camera's 4:3 modes show the middle 75%
of the sensor's width, so lens values from another mode would not match. `calib.py` refuses to
run if the camera does not give exactly 640x480.

All commands are run from `rl-so101/` with the environment in `EECE576/.venv-calib`:

```bash
cd ~/Downloads/EECE576/rl-so101
```
```bash
source ../.venv-calib/bin/activate
```

## 0. Before starting

- Close the browser tab with the camera page (http://localhost:8000). A camera can only be open in one program.
- The first time, macOS asks to allow camera access for the terminal app. Allow it.
- Check that the 100 mm line on the printed board measures 100 mm. If it does not, measure one
  square and pass that size as `--square-mm` in every step below. The default is 35.
- Mount the board on something flat and stiff.

## 1. Find the camera index

```bash
python lerobot-tabletop/camera_calibration/calib.py cameras
```

It saves one frame per index to `lerobot-tabletop/camera_calibration/captures/_which_camera/`.
Open them and note the index that shows the table from above (not the FaceTime camera). The steps below use `1`; replace it with that index.

## 2. Lens values (intrinsics)

```bash
python lerobot-tabletop/camera_calibration/calib.py capture --name overhead --device 1
```

Open http://localhost:8001 to watch. Hold the board in front of the camera and move it slowly:
centre, all four corners of the image, near and far, tilted about 30-45 degrees each way. Hold
it still for a moment in each pose. A view is saved automatically each time (30 views). Try to
make the green coverage reach the image edges and corners, because that is where this lens
distorts most.

```bash
python lerobot-tabletop/camera_calibration/calib.py intrinsics --name overhead
```

Check the printed reprojection error: under about 0.5 px is good, above 1 px means redo the
capture. This is a very wide lens, so if the error is high, try adding `--rational`.

## 3. Camera position (extrinsics)

Lay the board flat in front of the arm, text upright as seen from behind the arm, long edge
square to the arm's forward direction, the whole pattern visible and nothing covering it. The
`origin` dot is the bottom-left corner of the pattern: the corner nearest the arm, on the left.

Measure in mm from the base's rotation (shoulder-pan) axis to the origin dot:
- forward (along the arm's forward direction)
- left (positive to the left, negative to the right)
- board thickness (0 for bare paper)

Convert to metres and add 0.0111 to forward (the sim's origin is 11.1 mm behind the pan axis):

```bash
python lerobot-tabletop/camera_calibration/calib.py extrinsics --name overhead --device 1 --board-in-world <forward/1000+0.0111> <left/1000> <thickness/1000> -90
```

Example: forward 120 mm, left 122 mm, bare paper gives `--board-in-world 0.1311 0.122 0 -90`.

Check `results/overhead_extrinsics_check.jpg`. The axes are drawn in OpenCV's board frame, at
the pattern's top-left corner (the far corner on the left, away from the arm, not the origin dot):
the red axis should point to the arm's right and the green one toward the arm. The printed position spread should be a few mm
at most. The printed `overhead_camera` block goes into `mounts/config.json`.
