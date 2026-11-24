# Sim-to-real: check camera agreement first

**Known open item / first diagnostic when investigating sim-to-real errors:**
rigorously verify that both simulated camera images match the physical setup.
Applying calibration parameters is not proof that the complete image pipelines
or camera-to-robot registrations agree. Check this before attributing a transfer
failure solely to the policy, demonstration count, or training duration.

## Current camera status

- Native image size: 640×480. Each camera uses its own measured fx, fy, cx, cy.
- Overhead lens and optical pose: ChArUco measurements from branch
  `camera-calibration`, commit `0bb66e5c12d1cda55feff71389e3a3dace46478b`.
  The calibration used the physical follower on its current side. Board offsets
  are trusted: 205 mm forward, 20 mm left of the shoulder-pan axis, 5 mm above
  the table; board yaw -90 degrees in the documented calibration world.
- The calibration's world origin assumed the branch's old follower position.
  Converting to the current simulation adds [0, 0.23206424184417723, 0] metres.
  The optical camera's active position in config is then relative to the stand.
  Do not paste the measured world coordinates directly into that local field.
- Wrist lens: measured. **Wrist camera position and mounting tilt remain nominal
  CAD estimates**, including the lens offset from the PCB. These extrinsics have
  not been measured with hand-eye calibration.
- Expected wrist image orientation: gripper tongs at the bottom of the raw,
  unrotated real feed. Existing CAD roll=0 projects the grasp frame into the
  lower half of the simulated image. This is a coarse orientation reference,
  not a measurement of all three orientation angles or a paired real/sim check.
  A user-supplied monitor photo supports the displayed gripper-at-bottom
  orientation, but does not verify the streamer applies no rotation/cropping,
  or establish quantitative pixel agreement.
- Visible overhead PCB/lens proxies retain their nominal STL assembly placement;
  they are not the definition of the calibrated optical frame. The measured
  optical pose and CAD assembly need physical co-registration verification.

Exact source result JSON and conversion provenance are preserved under
`lerobot-tabletop/mounts/calibration/`. Older provisional lens estimates are
superseded and must not replace these measurements.

## Required verification before claiming transfer fidelity

1. Capture paired real/sim views at matched **actual follower** joint positions
   and gripper openings, at several arm poses and target depths/locations.
2. Place a measured calibration board or other landmarks in a shared frame.
   Compare their projected corners numerically and with image overlays. Include
   image edges, not only the centre; retain held-out views for validation.
3. Confirm camera identity, native resolution/mode, principal points, cropping,
   image rotation/mirroring, and the exact preprocessing seen by the policy.
4. For each real camera independently, use its saved K and D to undistort to
   the **same output K**, e.g. `cv2.undistort(raw, K, D, None, K)`. The simulator
   produces an ideal pinhole view. Cropping, changing K with
   `getOptimalNewCameraMatrix`, resizing or rotating requires a corresponding
   change to the simulated projection and training preprocessing.
5. If alignment varies with wrist pose, investigate wrist-to-camera extrinsics,
   robot calibration/kinematics, mounting flex and synchronization. Same-scale
   STL printing does not prove the lens centre and tilt agree. Use hand-eye
   calibration if CAD placement is insufficient.

The current setup is adequate for proceeding with simulation experiments, but
this checklist remains **open until paired physical validation is performed**.
After camera agreement, investigate timing/latency, action units and limits,
robot dynamics, contact/friction, lighting/exposure, and demonstration coverage.

## Dataset and policy compatibility

This camera update changes the observation distribution. Existing recordings
and frozen robustness bundles preserve their original models and camera views.
They are not silently updated. Re-render appropriate saved simulation states
with a deliberate scene migration, or collect new demonstrations for the new
scene; keep scene/camera versions associated with each training run.
