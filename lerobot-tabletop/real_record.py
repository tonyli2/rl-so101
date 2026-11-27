#!/usr/bin/env python3
"""Record real leader/follower pick-and-place demonstrations from a browser page.

    python lerobot-tabletop/real_record.py            # then open http://localhost:8002

Cameras are opened by the browser page (Chrome already has macOS camera permission; Python
launched from Claude or a script often cannot get it). The page sends every frame from each camera
at its native 640x480 mode (JPEG quality 0.95) to this server; the control loop uses the newest
frame of each camera, like LeRobot's read_latest. Keep the page's tab visible while recording:
Chrome slows hidden tabs, and frames older than 0.25 s stop the recording.

The page walks through setup (which camera is which, which serial port is the leader), then
runs teleoperation continuously. Recording happens only between Start and Save, so resets are
never recorded. Each episode has a planned cube placement: a small offset and rotation from the
cube's home spot, shown as numbers and as an outline on the overhead camera. The plan covers
+-3 cm in X and Y and +-20 degrees evenly over 50 episodes (5x5 grid, each spot twice with
different rotations, fixed shuffled order). The bin stays where it is.

Output: a native LeRobot dataset (two 640x480 cameras "overhead" and "wrist", 30 Hz, follower
joint state, leader joint action, degrees with gripper 0-100), plus side files next to it:
<root>_plan.json (placements), <root>_episodes.jsonl (placement used per saved episode),
<root>_rest.json (rest pose). Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIBRATION_DIR = HERE / "calibration"
MOUNTS = HERE / "mounts"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
W, H, FPS = 640, 480, 30
CUBE = 0.030  # m
REST_TOL_DEG = 5.0
REST_TOL_GRIPPER = 10.0
RESUME_RAMP_S = 1.5
MARK_TOL_CM = 1.5  # placement only needs to be close: the actual position is logged at Start


# ------------------------------------------------------------------ placement plan
def make_plan(n: int, seed: int, span_cm: float = 3.0, yaw_deg: float = 20.0) -> list[dict]:
    """Even coverage: 5x5 grid over +-span, every spot twice with different yaw, shuffled."""
    steps = np.linspace(-span_cm, span_cm, 5)
    yaws = np.linspace(-yaw_deg, yaw_deg, 5)
    rng = random.Random(seed)
    spots = [(float(dx), float(dy)) for dx in steps for dy in steps]
    plan = []
    for rep in range(math.ceil(n / len(spots))):
        order = spots[:]
        rng.shuffle(order)
        for k, (dx, dy) in enumerate(order):
            # A Latin-square style yaw assignment so each spot gets two different rotations.
            i, j = int(np.argmin(abs(steps - dx))), int(np.argmin(abs(steps - dy)))
            yaw = float(yaws[(i + j + 2 * rep) % 5])
            plan.append({"forward_cm": round(dx, 2), "left_cm": round(dy, 2), "yaw_deg": round(yaw, 1)})
    return plan[:n]


def describe(t: dict) -> str:
    def lin(v, pos, neg):
        if abs(v) < 0.05:
            return None
        return f"{abs(v):.1f} cm {pos if v > 0 else neg}"

    parts = [lin(t["forward_cm"], "forward (away from the arm)", "back (toward the arm)"),
             lin(t["left_cm"], "left", "right")]
    if abs(t["yaw_deg"]) >= 0.5:
        parts.append(f"rotated {abs(t['yaw_deg']):.0f} deg {'counter-clockwise' if t['yaw_deg'] > 0 else 'clockwise'} (seen from above)")
    parts = [p for p in parts if p]
    return "at the home spot, not rotated" if not parts else ", ".join(parts)


# ------------------------------------------------------------------ overhead geometry
class Overhead:
    """Project table points into the raw overhead image with the measured camera model."""

    def __init__(self):
        cfg = json.loads((MOUNTS / "config.json").read_text())
        cam = cfg["overhead_camera"]
        intr = cam.get("intrinsics_640x480") or json.loads((MOUNTS / "calibration" / "overhead_intrinsics.json").read_text())
        self.K = np.array(intr["camera_matrix"], float)
        self.D = np.array(intr["distortion"], float)
        self.C = np.array(cfg["stand_position"], float) + np.array(cam["pos"], float)
        xa, ya = np.array(cam["xyaxes"][:3], float), np.array(cam["xyaxes"][3:], float)
        xa, ya = xa / np.linalg.norm(xa), ya / np.linalg.norm(ya)
        za = np.cross(xa, ya)
        # MuJoCo camera: +x right, +y up, looks along -z. OpenCV: +x right, +y down, looks along +z.
        self.R = np.stack([xa, -ya, -za])  # rows: world -> OpenCV camera
        self.home = np.array(cfg["task"]["cube_position"][:2], float)

    def project(self, pts):
        pc = (np.asarray(pts, float) - self.C) @ self.R.T
        uv = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        return uv, pc[:, 2] > 0

    def cube_corners(self, target, z):
        cx = self.home[0] + target["forward_cm"] / 100
        cy = self.home[1] + target["left_cm"] / 100
        a = math.radians(target["yaw_deg"])
        h = CUBE / 2
        loc = [(-h, -h), (h, -h), (h, h), (-h, h)]
        return [[cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a), z] for x, y in loc]

    def table_point(self, u, v, z=CUBE / 2):
        """Pixel -> world XY on the horizontal plane at height z."""
        n = cv2.undistortPoints(np.array([[[u, v]]], np.float32), self.K, self.D).reshape(2)
        ray = self.R.T @ np.array([n[0], n[1], 1.0])
        s = (z - self.C[2]) / ray[2]
        return (self.C + s * ray)[:2]


def find_red_cube(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return tuple(cents[k]) if stats[k, cv2.CC_STAT_AREA] >= 60 else None


# ------------------------------------------------------------------ app
class App:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root).resolve()
        self.side = lambda suffix: self.root.parent / f"{self.root.name}_{suffix}"
        self.lock = threading.RLock()
        self.state = "setup"
        self.message = "Pick the cameras and find the leader arm."
        self.error = ""
        # setup
        self.cam = {"overhead": None, "wrist": None}  # browser mode: (rgb, receive time, count)
        self.cam_thumbs: dict[int, bytes] = {}  # opencv mode: camera index -> JPEG thumbnail
        self.cam_role = {"overhead": a.overhead, "wrist": a.wrist}
        self.read_errors = 0
        self.stale_frames = 0
        self.paused = False
        self.hold = None  # follower joints held while paused
        self.ramp = None  # (start joints, start time) while easing back to the leader after a pause
        self.ports = sorted(glob.glob("/dev/tty.usbmodem*"))
        self.port_motion = {p: 0.0 for p in self.ports}
        self.leader_port = a.leader_port
        self.follower_port = a.follower_port
        self._probe_stop = threading.Event()
        self._probe_done = threading.Event()
        self.loop_done = threading.Event()
        # run
        self.robot = self.leader = self.dataset = None
        self.jpg = {"overhead": None, "wrist": None}
        self.joints = {}
        self.frames = 0
        self.rec_started = 0.0
        self.saving = False
        self.overhead = Overhead()
        plan_path = self.side("plan.json")
        if plan_path.exists():
            self.plan = json.loads(plan_path.read_text())["placements"]
        else:
            self.plan = make_plan(a.episodes, a.seed)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps({"seed": a.seed, "span_cm": 3.0, "yaw_deg": 20.0,
                                             "home_world_xy_m": self.overhead.home.tolist(),
                                             "placements": self.plan}, indent=1))
        rest = self.side("rest.json")
        self.rest = json.loads(rest.read_text()) if rest.exists() else None
        self.cube_info = ""
        self.cube_xy = None  # latest detected cube centre, world XY (m)
        self.start_cube = None
        home = self.side("home.json")
        if home.exists():  # the physical mark, as seen by the camera: offsets are measured from here
            self.overhead.home = np.array(json.loads(home.read_text())["home_world_xy_m"], float)

    # ---------------- setup: cameras and ports
    def probe_cameras(self):
        thumbs = {}
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            frame, t0 = None, time.time()
            while cap.isOpened() and time.time() - t0 < 1.5:
                ok, f = cap.read()
                frame = f if ok else frame
            cap.release()
            if frame is not None:
                cv2.putText(frame, f"camera {i} ({frame.shape[1]}x{frame.shape[0]})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                thumbs[i] = cv2.imencode(".jpg", cv2.resize(frame, (320, 240)))[1].tobytes()
        with self.lock:
            self.cam_thumbs = thumbs
            self.error = "" if thumbs else (
                "No camera opened. Start this script from the macOS Terminal app (not from Claude) and click Allow "
                "when macOS asks about the camera, or turn on Terminal in System Settings > Privacy & Security > "
                "Camera and restart it. (--cameras browser uses Chrome instead.)")
        return bool(thumbs)

    def probe_cameras_until_ok(self, seconds=90):
        """macOS shows its camera dialog asynchronously; keep trying while the user answers it."""
        t0 = time.time()
        while time.time() - t0 < seconds and self.state == "setup":
            if self.probe_cameras():
                return
            time.sleep(2)

    def put_frame(self, name, jpeg):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (H, W):
            return f"frame must be {W}x{H}"
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.lock:
            prev = self.cam.get(name)
            self.cam[name] = (rgb, time.time(), (prev[2] + 1) if prev else 1)
        return "ok"

    def latest(self, name):
        with self.lock:
            c = self.cam.get(name)
        return (None, 1e9) if c is None else (c[0], time.time() - c[1])

    def probe_ports(self):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors import SerialMotorsBus as FeetechMotorsBus

        buses = {}
        for p in self.ports:
            try:
                b = FeetechMotorsBus(p, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS)})
                b.connect(handshake=False)
                buses[p] = b
            except Exception as e:  # noqa: BLE001
                self.error = f"Could not open {p}: {e}"
        # Each arm stores its calibration offsets in its motors: match them to the calibration files.
        cal = {n: json.loads((CALIBRATION_DIR / f"{n}.json").read_text()) for n in ("leader", "follower")}
        found = {}
        for p, b in buses.items():
            try:
                h = b.sync_read("Homing_Offset", normalize=False)
                for n, c in cal.items():
                    if all(int(h[j]) == int(c[j]["homing_offset"]) for j in JOINTS):
                        found[n] = p
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            if "leader" in found and "follower" in found and not (self.leader_port or self.follower_port):
                self.leader_port, self.follower_port = found["leader"], found["follower"]
                self.message = "Found the leader and follower from their calibration. Check the cameras, then Connect."
            elif len(buses) == 2 and len(found) < 2:
                self.error = ("An arm's stored calibration does not match calibration/leader.json or follower.json: "
                              "pick the leader by wiggling it, and expect Connect to refuse until it is recalibrated.")
        hist = {p: [] for p in buses}
        while not self._probe_stop.is_set():
            for p, b in buses.items():
                try:
                    v = b.sync_read("Present_Position", normalize=False)
                    hist[p].append((time.time(), np.array([v[n] for n in JOINTS], float)))
                except Exception:  # noqa: BLE001
                    pass
                hist[p] = [h for h in hist[p] if h[0] > time.time() - 3]
                if len(hist[p]) > 2:
                    arr = np.stack([h[1] for h in hist[p]])
                    self.port_motion[p] = float((arr.max(0) - arr.min(0))[:5].max())
            time.sleep(0.1)
        for b in buses.values():
            try:
                b.disconnect(disable_torque=False)
            except Exception:  # noqa: BLE001
                pass
        self._probe_done.set()

    def connect(self):
        from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
        from lerobot.utils.feature_utils import combine_feature_dicts

        # The port scan must have closed its own handles before the arms are opened for real:
        # closing a scan handle after the real connection is made can break that connection.
        self._probe_stop.set()
        if not self._probe_done.wait(timeout=10):
            raise RuntimeError("The arm port scan did not stop; restart the recorder.")
        time.sleep(0.3)
        cams = {}
        if self.a.cameras == "opencv":
            from lerobot.cameras.opencv import OpenCVCameraConfig

            # Both cameras are read by LeRobot's own background threads; each control tick takes the
            # newest frame of each (read_latest), so the two views are at most one frame apart.
            cams = {role: OpenCVCameraConfig(index_or_path=idx, fps=FPS, width=W, height=H, backend=cv2.CAP_AVFOUNDATION)
                    for role, idx in self.cam_role.items()}
        else:
            for name in ("overhead", "wrist"):
                if self.latest(name)[1] > 1.0:
                    raise RuntimeError(f"No live frames from the {name} camera: assign it on the page and keep the tab open.")
        self.leader = SOLeader(SOLeaderTeleopConfig(port=self.leader_port, id="leader", calibration_dir=CALIBRATION_DIR,
                                                    num_read_retries=5))
        self.robot = SOFollower(SOFollowerRobotConfig(port=self.follower_port, id="follower", cameras=cams,
                                                      calibration_dir=CALIBRATION_DIR, num_read_retries=5))
        self.leader.connect(calibrate=False)
        if not self.leader.is_calibrated:
            raise RuntimeError("The leader's motors do not match calibration/leader.json. Recalibrate it with "
                               "lerobot-calibrate before recording.")
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            raise RuntimeError("The follower's motors do not match calibration/follower.json. Recalibrate "
                               "(and re-measure follower_sim_offsets.json) before recording.")
        self.tp, self.rp, self.op = make_default_processors()
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(self.tp, create_initial_features(action=self.robot.action_features), use_videos=True),
            aggregate_pipeline_dataset_features(self.op, create_initial_features(observation={
                "overhead": (H, W, 3), "wrist": (H, W, 3), **self.robot.observation_features}), use_videos=True),
        )
        if (self.root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(self.a.repo_id, root=self.root, image_writer_threads=8)
        else:
            self.dataset = LeRobotDataset.create(self.a.repo_id, FPS, root=self.root, robot_type=self.robot.name,
                                                 features=features, use_videos=True, image_writer_threads=8)
        threading.Thread(target=self.control_loop, daemon=True).start()
        with self.lock:
            self.state = "ready"
            self.message = "Connected. The follower now copies the leader."

    # ---------------- control loop (always running after connect)
    def control_loop(self):
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame

        period = 1.0 / FPS
        nxt = time.perf_counter()
        last_ui = last_snap = 0.0
        while self.state != "stopped":
            try:
                obs = self.robot.get_observation()
                fresh = True
                if self.a.cameras == "browser":
                    ages = {}
                    for name in ("overhead", "wrist"):
                        img, ages[name] = self.latest(name)
                        obs[name] = img if img is not None else np.zeros((H, W, 3), np.uint8)
                    fresh = max(ages.values()) <= self.a.max_frame_age
                obs_p = self.op(obs)
                act = self.leader.get_action()
                act_p = self.tp((act, obs))
                if self.paused:
                    # Hold the follower where it stopped; the leader can be put down meanwhile.
                    if self.hold is None:
                        self.hold = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}
                    self.robot.send_action(dict(self.hold))
                elif self.ramp is not None:
                    # Ease from the held pose to wherever the leader is now, instead of jumping.
                    start, t0 = self.ramp
                    k = min(1.0, (time.time() - t0) / RESUME_RAMP_S)
                    target = self.rp((act_p, obs))
                    self.robot.send_action({key: start[key] + k * (float(target[key]) - start[key]) for key in start})
                    if k >= 1.0:
                        self.ramp = None
                else:
                    self.robot.send_action(self.rp((act_p, obs)))
                with self.lock:
                    if self.state == "recording" and not fresh:
                        self.stale_frames += 1
                        if self.stale_frames >= 3:
                            self.state = "review"
                            self.message = ("Camera frames stopped arriving (is the page's tab hidden or the browser "
                                            "busy?). Discard this episode and try again.")
                    if self.state == "recording" and fresh:
                        self.stale_frames = 0
                        frame = {**build_dataset_frame(self.dataset.features, obs_p, prefix=OBS_STR),
                                 **build_dataset_frame(self.dataset.features, act_p, prefix=ACTION),
                                 "task": self.a.task}
                        self.dataset.add_frame(frame)
                        self.frames += 1
                        if time.time() - self.rec_started > self.a.max_episode_s:
                            self.state = "review"
                            self.message = f"Stopped at {self.a.max_episode_s:.0f} s. Save or Discard."
                    self.joints = {j: float(obs[f"{j}.pos"]) for j in JOINTS}
                    self.read_errors = 0
                    if self.error.startswith("Arm"):
                        self.error = ""
                now = time.perf_counter()
                if now - last_ui > 1 / 15:
                    last_ui = now
                    self.update_ui(obs)
                if self.a.snapshot_every > 0 and now - last_snap >= self.a.snapshot_every:
                    last_snap = now
                    self.snapshot(obs)
            except Exception as e:  # noqa: BLE001
                # Feetech buses occasionally time out on a read; skip this tick and carry on. Only
                # repeated failures (a cable or power problem) are reported and stop a recording.
                self.read_errors += 1
                print(f"control tick failed ({self.read_errors} in a row): {e}", flush=True)
                if self.read_errors >= 10:
                    with self.lock:
                        self.error = f"Arm/camera communication keeps failing: {e}"
                        if self.state == "recording":
                            self.state = "review"
                            self.message = "Communication failed during recording. Discard this episode."
                    time.sleep(0.1)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 0.5:
                nxt = time.perf_counter()
        self.loop_done.set()

    def snapshot(self, obs):
        """Lossless raw frames through the whole session (resets too), for sim/real image comparison.

        Saved as PNG in <root>_frames/<camera>/, named with the time, the phase (rec/reset) and the
        episode index. These are the raw camera images: not undistorted, not resized."""
        phase = "rec" if self.state in ("recording", "review") else "reset"
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        imgs = {k: np.ascontiguousarray(obs[k]).copy() for k in ("overhead", "wrist")}
        ep = self.episode_index()

        def write():
            for cam, img in imgs.items():
                d = self.side("frames") / cam
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stamp}_{phase}_ep{ep:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        threading.Thread(target=write, daemon=True).start()

    def update_ui(self, obs):
        ov = np.ascontiguousarray(obs["overhead"]).copy()
        wr = np.ascontiguousarray(obs["wrist"]).copy()
        t = self.target()
        info = ""
        if t is not None:
            o = self.overhead
            home = {"forward_cm": 0, "left_cm": 0, "yaw_deg": 0}
            for tgt, col, th in ((home, (150, 150, 150), 1), (t, (255, 220, 0), 2)):
                for z in (0.0, CUBE):
                    uv, ok = o.project(o.cube_corners(tgt, z))
                    if ok.all():
                        cv2.polylines(ov, [uv.astype(np.int32)], True, col, th if z else 1, cv2.LINE_AA)
            c = find_red_cube(ov)
            if c is not None:
                xy = o.table_point(*c)
                tx = o.home + np.array([t["forward_cm"], t["left_cm"]]) / 100
                d = (xy - tx) * 100
                dist = float(np.hypot(*d))
                good = dist < MARK_TOL_CM
                cv2.drawMarker(ov, (int(c[0]), int(c[1])), (0, 255, 0) if good else (255, 60, 60), cv2.MARKER_CROSS, 18, 2)
                info = (f"cube is {dist:.1f} cm from the target" + (" - OK" if good else
                        f" (move it {abs(d[0]):.1f} cm {'back' if d[0] > 0 else 'forward'}, "
                        f"{abs(d[1]):.1f} cm {'right' if d[1] > 0 else 'left'})"))
                self.cube_xy = xy
            else:
                self.cube_xy = None
                info = "cube not found in the overhead view"
        if self.state == "recording":
            cv2.circle(ov, (20, 20), 9, (255, 0, 0), -1)
        jp = {}
        for name, img in (("overhead", ov), ("wrist", wr)):
            jp[name] = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        with self.lock:
            self.jpg.update(jp)
            self.cube_info = info

    # ---------------- episode control
    def episode_index(self):
        return self.dataset.num_episodes if self.dataset is not None else 0

    def target(self):
        i = self.episode_index()
        return self.plan[i] if i < len(self.plan) else None

    def rest_status(self):
        if not self.rest or not self.joints:
            return None
        diffs = {j: self.joints[j] - self.rest[j] for j in JOINTS}
        ok = all(abs(d) <= (REST_TOL_GRIPPER if j == "gripper" else REST_TOL_DEG) for j, d in diffs.items())
        return {"ok": ok, "diffs": {j: round(d, 1) for j, d in diffs.items()}}

    def command(self, c):
        with self.lock:
            if c == "start" and self.state == "ready":
                if self.paused or self.ramp is not None:
                    return "Resume teleop first (and let the follower catch up)."
                if self.saving:
                    return "Still saving the last episode."
                if self.target() is None:
                    return "All planned episodes are recorded."
                self.frames = 0
                self.start_cube = None if self.cube_xy is None else \
                    [round(float(v), 2) for v in (self.cube_xy - self.overhead.home) * 100]
                self.rec_started = time.time()
                self.state = "recording"
                self.message = f"Recording episode {self.episode_index()}."
            elif c == "save" and self.state in ("recording", "review"):
                if self.frames < FPS:
                    return "Less than 1 s recorded; Discard instead."
                self.state = "ready"
                self.saving = True
                idx, tgt, frames = self.episode_index(), self.target(), self.frames
                threading.Thread(target=self._save, args=(idx, tgt, frames, self.start_cube), daemon=True).start()
                self.message = f"Saving episode {idx}..."
            elif c == "discard" and self.state in ("recording", "review"):
                self.dataset.clear_episode_buffer()
                self.state = "ready"
                self.message = "Discarded. Reset and press Start to redo this placement."
            elif c == "sethome":
                if self.cube_xy is None:
                    return "The cube is not visible in the overhead view."
                self.overhead.home = np.array(self.cube_xy, float)
                self.side("home.json").write_text(json.dumps(
                    {"home_world_xy_m": self.overhead.home.round(5).tolist(),
                     "note": "cube centre detected on the physical mark; placements are offsets from here"}, indent=1))
                self.message = "Home set to the cube on the mark. Offsets are now measured from the mark."
            elif c == "setrest" and self.joints:
                self.rest = dict(self.joints)
                self.side("rest.json").write_text(json.dumps(self.rest, indent=1))
                self.message = "Rest pose saved. Return here before every episode."
            elif c == "pause" and self.state == "ready" and not self.paused:
                self.paused, self.hold, self.ramp = True, None, None
                self.message = "Teleop paused: the follower holds still and ignores the leader. Resume when ready."
            elif c == "resume" and self.state == "ready" and self.paused:
                self.ramp = (dict(self.hold) if self.hold else {f"{j}.pos": self.joints[j] for j in JOINTS}, time.time())
                self.paused, self.hold = False, None
                self.message = f"Resuming: the follower eases to the leader over {RESUME_RAMP_S:.1f} s."
            elif c == "stop" and self.state in ("ready", "setup"):
                self.state = "stopped"
                threading.Thread(target=self._shutdown, daemon=True).start()
                self.message = "Stopping: finishing the dataset files..."
            else:
                return f"'{c}' is not available now."
        return "ok"

    def _save(self, idx, tgt, frames, start_cube=None):
        try:
            self.dataset.save_episode()
            with self.side("episodes.jsonl").open("a") as f:
                f.write(json.dumps({"episode_index": idx, "placement": tgt, "frames": frames,
                                    "cube_detected_at_start_cm": None if start_cube is None else
                                    {"forward_cm": start_cube[0], "left_cm": start_cube[1]},
                                    "rest_pose": self.rest, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            msg = f"Saved episode {idx} ({frames / FPS:.1f} s). Reset, place the cube on the new outline, press Start."
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = f"Saving failed: {e}"
        with self.lock:
            self.saving = False
            self.message = msg

    def _shutdown(self):
        while self.saving:
            time.sleep(0.2)
        if self.robot is not None:
            self.loop_done.wait(timeout=3)
        try:
            if self.dataset is not None:
                self.dataset.finalize()
        finally:
            for dev in (self.robot, self.leader):
                try:
                    if dev is not None and dev.is_connected:
                        dev.disconnect()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            with self.lock:
                self.message = f"Stopped. Dataset: {self.root}"
            print(f"Dataset written to {self.root}", flush=True)

    def status(self):
        with self.lock:
            t = self.target()
            return {
                "state": self.state, "message": self.message, "error": self.error,
                "episode": self.episode_index(), "total": len(self.plan), "saving": self.saving,
                "seconds": round(time.time() - self.rec_started, 1) if self.state == "recording" else None,
                "target": t, "target_text": describe(t) if t else "", "cube": self.cube_info,
                "rest": self.rest_status(), "has_rest": self.rest is not None, "paused": self.paused,
                "has_home": self.side("home.json").exists(),
                "joints": {j: round(v, 1) for j, v in self.joints.items()},
                "cam_age": ({n: 0.0 for n in self.cam} if self.a.cameras == "opencv" and self.state not in ("setup", "connecting")
                            else {n: (round(time.time() - c[1], 2) if c else None) for n, c in self.cam.items()}),
                "camera_source": self.a.cameras, "cams": sorted(self.cam_thumbs), "cam_role": self.cam_role,
                "ports": [{"port": p, "motion": round(self.port_motion.get(p, 0), 1)} for p in self.ports],
                "leader_port": self.leader_port, "follower_port": self.follower_port,
            }


# ------------------------------------------------------------------ web page
PAGE = (HERE / "real_record.html")


def serve(app: App, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            if u.path == "/status":
                return self._send(200, "application/json", json.dumps(app.status()).encode())
            if u.path.startswith("/thumb/"):
                b = app.cam_thumbs.get(int(u.path.rsplit("/", 1)[1].split("?")[0]))
                return self._send(200 if b else 404, "image/jpeg", b or b"")
            if u.path in ("/overhead.mjpg", "/wrist.mjpg"):
                name = u.path[1:].split(".")[0]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    last = None
                    while app.state != "stopped":
                        j = app.jpg.get(name)
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                        time.sleep(1 / 20)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return None
            return self._send(404, "text/plain", b"not found")

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/frame":
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                name = q.get("cam")
                if name not in ("overhead", "wrist"):
                    return self._send(400, "text/plain", b"bad cam")
                return self._send(200, "text/plain", app.put_frame(name, body).encode())
            if u.path == "/cmd":
                return self._send(200, "application/json", json.dumps({"result": app.command(q.get("c", ""))}).encode())
            if u.path == "/assign" and app.state == "setup":
                with app.lock:
                    if "cam" in q:
                        role, idx = q["role"], int(q["cam"])
                        other = "wrist" if role == "overhead" else "overhead"
                        if app.cam_role.get(other) == idx:
                            app.cam_role[other] = app.cam_role.get(role)
                        app.cam_role[role] = idx
                    if "leader" in q:
                        app.leader_port = q["leader"]
                        rest = [p for p in app.ports if p != q["leader"]]
                        app.follower_port = rest[0] if len(rest) == 1 else app.follower_port
                return self._send(200, "application/json", b'{"result":"ok"}')
            if u.path == "/connect" and app.state == "setup":
                cams_ok = (None not in app.cam_role.values()) if app.a.cameras == "opencv" else \
                    max(app.latest(n)[1] for n in ("overhead", "wrist")) <= 1.0
                if not cams_ok or not app.leader_port or not app.follower_port:
                    return self._send(200, "application/json", b'{"result":"Assign both cameras (live) and the leader first."}')
                with app.lock:
                    app.state = "connecting"
                    app.message = "Connecting the arms and cameras..."

                def go():
                    try:
                        app.connect()
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        with app.lock:
                            app.state = "setup"
                            app.error = f"Connect failed: {e}"
                            app._probe_stop.clear()
                            app._probe_done.clear()
                        threading.Thread(target=app.probe_ports, daemon=True).start()

                threading.Thread(target=go, daemon=True).start()
                return self._send(200, "application/json", b'{"result":"ok"}')
            return self._send(404, "text/plain", b"not found")

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    return srv


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE / "recordings" / "real-pickplace-1"))
    ap.add_argument("--repo-id", dest="repo_id", default="local/real-pickplace-1")
    ap.add_argument("--task", default="Put the red cube in the grey bin.")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episode-s", dest="max_episode_s", type=float, default=60.0)
    ap.add_argument("--cameras", choices=["opencv", "browser"], default="opencv",
                    help="opencv: Python reads both cameras (needs camera permission for Terminal); "
                         "browser: the Chrome page captures them and sends frames")
    ap.add_argument("--overhead", type=int, default=None, help="opencv camera index (else pick on the page)")
    ap.add_argument("--wrist", type=int, default=None)
    ap.add_argument("--max-frame-age", dest="max_frame_age", type=float, default=0.25,
                    help="seconds; older camera frames pause recording")
    ap.add_argument("--leader-port", dest="leader_port", default=None)
    ap.add_argument("--follower-port", dest="follower_port", default=None)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=float, default=2.0,
                    help="save raw PNG frames of both cameras every N seconds for the whole session (0 = off)")
    return ap


def main():
    a = build_parser().parse_args()
    app = App(a)
    srv = serve(app, a.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Open http://localhost:{a.port}  (Ctrl+C to quit; use the page's Stop button to finish the dataset)", flush=True)
    threading.Thread(target=app.probe_ports, daemon=True).start()
    if a.cameras == "opencv":
        app.probe_cameras_until_ok()
    try:
        while True:
            time.sleep(0.5)
            if app.state == "stopped" and app.message.startswith("Stopped."):
                time.sleep(1)
                break
    except KeyboardInterrupt:
        print("Ctrl+C: finishing the dataset...", flush=True)
        if app.state not in ("stopped", "setup"):
            with app.lock:
                if app.state in ("recording", "review"):
                    app.dataset.clear_episode_buffer()
                app.state = "stopped"
            app._shutdown()
    srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
