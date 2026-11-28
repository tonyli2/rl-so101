#!/usr/bin/env python3
"""Run a LeRobot ACT checkpoint on the real SO-101 follower, in a desktop window.

    cd ~/Downloads/EECE576/rl-so101/lerobot-tabletop
    ../../.venv-act/bin/python real_run_act.py --policy <checkpoint>/pretrained_model

Start it from the macOS Terminal app or Finder (camera permission), like real_record_gui.py.

Observations match the recordings made with real_record_gui.py exactly: both cameras raw
640x480 RGB from LeRobot's OpenCV cameras (no undistortion, no resizing), follower joints in the
real LeRobot convention (degrees, gripper 0-100; no sim offsets).

Timing. On an M1 Pro, ACT on two 640x480 images takes ~120 ms on the GPU (MPS) and ~0.6 s on
the CPU, so a fresh prediction every 33 ms (how the model was evaluated in sim) is impossible.
Instead, a worker thread predicts a whole chunk (chunk_size actions, normally 100) from the
newest observation, over and over. Each chunk is anchored at the control tick its observation was
taken, so the delay is compensated, and the 30 Hz control loop sends ACT's temporal ensemble of
all chunks covering the current tick (weights exp(-coeff * i), oldest first, coeff from the
checkpoint's config, 0.01 in training). Each command is also limited to --max-step degrees per
tick from the arm's present position.

macOS 13.1 lacks an MPS op ACT uses (int pow in the 2-D sine position embedding); casting that
module's temperature to float avoids it without changing results.

Trials are logged to <policy>/../../real_trials.jsonl: placement detected at start, result,
duration.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6 import QtCore, QtGui, QtWidgets

from real_record import CALIBRATION_DIR, FPS, H, JOINTS, W, Overhead, find_red_cube

HERE = Path(__file__).resolve().parent
TASK = "Put the red cube in the grey bin."
BIG = "font-size:17px; font-weight:700; padding:10px 18px; border-radius:6px;"


# ------------------------------------------------------------------ policy
class Policy:
    def __init__(self, path: Path, device: str):
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = torch.device(device)
        cfg = ACTConfig.from_pretrained(path)
        cfg.device = device
        self.cfg = cfg
        self.policy = ACTPolicy.from_pretrained(path, config=cfg).to(self.device).eval()
        for m in self.policy.modules():  # see module docstring (macOS 13.1 MPS)
            if hasattr(m, "_temperature"):
                m._temperature = float(m._temperature)
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        feats = list(cfg.input_features)
        self.image_keys = [k for k in feats if k.startswith("observation.images.")]
        if sorted(self.image_keys) != ["observation.images.overhead", "observation.images.wrist"]:
            raise ValueError(f"checkpoint expects inputs {feats}; this runner provides overhead + wrist cameras")
        self.chunk = int(cfg.chunk_size)
        self.coeff = float(cfg.temporal_ensemble_coeff or 0.0)

    def chunk_from(self, state: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.policies.utils import prepare_observation_for_inference

        obs = {"observation.state": state.astype(np.float32),
               "observation.images.overhead": images["overhead"], "observation.images.wrist": images["wrist"]}
        with torch.inference_mode():
            batch = self.pre(prepare_observation_for_inference(obs, self.device, TASK, "so_follower"))
            actions = self.post(self.policy.predict_action_chunk(batch))
        return actions[0].float().cpu().numpy()  # (chunk, 6)


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, a):
        self.a = a
        self.lock = threading.RLock()
        self.state = "setup"  # setup, connecting, idle, running, returning, stopped
        self.message = "Loading the policy..."
        self.error = ""
        self.cam_thumbs: dict[int, bytes] = {}
        self.cam_role = {"overhead": a.overhead, "wrist": a.wrist}
        self.follower_port = a.follower_port
        self.robot = None
        self.policy: Policy | None = None
        self.jpg = {"overhead": None, "wrist": None}
        self.joints: dict[str, float] = {}
        self.tick = 0
        self.chunks: list[tuple[int, np.ndarray]] = []  # (start tick, actions)
        self.latency = []
        self.run_started = 0.0
        self.trial_start_cube = None
        self.cube_xy = None
        self.overhead = Overhead()
        rec = HERE / "recordings"
        home = rec / "real-pickplace-1_home.json"
        if home.exists():
            self.overhead.home = np.array(json.loads(home.read_text())["home_world_xy_m"], float)
        rest = rec / "real-pickplace-1_rest.json"
        self.rest = json.loads(rest.read_text()) if rest.exists() else None
        self.trials_path = Path(a.policy).resolve().parent.parent / "real_trials.jsonl"
        self.latest_obs = None  # (tick, state, images)
        self.loop_done = threading.Event()

    # ---------------- setup
    def load_policy(self):
        try:
            t0 = time.time()
            self.policy = Policy(Path(self.a.policy), self.a.device)
            # Warm up (first MPS call compiles kernels).
            z = {n: np.zeros((H, W, 3), np.uint8) for n in ("overhead", "wrist")}
            self.policy.chunk_from(np.zeros(6, np.float32), z)
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                self.policy.chunk_from(np.zeros(6, np.float32), z)
                ts.append(time.perf_counter() - t)
            with self.lock:
                self.message = (f"Policy loaded on {self.a.device} in {time.time() - t0:.0f} s; one chunk takes "
                                f"{1000 * float(np.median(ts)):.0f} ms. Pick the cameras, then Connect.")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            with self.lock:
                self.error = f"Could not load the policy: {e}"

    def probe_cameras(self, seconds=90):
        t_end = time.time() + seconds
        while time.time() < t_end and self.state == "setup":
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
                    cv2.putText(frame, f"camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    thumbs[i] = cv2.imencode(".jpg", cv2.resize(frame, (320, 240)))[1].tobytes()
            with self.lock:
                self.cam_thumbs = thumbs
            if thumbs:
                return
            with self.lock:
                self.error = ("No camera opened. Start this from the macOS Terminal app or Finder (not from Claude) "
                              "and allow camera access for Terminal.")
            time.sleep(2)

    def find_follower(self):
        if self.follower_port:
            return
        from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

        cal = json.loads((CALIBRATION_DIR / "follower.json").read_text())
        for p in sorted(glob.glob("/dev/tty.usbmodem*")):
            b = SerialMotorsBus(p, {n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS)})
            try:
                b.connect(handshake=False)
                h = b.sync_read("Homing_Offset", normalize=False)
                if all(int(h[j]) == int(cal[j]["homing_offset"]) for j in JOINTS):
                    with self.lock:
                        self.follower_port = p
            except Exception:  # noqa: BLE001
                pass
            finally:
                try:
                    b.disconnect(disable_torque=False)
                except Exception:  # noqa: BLE001
                    pass
        if not self.follower_port:
            with self.lock:
                self.error = "Follower arm not found (no port matches calibration/follower.json)."

    def connect(self):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        cams = {role: OpenCVCameraConfig(index_or_path=idx, fps=FPS, width=W, height=H, backend=cv2.CAP_AVFOUNDATION)
                for role, idx in self.cam_role.items()}
        self.robot = SOFollower(SOFollowerRobotConfig(port=self.follower_port, id="follower", cameras=cams,
                                                      calibration_dir=CALIBRATION_DIR, num_read_retries=5))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            raise RuntimeError("The follower's motors do not match calibration/follower.json.")
        threading.Thread(target=self.control_loop, daemon=True).start()
        threading.Thread(target=self.inference_loop, daemon=True).start()
        with self.lock:
            self.state = "idle"
            self.message = ("Connected; the follower holds its pose. Put the cube down, move hands away, "
                            "then Run policy.")

    # ---------------- loops
    def inference_loop(self):
        last_tick = -1
        while self.state != "stopped":
            if self.state != "running" or self.latest_obs is None or self.latest_obs[0] == last_tick:
                time.sleep(0.005)
                continue
            tick, state, imgs = self.latest_obs
            last_tick = tick
            t = time.perf_counter()
            try:
                actions = self.policy.chunk_from(state, imgs)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                with self.lock:
                    self.error = f"Policy error: {e}"
                    self.state = "idle"
                continue
            dt = time.perf_counter() - t
            with self.lock:
                if self.state == "running":
                    self.chunks.append((tick, actions))
                    self.chunks = [c for c in self.chunks if c[0] + len(c[1]) > self.tick]
                    self.latency = (self.latency + [dt])[-30:]

    def ensembled(self, tick):
        preds = [a[tick - s] for s, a in self.chunks if s <= tick < s + len(a)]
        if not preds:
            return None
        w = np.exp(-self.policy.coeff * np.arange(len(preds)))  # oldest chunk first, as in ACT
        return (np.stack(preds) * w[:, None]).sum(0) / w.sum()

    def control_loop(self):
        period = 1.0 / FPS
        nxt = time.perf_counter()
        last_ui = 0.0
        fails = 0
        hold = None
        ret = None  # (start pose, start time) while returning to rest
        while self.state != "stopped":
            try:
                obs = self.robot.get_observation()
                fails = 0
                present = np.array([obs[f"{j}.pos"] for j in JOINTS], float)
                imgs = {n: obs[n] for n in ("overhead", "wrist")}
                with self.lock:
                    self.tick += 1
                    tick = self.tick
                    self.joints = dict(zip(JOINTS, present.tolist()))
                    self.latest_obs = (tick, present.astype(np.float32), imgs)
                    st = self.state
                    target = None
                    if st == "running":
                        target = self.ensembled(tick)
                        hold = None
                        if time.time() - self.run_started > self.a.max_run_s:
                            self.state = "idle"
                            self.message = f"Stopped after {self.a.max_run_s:.0f} s. Mark the trial Success or Fail."
                    elif st == "returning" and self.rest:
                        if ret is None:
                            ret = (present.copy(), time.time())
                        k = min(1.0, (time.time() - ret[1]) / 3.0)
                        goal = np.array([self.rest[j] for j in JOINTS], float)
                        target = ret[0] + k * (goal - ret[0])
                        if k >= 1.0:
                            self.state, ret = "idle", None
                            self.message = "At the rest pose."
                    if st != "returning":
                        ret = None
                if target is None:  # idle, or no chunk yet: hold where the arm is
                    if hold is None:
                        hold = present.copy()
                    target = hold
                else:
                    hold = None
                step = np.clip(np.asarray(target, float) - present, -self.a.max_step, self.a.max_step)
                cmd = present + step
                self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, cmd)})
                now = time.perf_counter()
                if now - last_ui > 1 / 15:
                    last_ui = now
                    self.update_ui(imgs)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"control tick failed ({fails} in a row): {e}", flush=True)
                if fails >= 10:
                    with self.lock:
                        self.error = f"Arm/camera communication keeps failing: {e}"
                        if self.state == "running":
                            self.state = "idle"
                    time.sleep(0.1)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 0.5:
                nxt = time.perf_counter()
        self.loop_done.set()

    def update_ui(self, imgs):
        ov = np.ascontiguousarray(imgs["overhead"]).copy()
        wr = np.ascontiguousarray(imgs["wrist"]).copy()
        c = find_red_cube(ov)
        if c is not None:
            self.cube_xy = self.overhead.table_point(*c)
            cv2.drawMarker(ov, (int(c[0]), int(c[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        else:
            self.cube_xy = None
        uv, ok = self.overhead.project(self.overhead.cube_corners({"forward_cm": 0, "left_cm": 0, "yaw_deg": 0}, 0.0))
        if ok.all():
            cv2.polylines(ov, [uv.astype(np.int32)], True, (150, 150, 150), 1, cv2.LINE_AA)
        if self.state == "running":
            cv2.circle(ov, (20, 20), 9, (255, 0, 0), -1)
        jp = {n: cv2.imencode(".jpg", cv2.cvtColor(i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
              for n, i in (("overhead", ov), ("wrist", wr))}
        with self.lock:
            self.jpg.update(jp)

    # ---------------- commands
    def command(self, c):
        with self.lock:
            if c == "run" and self.state == "idle":
                self.chunks = []
                self.policy.policy.reset()
                self.run_started = time.time()
                self.trial_start_cube = None if self.cube_xy is None else \
                    [round(float(v), 2) for v in (self.cube_xy - self.overhead.home) * 100]
                self.state = "running"
                self.message = "Running the policy. Space or Stop holds the arm where it is."
            elif c == "stop" and self.state in ("running", "returning"):
                self.state = "idle"
                self.chunks = []
                self.message = "Stopped: the arm holds its pose. Mark the trial, then Return to rest."
            elif c == "rest" and self.state == "idle":
                if not self.rest:
                    return "No rest pose saved (record one in the recorder first)."
                self.state = "returning"
                self.message = "Returning to the rest pose over 3 s..."
            elif c in ("success", "fail") and self.state == "idle" and self.run_started:
                rec = {"result": c, "policy": str(Path(self.a.policy).resolve()),
                       "cube_at_start_cm": None if self.trial_start_cube is None else
                       {"forward_cm": self.trial_start_cube[0], "left_cm": self.trial_start_cube[1]},
                       "seconds": round(time.time() - self.run_started, 1),
                       "median_chunk_ms": round(1000 * float(np.median(self.latency)), 1) if self.latency else None,
                       "max_step_deg": self.a.max_step, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                with self.trials_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                self.run_started = 0.0
                self.message = f"Logged {c} to {self.trials_path.name}."
            elif c == "quit" and self.state in ("idle", "setup"):
                self.state = "stopped"
            else:
                return f"'{c}' is not available now."
        return "ok"

    def shutdown(self):
        with self.lock:
            self.state = "stopped"
        self.loop_done.wait(timeout=3)
        if self.robot is not None and self.robot.is_connected:
            try:
                self.robot.disconnect()
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    def trials_summary(self):
        if not self.trials_path.exists():
            return "no trials logged yet"
        rows = [json.loads(l) for l in self.trials_path.read_text().splitlines() if l.strip()]
        mine = [r for r in rows if r.get("policy") == str(Path(self.a.policy).resolve())]
        ok = sum(r["result"] == "success" for r in mine)
        return f"{ok}/{len(mine)} successes with this checkpoint"


# ------------------------------------------------------------------ window
class Window(QtWidgets.QWidget):
    def __init__(self, r: Runner):
        super().__init__()
        self.r = r
        self.setWindowTitle(f"ACT runner - {Path(r.a.policy).parent.name}")
        self.setStyleSheet("QWidget{background:#151515;color:#eee;font-size:14px} QLabel#h{color:#999;font-weight:600}")
        v = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.err = QtWidgets.QLabel()
        self.err.setWordWrap(True)
        self.err.setStyleSheet("color:#ff6b6b")
        v.addWidget(self.status)
        v.addWidget(self.err)
        self.stack = QtWidgets.QStackedWidget()
        v.addWidget(self.stack, 1)
        # setup page
        s = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(s)
        h = QtWidgets.QLabel("Pick which feed is the overhead camera and which is the wrist camera")
        h.setObjectName("h")
        sv.addWidget(h)
        self.thumb_row = QtWidgets.QHBoxLayout()
        sv.addLayout(self.thumb_row)
        self.follower = QtWidgets.QLabel()
        sv.addWidget(self.follower)
        self.connect_btn = QtWidgets.QPushButton("Connect follower + cameras")
        self.connect_btn.setStyleSheet(BIG + "background:#2ecc71;color:#000")
        self.connect_btn.clicked.connect(self.connect)
        sv.addWidget(self.connect_btn, 0, QtCore.Qt.AlignLeft)
        sv.addStretch(1)
        self.stack.addWidget(s)
        # run page
        rp = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(rp)
        feeds = QtWidgets.QHBoxLayout()
        self.views = {}
        for n in ("overhead", "wrist"):
            lab = QtWidgets.QLabel()
            lab.setMinimumSize(480, 360)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet("background:#000")
            lab.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            feeds.addWidget(lab, 1)
            self.views[n] = lab
        rv.addLayout(feeds, 1)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        rv.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        self.b_run = self._btn("Run policy  [Space]", "#2ecc71", "#000", lambda: self.r.command("run"))
        self.b_stop = self._btn("STOP  [Space / Esc]", "#e74c3c", "#fff", lambda: self.r.command("stop"))
        self.b_ok = self._btn("Success", "#2980b9", "#fff", lambda: self.r.command("success"))
        self.b_fail = self._btn("Fail", "#555", "#fff", lambda: self.r.command("fail"))
        self.b_rest = self._btn("Return to rest", "#8e44ad", "#fff", lambda: self.r.command("rest"))
        for b in (self.b_run, self.b_stop, self.b_ok, self.b_fail, self.b_rest):
            row.addWidget(b)
        rv.addLayout(row)
        self.stack.addWidget(rp)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.space)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, activated=lambda: self.r.command("stop"))
        self.thumbs_key = None
        QtCore.QTimer(self, interval=66, timeout=self.refresh).start()
        self.resize(1400, 860)

    def _btn(self, text, bg, fg, fn):
        b = QtWidgets.QPushButton(text)
        b.setStyleSheet(BIG + f"background:{bg};color:{fg}")
        b.setFocusPolicy(QtCore.Qt.NoFocus)

        def go():
            res = fn()
            if res != "ok":
                QtWidgets.QMessageBox.information(self, "ACT runner", res)

        b.clicked.connect(go)
        return b

    def space(self):
        st = self.r.state
        self.r.command("stop" if st in ("running", "returning") else "run" if st == "idle" else "")

    def connect(self):
        r = self.r
        if r.policy is None or None in r.cam_role.values() or not r.follower_port:
            QtWidgets.QMessageBox.warning(self, "ACT runner", "Wait for the policy to load, assign both cameras, "
                                          "and make sure the follower was found.")
            return
        with r.lock:
            r.state = "connecting"
            r.message = "Connecting..."

        def go():
            try:
                r.connect()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                with r.lock:
                    r.state, r.error = "setup", f"Connect failed: {e}"

        threading.Thread(target=go, daemon=True).start()

    def assign(self, role, idx):
        r = self.r
        with r.lock:
            other = "wrist" if role == "overhead" else "overhead"
            if r.cam_role.get(other) == idx:
                r.cam_role[other] = r.cam_role.get(role)
            r.cam_role[role] = idx

    def refresh(self):
        r = self.r
        st = r.state
        self.status.setText(("<b style='color:#e74c3c'>&#9679; RUNNING</b>  " if st == "running" else f"<b>{st}</b>  ")
                            + r.message)
        self.err.setText(r.error)
        if st in ("setup", "connecting"):
            self.stack.setCurrentIndex(0)
            key = (tuple(r.cam_thumbs), tuple(sorted(r.cam_role.items(), key=str)))
            if key != self.thumbs_key:
                self.thumbs_key = key
                while self.thumb_row.count():
                    it = self.thumb_row.takeAt(0)
                    if it.widget():
                        it.widget().deleteLater()
                for i, jpg in sorted(r.cam_thumbs.items()):
                    box = QtWidgets.QVBoxLayout()
                    img = QtWidgets.QLabel()
                    pm = QtGui.QPixmap()
                    pm.loadFromData(jpg, "JPG")
                    img.setPixmap(pm)
                    box.addWidget(img)
                    rowl = QtWidgets.QHBoxLayout()
                    for role in ("overhead", "wrist"):
                        b = QtWidgets.QPushButton(role.capitalize())
                        b.clicked.connect(lambda _=False, rr=role, ii=i: self.assign(rr, ii))
                        rowl.addWidget(b)
                    tag = QtWidgets.QLabel(next((k for k, v in r.cam_role.items() if v == i), ""))
                    tag.setStyleSheet("color:#2ecc71;font-weight:700")
                    rowl.addWidget(tag)
                    box.addLayout(rowl)
                    w = QtWidgets.QWidget()
                    w.setLayout(box)
                    self.thumb_row.addWidget(w)
                self.thumb_row.addStretch(1)
            self.follower.setText(f"Follower: {r.follower_port or 'searching...'}   (the leader is not used)")
            self.connect_btn.setEnabled(st == "setup")
            return
        if st == "stopped":
            QtWidgets.QApplication.quit()
            return
        self.stack.setCurrentIndex(1)
        for n, lab in self.views.items():
            j = r.jpg.get(n)
            if j is not None:
                pm = QtGui.QPixmap()
                pm.loadFromData(j, "JPG")
                lab.setPixmap(pm.scaled(lab.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        lat = f"{1000 * float(np.median(r.latency)):.0f} ms per chunk" if r.latency else "-"
        cube = "cube not seen" if r.cube_xy is None else \
            "cube at forward {:+.1f} cm, left {:+.1f} cm from the mark".format(*((r.cube_xy - r.overhead.home) * 100))
        self.info.setText(f"{cube}   |   inference {lat}, {len(r.chunks)} chunks blended   |   "
                          f"max {r.a.max_step:g} deg per tick   |   {r.trials_summary()}")
        self.b_run.setEnabled(st == "idle")
        self.b_stop.setEnabled(st in ("running", "returning"))
        self.b_ok.setEnabled(st == "idle" and bool(r.run_started))
        self.b_fail.setEnabled(st == "idle" and bool(r.run_started))
        self.b_rest.setEnabled(st == "idle" and bool(r.rest))

    def closeEvent(self, e):
        if self.r.state in ("running", "returning"):
            self.r.command("stop")
        self.r.shutdown()
        e.accept()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="path to a checkpoint's pretrained_model folder")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-step", dest="max_step", type=float, default=4.0,
                    help="max change per joint per 1/30 s tick, degrees (4 = 120 deg/s)")
    ap.add_argument("--max-run-s", dest="max_run_s", type=float, default=60.0)
    ap.add_argument("--overhead", type=int, default=None)
    ap.add_argument("--wrist", type=int, default=None)
    ap.add_argument("--follower-port", dest="follower_port", default=None)
    a = ap.parse_args()
    if not (Path(a.policy) / "config.json").exists():
        sys.exit(f"{a.policy} has no config.json: pass the checkpoint's pretrained_model folder.")
    qapp = QtWidgets.QApplication(sys.argv)
    r = Runner(a)
    threading.Thread(target=r.load_policy, daemon=True).start()
    threading.Thread(target=r.find_follower, daemon=True).start()
    threading.Thread(target=r.probe_cameras, daemon=True).start()
    w = Window(r)
    w.show()
    qapp.exec()
    r.shutdown()


if __name__ == "__main__":
    main()
