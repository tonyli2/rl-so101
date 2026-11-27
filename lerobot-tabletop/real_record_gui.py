#!/usr/bin/env python3
"""Desktop window for recording real leader/follower demonstrations (same recorder as real_record.py).

Start it from the macOS **Terminal app** (not from Claude), so macOS can grant Terminal camera access:

    cd ~/Downloads/EECE576/rl-so101/lerobot-tabletop
    ../../.venv-lerobot/bin/python real_record_gui.py

Keys: Space = Start / Save, D = Discard. Accepts the same options as real_record.py.
"""

from __future__ import annotations

import sys
import threading
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from real_record import JOINTS, REST_TOL_DEG, REST_TOL_GRIPPER, App, build_parser

BIG = "font-size:17px; font-weight:700; padding:10px 18px; border-radius:6px;"


class Window(QtWidgets.QWidget):
    def __init__(self, app: App):
        super().__init__()
        self.app = app
        self.setWindowTitle("Real demo recorder")
        self.setStyleSheet("QWidget{background:#151515;color:#eee;font-size:14px} QLabel#h{color:#999;font-weight:600}")
        root = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.err = QtWidgets.QLabel()
        self.err.setWordWrap(True)
        self.err.setStyleSheet("color:#ff6b6b")
        root.addWidget(self.status)
        root.addWidget(self.err)
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)
        self.stack.addWidget(self._setup_page())
        self.stack.addWidget(self._run_page())
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.space)
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.discard())
        self.timer = QtCore.QTimer(self, interval=66, timeout=self.refresh)
        self.timer.start()
        self.thumbs_shown = None
        self.resize(1400, 900)

    # ---------------- setup page
    def _setup_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QLabel("1. CAMERAS: pick which feed is the overhead camera and which is the wrist camera")
        h.setObjectName("h")
        v.addWidget(h)
        self.thumb_row = QtWidgets.QHBoxLayout()
        v.addLayout(self.thumb_row)
        h = QtWidgets.QLabel("2. ARMS (found automatically from their stored calibration)")
        h.setObjectName("h")
        v.addWidget(h)
        arms = QtWidgets.QHBoxLayout()
        self.arms = QtWidgets.QLabel()
        swap = QtWidgets.QPushButton("Swap leader / follower")
        swap.clicked.connect(self.swap_arms)
        arms.addWidget(self.arms, 1)
        arms.addWidget(swap)
        v.addLayout(arms)
        h = QtWidgets.QLabel("3. Hold the leader roughly in the follower's pose (the follower jumps to match it), then Connect")
        h.setObjectName("h")
        v.addWidget(h)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setStyleSheet(BIG + "background:#2ecc71;color:#000")
        self.connect_btn.clicked.connect(self.connect)
        v.addWidget(self.connect_btn, 0, QtCore.Qt.AlignLeft)
        v.addStretch(1)
        return w

    def rebuild_thumbs(self, s):
        key = (tuple(s["cams"]), tuple(sorted(s["cam_role"].items(), key=str)))
        if key == self.thumbs_shown:
            return
        self.thumbs_shown = key
        while self.thumb_row.count():
            item = self.thumb_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not s["cams"]:
            self.thumb_row.addWidget(QtWidgets.QLabel("Looking for cameras... (click Allow if macOS asks about the camera)"))
            return
        for i in s["cams"]:
            box = QtWidgets.QVBoxLayout()
            img = QtWidgets.QLabel()
            pm = QtGui.QPixmap()
            pm.loadFromData(self.app.cam_thumbs[i], "JPG")
            img.setPixmap(pm)
            box.addWidget(img)
            row = QtWidgets.QHBoxLayout()
            for role in ("overhead", "wrist"):
                b = QtWidgets.QPushButton(role.capitalize())
                b.clicked.connect(lambda _=False, r=role, idx=i: self.assign_cam(r, idx))
                row.addWidget(b)
            tag = next((r for r, v in s["cam_role"].items() if v == i), "")
            t = QtWidgets.QLabel(tag)
            t.setStyleSheet("color:#2ecc71;font-weight:700")
            row.addWidget(t)
            box.addLayout(row)
            cont = QtWidgets.QWidget()
            cont.setLayout(box)
            self.thumb_row.addWidget(cont)
        self.thumb_row.addStretch(1)

    def assign_cam(self, role, idx):
        with self.app.lock:
            other = "wrist" if role == "overhead" else "overhead"
            if self.app.cam_role.get(other) == idx:
                self.app.cam_role[other] = self.app.cam_role.get(role)
            self.app.cam_role[role] = idx

    def swap_arms(self):
        with self.app.lock:
            self.app.leader_port, self.app.follower_port = self.app.follower_port, self.app.leader_port

    def connect(self):
        a = self.app
        if None in a.cam_role.values() or not a.leader_port or not a.follower_port:
            QtWidgets.QMessageBox.warning(self, "Not ready", "Assign both cameras first (and make sure both arms were found).")
            return
        with a.lock:
            a.state = "connecting"
            a.message = "Connecting the arms and cameras..."

        def go():
            try:
                a.connect()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                with a.lock:
                    a.state = "setup"
                    a.error = f"Connect failed: {e}"
                    a._probe_stop.clear()
                    a._probe_done.clear()
                threading.Thread(target=a.probe_ports, daemon=True).start()

        threading.Thread(target=go, daemon=True).start()

    # ---------------- run page
    def _run_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        feeds = QtWidgets.QHBoxLayout()
        self.views = {}
        for name, cap in (("overhead", "Overhead  (yellow = where the cube goes, grey = your mark)"), ("wrist", "Wrist")):
            col = QtWidgets.QVBoxLayout()
            lab = QtWidgets.QLabel(cap)
            lab.setObjectName("h")
            img = QtWidgets.QLabel()
            img.setMinimumSize(480, 360)
            img.setAlignment(QtCore.Qt.AlignCenter)
            img.setStyleSheet("background:#000")
            img.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            col.addWidget(lab)
            col.addWidget(img, 1)
            feeds.addLayout(col, 1)
            self.views[name] = img
        v.addLayout(feeds, 1)
        bottom = QtWidgets.QHBoxLayout()
        v.addLayout(bottom)
        # episode panel
        ep = QtWidgets.QVBoxLayout()
        self.ep_title = QtWidgets.QLabel()
        self.ep_title.setStyleSheet("font-size:22px;font-weight:700")
        self.target = QtWidgets.QLabel()
        self.target.setStyleSheet("font-size:18px;color:#ffd400")
        self.target.setWordWrap(True)
        self.offsets = QtWidgets.QLabel()
        self.offsets.setWordWrap(True)
        self.cube = QtWidgets.QLabel()
        self.home_hint = QtWidgets.QLabel()
        self.home_hint.setStyleSheet("color:#ff6b6b")
        for x in (self.ep_title, self.target, self.offsets, self.cube, self.home_hint):
            ep.addWidget(x)
        btns = QtWidgets.QHBoxLayout()
        self.b_start = self._btn("Start  [Space]", "#2ecc71", "#000", lambda: self.app.command("start"))
        self.b_save = self._btn("Save  [Space]", "#3498db", "#fff", lambda: self.app.command("save"))
        self.b_discard = self._btn("Discard  [D]", "#555", "#fff", lambda: self.discard())
        self.b_home = self._btn("Cube is on the mark", "#333", "#fff", self.set_home)
        for b in (self.b_start, self.b_save, self.b_discard, self.b_home):
            btns.addWidget(b)
        ep.addLayout(btns)
        bottom.addLayout(ep, 3)
        # reset panel
        rs = QtWidgets.QVBoxLayout()
        h = QtWidgets.QLabel("RESET: arm to rest (all green), cube on the yellow outline, hands out of view, Start")
        h.setObjectName("h")
        h.setWordWrap(True)
        self.rest = QtWidgets.QLabel()
        rs.addWidget(h)
        rs.addWidget(self.rest)
        rb = QtWidgets.QVBoxLayout()
        self.b_pause = self._btn("Pause teleop", "#8e44ad", "#fff", self.toggle_pause)
        rs.addWidget(self.b_pause)
        self.b_rest = self._btn("Set rest pose = current", "#333", "#fff", self.set_rest)
        self.b_stop = self._btn("Stop && finish dataset", "#222", "#ff6b6b", self.stop)
        rb.addWidget(self.b_rest)
        rb.addWidget(self.b_stop)
        rs.addLayout(rb)
        bottom.addLayout(rs, 2)
        return w

    def _btn(self, text, bg, fg, fn):
        b = QtWidgets.QPushButton(text)
        b.setStyleSheet(BIG + f"background:{bg};color:{fg}")
        b.setFocusPolicy(QtCore.Qt.NoFocus)  # keep Space for Start/Save, not for the focused button
        b.clicked.connect(lambda: self.report(fn()))
        return b

    def report(self, result):
        if isinstance(result, str) and result != "ok":
            QtWidgets.QMessageBox.information(self, "Recorder", result)

    def space(self):
        st = self.app.state
        if st == "ready":
            self.report(self.app.command("start"))
        elif st in ("recording", "review"):
            self.report(self.app.command("save"))

    def discard(self):
        if self.app.state in ("recording", "review") and self.ask("Discard this episode?"):
            return self.app.command("discard")
        return None

    def set_home(self):
        if self.ask("Is the cube sitting on your home mark, not rotated?"):
            return self.app.command("sethome")
        return None

    def toggle_pause(self):
        if self.app.paused:
            if self.ask("Resume? Hold the leader roughly where you want the follower: it eases there over 1.5 s."):
                return self.app.command("resume")
            return None
        return self.app.command("pause")

    def set_rest(self):
        if self.ask("Use the follower's current pose as the rest pose for every episode?"):
            return self.app.command("setrest")
        return None

    def stop(self):
        if self.ask("Stop recording and finish the dataset files?"):
            return self.app.command("stop")
        return None

    def ask(self, q):
        return QtWidgets.QMessageBox.question(self, "Recorder", q) == QtWidgets.QMessageBox.Yes

    # ---------------- refresh
    def refresh(self):
        s = self.app.status()
        rec = s["state"] == "recording"
        self.status.setText(("<b style='color:#e74c3c'>&#9679; RECORDING %ss</b>  " % s["seconds"] if rec else
                             f"<b>{s['state']}</b>  ") + (s["message"] or ""))
        self.err.setText(s["error"] or "")
        if s["state"] in ("setup", "connecting"):
            self.stack.setCurrentIndex(0)
            self.rebuild_thumbs(s)
            self.arms.setText(f"leader: {s['leader_port'] or '?'}     follower: {s['follower_port'] or '?'}")
            self.connect_btn.setEnabled(s["state"] == "setup")
            return
        if s["state"] == "stopped" and s["message"].startswith("Stopped."):
            QtWidgets.QApplication.quit()
            return
        self.stack.setCurrentIndex(1)
        for name, lab in self.views.items():
            j = self.app.jpg.get(name)
            if j is not None:
                pm = QtGui.QPixmap()
                pm.loadFromData(j, "JPG")
                lab.setPixmap(pm.scaled(lab.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        t = s["target"]
        self.ep_title.setText(f"All {s['total']} episodes recorded" if not t else
                              f"Episode {s['episode'] + 1} of {s['total']}" + ("  (saving previous...)" if s["saving"] else ""))
        self.target.setText(f"Place the cube {s['target_text']}" if t else "")
        if t:
            sign = lambda v: f"{'+' if v > 0 else ''}{v:.1f}"  # noqa: E731
            self.offsets.setText(f"forward {sign(t['forward_cm'])} cm  (+ = away from the arm)     "
                                 f"left {sign(t['left_cm'])} cm  (+ = left, seen from behind the arm)     "
                                 f"rotation {sign(t['yaw_deg'])}°  (+ = counter-clockwise from above)")
        ok = "OK" in (s["cube"] or "")
        self.cube.setText(f"<span style='color:{'#2ecc71' if ok else '#ff6b6b'}'>{s['cube']}</span>")
        self.home_hint.setText("" if s["has_home"] else
                               "Once: put the cube on your mark and click \"Cube is on the mark\" so offsets are measured from it.")
        r = s["rest"]
        if not s["has_rest"]:
            self.rest.setText("<span style='color:#ff6b6b'>No rest pose yet: fold the arm and click \"Set rest pose\".</span>")
        elif r:
            rows = []
            for jn in JOINTS:
                d = r["diffs"][jn]
                tol = REST_TOL_GRIPPER if jn == "gripper" else REST_TOL_DEG
                rows.append(f"<tr><td>{jn}</td><td style='color:{'#2ecc71' if abs(d) <= tol else '#ff6b6b'}'>"
                            f"{'+' if d > 0 else ''}{d:.1f}</td></tr>")
            rows.append(f"<tr><td colspan=2><b style='color:{'#2ecc71' if r['ok'] else '#ff6b6b'}'>"
                        f"{'At rest pose' if r['ok'] else 'Not at rest pose yet'}</b></td></tr>")
            self.rest.setText("<table>" + "".join(rows) + "</table>")
        st = s["state"]
        self.b_start.setEnabled(st == "ready" and not s["saving"] and bool(t))
        self.b_save.setEnabled(st in ("recording", "review"))
        self.b_discard.setEnabled(st in ("recording", "review"))
        self.b_stop.setEnabled(st == "ready")
        self.b_pause.setEnabled(st == "ready")
        self.b_pause.setText("Resume teleop" if s["paused"] else "Pause teleop")
        if s["paused"]:
            self.b_start.setEnabled(False)

    def closeEvent(self, e):
        if self.app.state in ("recording", "review"):
            e.ignore()
            QtWidgets.QMessageBox.warning(self, "Recorder", "Save or Discard the current episode first.")
            return
        if self.app.state not in ("setup", "stopped", "connecting"):
            e.ignore()
            self.app.command("stop")  # the window closes itself once the dataset is finished
            return
        e.accept()


def main():
    a = build_parser().parse_args()
    a.cameras = "opencv"
    qapp = QtWidgets.QApplication(sys.argv)
    app = App(a)
    threading.Thread(target=app.probe_ports, daemon=True).start()
    threading.Thread(target=app.probe_cameras_until_ok, daemon=True).start()
    w = Window(app)
    w.show()
    qapp.exec()


if __name__ == "__main__":
    main()
