#!/usr/bin/env python3
"""Camera calibration for the SO-101 setup: intrinsics + extrinsics from a ChArUco board.

Run with:  python lerobot-tabletop/camera_calibration/calib.py <command> ...

  board                         write the printable board (board.pdf / board.png)
  capture   --name cam1         collect calibration views (live preview on http://localhost:8000)
  intrinsics --name cam1        compute camera matrix + distortion  -> results/cam1_intrinsics.json
  extrinsics --name cam1        camera pose from the board lying flat -> results/cam1_extrinsics.json
  selftest                      check the maths on synthetic images (no camera needed)

Frames used here
  board/table frame: origin at the board's bottom-left corner (text upright),
                     +x right along the long edge, +y up along the short edge, +z up out of the table.
  camera (OpenCV):   +x right, +y down, +z forward (out of the lens).
  camera (MuJoCo/OpenGL): +x right, +y up, camera looks along -z.
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

SQ_X, SQ_Y = 7, 5  # squares along the long / short edge
SQUARE_MM, MARKER_RATIO = 35.0, 26.0 / 35.0
DICT = cv2.aruco.DICT_5X5_100
HERE = os.path.dirname(os.path.abspath(__file__))
CAPDIR, RESDIR = os.path.join(HERE, "captures"), os.path.join(HERE, "results")


# ---------------------------------------------------------------- board + detection
def make_board(square_mm=SQUARE_MM):
    s = square_mm / 1000.0
    return cv2.aruco.CharucoBoard(
        (SQ_X, SQ_Y), s, s * MARKER_RATIO, cv2.aruco.getPredefinedDictionary(DICT)
    )


def detect(gray, board):
    """Return (corners Nx1x2, ids Nx1) or (None, None)."""
    c, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None, None
    return c, ids


IS_MAC = sys.platform == "darwin"
DEFAULT_DEVICE = "0" if IS_MAC else "/dev/video0"


def open_camera(device, width, height):
    """Open a camera: a /dev/videoN path or index on Linux (V4L2), an index on macOS (AVFoundation)."""
    if IS_MAC:
        if not str(device).isdigit():
            sys.exit(f"On macOS --device is a camera index (0, 1, ...), not '{device}'. Run 'calib.py cameras' to see which is which.")
        cap = cv2.VideoCapture(int(device), cv2.CAP_AVFOUNDATION)
    else:
        dev = os.path.realpath(device) if not str(device).isdigit() else int(device)
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(
            f"Could not open camera {device}. Is another program using it (a browser tab with the camera page, "
            f"camera_stream.py, show_cameras.sh)? On macOS, also allow camera access for the terminal app."
        )
    frame = None
    for _ in range(10):
        ok, f = cap.read()
        frame = f if ok else frame
    if frame is None:
        sys.exit(f"Camera {device} opened but gave no frames.")
    h, w = frame.shape[:2]
    if (w, h) != (width, height):
        # A different mode means a different crop of the sensor, so the calibration would not match.
        sys.exit(f"Camera {device} gives {w}x{h}, not the requested {width}x{height}.")
    return cap


def cmd_cameras(a):
    """Save one frame from each camera index so you can see which index is which camera."""
    out = os.path.join(CAPDIR, "_which_camera")
    os.makedirs(out, exist_ok=True)
    backend = cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2
    for i in range(a.max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            frame = f if ok else frame
        cap.release()
        if frame is None:
            print(f"  index {i}: no camera")
            continue
        p = os.path.join(out, f"index{i}.jpg")
        cv2.imwrite(p, frame)
        print(f"  index {i}: {frame.shape[1]}x{frame.shape[0]} -> {p}")


# ---------------------------------------------------------------- maths helpers
def quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [
            s / 4,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
        q = [0, 0, 0, 0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i], q[1 + j], q[1 + k] = (
            s / 4,
            (R[j, i] + R[i, j]) / s,
            (R[k, i] + R[i, k]) / s,
        )
    q = np.array(q)
    q /= np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def m_pos(T):
    return [round(float(v), 5) for v in T[:3, 3]]


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def table_from_boardcv(square_mm):
    """OpenCV board frame (origin top-left, y down, z into the table) -> table frame (see module doc)."""
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[1, 3] = SQ_Y * square_mm / 1000.0
    return T


def fov_deg(K, w, h):
    return (
        2 * math.degrees(math.atan(w / (2 * K[0][0]))),
        2 * math.degrees(math.atan(h / (2 * K[1][1]))),
    )


def load_intrinsics(name, size):
    """Load results/<name>_intrinsics.json, scaling K if the image size differs (same aspect ratio only)."""
    p = os.path.join(RESDIR, f"{name}_intrinsics.json")
    if not os.path.exists(p):
        sys.exit(
            f"No intrinsics for '{name}' yet - run:  calib.py intrinsics --name {name}"
        )
    d = json.load(open(p))
    K, dist = np.array(d["camera_matrix"]), np.array(d["distortion"])
    w, h = size
    if (w, h) != (d["width"], d["height"]):
        if abs(w / h - d["width"] / d["height"]) > 1e-3:
            sys.exit(
                f"Image is {w}x{h} but intrinsics are for {d['width']}x{d['height']} (different aspect ratio)."
            )
        s = w / d["width"]
        K = K.copy()
        K[:2] *= s
        print(f"note: scaled intrinsics from {d['width']}x{d['height']} to {w}x{h}")
    return K, dist, d


# ---------------------------------------------------------------- board
def cmd_board(a):
    px = 420  # pixels per square
    dpi = px / SQUARE_MM * 25.4  # so one square prints at exactly SQUARE_MM
    img = make_board().generateImage((SQ_X * px, SQ_Y * px), marginSize=0, borderBits=1)
    pw, ph = (
        int(round(11 * dpi)),
        int(round(8.5 * dpi)),
    )  # US Letter landscape (also fits A4)
    page = np.full((ph, pw), 255, np.uint8)
    x0, y0 = (pw - img.shape[1]) // 2, (ph - img.shape[0]) // 2 - 40
    page[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    yb, f = y0 + img.shape[0], cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(page, (x0, yb), 14, 0, -1)  # origin marker
    cv2.arrowedLine(page, (x0, yb + 45), (x0 + 300, yb + 45), 0, 6, tipLength=0.15)
    cv2.putText(page, "+x", (x0 + 320, yb + 62), f, 1.6, 0, 4)
    cv2.arrowedLine(page, (x0 - 45, yb), (x0 - 45, yb - 300), 0, 6, tipLength=0.15)
    cv2.putText(page, "+y", (x0 - 110, yb - 320), f, 1.6, 0, 4)
    cv2.putText(page, "origin", (x0 - 60, yb + 110), f, 1.3, 0, 3)
    mm100 = int(round(100 * px / SQUARE_MM))
    xr = x0 + img.shape[1] - mm100
    cv2.line(page, (xr, yb + 60), (xr + mm100, yb + 60), 0, 5)
    for x in (xr, xr + mm100):
        cv2.line(page, (x, yb + 40), (x, yb + 80), 0, 5)
    cv2.putText(
        page, "this line must measure 100 mm", (xr - 60, yb + 130), f, 1.3, 0, 3
    )
    cv2.putText(
        page,
        f"ChArUco {SQ_X}x{SQ_Y}, DICT_5X5_100, square {SQUARE_MM:g} mm. Print at 100% / actual size (no 'fit to page').",
        (x0, y0 - 40),
        f,
        1.3,
        0,
        3,
    )
    cv2.imwrite(os.path.join(HERE, "board.png"), page)
    from PIL import Image

    Image.fromarray(page).save(os.path.join(HERE, "board.pdf"), resolution=dpi)
    print(
        f"wrote {HERE}/board.pdf and board.png  (print at 100% scale; squares should measure {SQUARE_MM:g} mm)"
    )


# ---------------------------------------------------------------- capture
class Preview:
    """Tiny MJPEG server so the capture can be watched from a browser over an SSH tunnel."""

    def __init__(self, port):
        self.jpg, self.cond = None, threading.Condition()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *x):
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<body style='margin:0;background:#111'><img src='/s' style='width:100%'>"
                    )
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=f"
                )
                self.end_headers()
                try:
                    while True:
                        with outer.cond:
                            outer.cond.wait(2)
                            j = outer.jpg
                        if j:
                            self.wfile.write(
                                b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                % len(j)
                                + j
                                + b"\r\n"
                            )
                except (BrokenPipeError, ConnectionResetError):
                    pass

        ThreadingHTTPServer.daemon_threads = True
        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def show(self, img):
        ok, j = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpg = j.tobytes()
                self.cond.notify_all()


def signature(c, w):
    pts = c.reshape(-1, 2)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    return np.array(
        [cx / w, cy / w, math.sqrt(max(rw * rh, 1)) / w, (ang % 90) / 90 * 0.3]
    )


def cmd_capture(a):
    board = make_board(a.square_mm)
    out = os.path.join(CAPDIR, a.name)
    os.makedirs(out, exist_ok=True)
    n = len(glob.glob(os.path.join(out, "*.png")))
    cap = open_camera(a.device, a.width, a.height)
    prev = Preview(a.port)
    print(
        f"Capturing '{a.name}' from {a.device} -> {out}\nPreview: http://localhost:{a.port}  "
        f"(from a laptop: ssh -L {a.port}:localhost:{a.port} ...)\n"
        f"Move the board slowly: centre, all four corners of the image, near and far, tilted each way. "
        f"A view is saved automatically each time the board is held still in a new pose. Target: {a.count} views.",
        flush=True,
    )
    sigs, covered, last, last_save = [], None, None, 0
    try:
        while n < a.count:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            h, w = frame.shape[:2]
            if covered is None:
                covered = np.zeros((h, w), np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            c, ids = detect(gray, board)
            vis, status = frame.copy(), "no board"
            if c is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, c, ids, (0, 255, 0))
                cur = {int(i): p for i, p in zip(ids.ravel(), c.reshape(-1, 2))}
                still = False
                if last:
                    common = [k for k in cur if k in last]
                    if len(common) >= 6:
                        still = (
                            np.mean([np.linalg.norm(cur[k] - last[k]) for k in common])
                            < w / 900
                        )
                last = cur
                sig = signature(c, w)
                novel = all(np.linalg.norm(sig - s) > 0.07 for s in sigs)
                status = (
                    f"{len(ids)} corners"
                    + ("" if len(ids) >= a.min_corners else " (too few)")
                    + ("" if still else " - hold still")
                    + ("" if novel else " - move to a new pose")
                )
                if (
                    len(ids) >= a.min_corners
                    and still
                    and novel
                    and time.time() - last_save > 0.7
                ):
                    cv2.imwrite(os.path.join(out, f"{n:03d}.png"), frame)
                    n += 1
                    sigs.append(sig)
                    last_save = time.time()
                    cv2.fillConvexPoly(
                        covered, cv2.convexHull(c.reshape(-1, 2).astype(np.int32)), 255
                    )
                    print(
                        f"  saved view {n}/{a.count}  ({len(ids)} corners, image coverage {covered.mean() / 2.55:.0f}%)",
                        flush=True,
                    )
            else:
                last = None
            tint = vis.copy()
            tint[covered > 0] = (0, 160, 0)
            vis = cv2.addWeighted(vis, 0.8, tint, 0.2, 0)
            # Draw the status on the preview copy at 960 px wide so it stays legible at 640x480.
            vis = cv2.resize(vis, (960, 960 * h // w))
            if time.time() - last_save < 0.3:
                cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 0), 20)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(
                vis,
                f"{a.name}: {n}/{a.count} saved | {status} | green = area covered",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            prev.show(vis)
    except KeyboardInterrupt:
        pass
    cap.release()
    print(f"Done: {n} views in {out}. Next:  calib.py intrinsics --name {a.name}")


# ---------------------------------------------------------------- intrinsics
def calibrate(views, board, size, rational=False, free_k3=False):
    """views: list of (label, corners, ids). Returns dict of results."""

    def run(vs):
        obj, img = zip(*[board.matchImagePoints(c, i) for _, c, i in vs])
        flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
        if not rational and not free_k3:
            # k3 is poorly constrained unless the board reaches the image corners, and a free k3
            # can make the distortion curve fold back there; k1, k2 extrapolate sanely.
            flags |= cv2.CALIB_FIX_K3
        rms, K, dist, rv, tv = cv2.calibrateCamera(
            obj, img, size, None, None, flags=flags
        )
        errs = [
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (cv2.projectPoints(o, r, t, K, dist)[0] - p) ** 2, axis=2
                        )
                    )
                )
            )
            for o, p, r, t in zip(obj, img, rv, tv)
        ]
        return rms, K, dist, errs

    rms, K, dist, errs = run(views)
    med = float(np.median(errs))
    keep = [v for v, e in zip(views, errs) if e <= max(3 * med, 1.0)]
    dropped = [v[0] for v, e in zip(views, errs) if e > max(3 * med, 1.0)]
    if dropped and len(keep) >= 8:
        views = keep
        rms, K, dist, errs = run(views)
    else:
        dropped = []
    return dict(
        rms=float(rms),
        K=K,
        dist=dist.ravel(),
        errs=errs,
        used=[v[0] for v in views],
        dropped=dropped,
    )


def cmd_intrinsics(a):
    board = make_board(a.square_mm)
    files = sorted(glob.glob(os.path.join(CAPDIR, a.name, "*.png")))
    if not files:
        sys.exit(
            f"No captures in {CAPDIR}/{a.name} - run:  calib.py capture --name {a.name}"
        )
    views, size, cover = [], None, None
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = size or (g.shape[1], g.shape[0])
        if (g.shape[1], g.shape[0]) != size:
            sys.exit(f"{f} has a different size from the other captures")
        if cover is None:
            cover = np.zeros(g.shape, np.uint8)
        c, ids = detect(g, board)
        if c is not None and len(ids) >= 8:
            views.append((os.path.basename(f), c, ids))
            cv2.fillConvexPoly(
                cover, cv2.convexHull(c.reshape(-1, 2).astype(np.int32)), 255
            )
    if len(views) < 10:
        sys.exit(
            f"Only {len(views)} usable views (need at least 10, ideally 25+). Capture more."
        )
    r = calibrate(views, board, size, a.rational, a.free_k3)
    w, h = size
    fx, fy = fov_deg(r["K"], w, h)
    coverage = cover.mean() / 2.55
    # The distortion must map larger angles to larger image radii all the way to the image corners,
    # otherwise undistorting near the corners gives nonsense.
    corners = np.array([[[0, 0]], [[w - 1, 0]], [[0, h - 1]], [[w - 1, h - 1]]], np.float32)
    r_max = float(np.max(np.linalg.norm(cv2.undistortPoints(corners, r["K"], r["dist"]).reshape(-1, 2), axis=1)))
    r_u = np.linspace(0, r_max, 400)
    r_u2 = cv2.projectPoints(
        np.stack([r_u, np.zeros_like(r_u), np.ones_like(r_u)], 1), np.zeros(3), np.zeros(3), np.eye(3), r["dist"]
    )[0].reshape(-1, 2)[:, 0]
    monotonic = bool(np.all(np.diff(r_u2) > 0))
    res = dict(
        name=a.name,
        width=w,
        height=h,
        model="opencv_rational8" if a.rational else ("opencv_plumb_bob5" if a.free_k3 else "opencv_plumb_bob5_k3_fixed_0"),
        camera_matrix=r["K"].tolist(),
        distortion=r["dist"].tolist(),
        fx=r["K"][0, 0],
        fy=r["K"][1, 1],
        cx=r["K"][0, 2],
        cy=r["K"][1, 2],
        fov_x_deg=fx,
        fov_y_deg=fy,
        rms_reprojection_px=r["rms"],
        views_used=len(r["used"]),
        views_dropped=r["dropped"],
        image_coverage_pct=coverage,
        distortion_monotonic_to_corners=monotonic,
        square_mm=a.square_mm,
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    if (
        abs(w / h - 4 / 3) < 1e-3
    ):  # the sim renders 640x480; 4:3 modes of this camera share one field of view
        k = 640 / w
        res["sim_640x480"] = dict(
            fx=res["fx"] * k,
            fy=res["fy"] * k,
            cx=res["cx"] * k,
            cy=res["cy"] * k,
            fovy_deg=fy,
            principal_offset_px=[res["cx"] * k - 320, res["cy"] * k - 240],
            note="MuJoCo has no lens distortion: undistort real images with camera_matrix+distortion "
            "(scaled to the stream size) before comparing with sim renders",
        )
    else:
        print(
            "  WARNING: not a 4:3 mode - the sim uses 640x480, whose field of view differs from the 16:9 modes."
        )
    os.makedirs(RESDIR, exist_ok=True)
    p = os.path.join(RESDIR, f"{a.name}_intrinsics.json")
    json.dump(res, open(p, "w"), indent=2)
    print(
        f"{a.name}: {w}x{h}, {len(r['used'])} views used, {len(r['dropped'])} dropped"
    )
    print(
        f"  fx={res['fx']:.1f} fy={res['fy']:.1f} cx={res['cx']:.1f} cy={res['cy']:.1f}"
    )
    print(f"  distortion={np.round(r['dist'], 4).tolist()}")
    print(
        f"  FOV: {fx:.1f} deg horizontal, {fy:.1f} deg vertical (MuJoCo fovy={fy:.2f})"
    )
    print(
        f"  RMS reprojection error: {r['rms']:.3f} px   image coverage: {coverage:.0f}%"
    )
    if r["rms"] > 1.0:
        print(
            "  WARNING: RMS above 1 px - check the board is flat and the captures are sharp."
        )
    if coverage < 60:
        print(
            "  WARNING: the board covered less than 60% of the image - distortion near the edges is poorly constrained."
        )
    if not monotonic:
        print(
            "  WARNING: the distortion model folds back before the image corners - undistorting there is invalid. "
            "Capture views in the corners, or drop --free-k3."
        )
    print(f"saved {p}")


# ---------------------------------------------------------------- extrinsics
def solve_pose(gray, board, K, dist):
    c, ids = detect(gray, board)
    if c is None:
        return None
    obj, img = board.matchImagePoints(c, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    err = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (cv2.projectPoints(obj, rvec, tvec, K, dist)[0] - img) ** 2, axis=2
                )
            )
        )
    )
    return rvec, tvec, err, len(ids)


def camera_pose(rvec, tvec, square_mm, board_in_base):
    """Camera pose (4x4, OpenCV camera axes) in the base frame."""
    T_cam_board = T_from(cv2.Rodrigues(rvec)[0], tvec)
    T_table_cam = table_from_boardcv(square_mm) @ np.linalg.inv(T_cam_board)
    x, y, z, yaw = board_in_base
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    T_base_table = T_from(np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]), [x, y, z])
    return T_base_table @ T_table_cam


def cmd_extrinsics(a):
    board = make_board(a.square_mm)
    if a.image:
        frames = [cv2.imread(p) for p in a.image]
        missing = [p for p, f in zip(a.image, frames) if f is None]
        if missing:
            sys.exit(f"Could not read {', '.join(missing)}")
    else:
        cap = open_camera(a.device, a.width, a.height)
        frames = [cap.read()[1] for _ in range(a.frames)]
        cap.release()
    h, w = frames[0].shape[:2]
    K, dist, intr = load_intrinsics(a.intrinsics or a.name, (w, h))
    sols = [
        (f, solve_pose(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), board, K, dist))
        for f in frames
    ]
    sols = [(f, s) for f, s in sols if s]
    if not sols:
        sys.exit(
            "Board not found. Make sure the whole board is visible, flat, well lit and not too small in the image."
        )
    Ts = [camera_pose(s[0], s[1], a.square_mm, a.board_in_base) for _, s in sols]
    pos = np.array([T[:3, 3] for T in Ts])
    frame, (rvec, tvec, err, ncorn) = sols[len(sols) // 2]
    T = Ts[len(sols) // 2].copy()
    T[:3, 3] = np.median(pos, axis=0)
    R_cv = T[:3, :3]
    R_gl = R_cv @ np.diag([1.0, -1.0, -1.0])
    fx, fy = fov_deg(K, w, h)
    res = dict(
        name=a.name,
        frame="sim world"
        if any(a.board_in_base)
        else "board/table (origin = board bottom-left corner, x right, y up, z up)",
        board_in_world_xyz_m_yaw_deg=list(a.board_in_base),
        position_m=T[:3, 3].tolist(),
        opencv=dict(
            note="camera axes: x right, y down, z forward",
            rotation_matrix=R_cv.tolist(),
            quat_wxyz=quat_wxyz(R_cv),
        ),
        mujoco=dict(
            note="camera axes: x right, y up, looks along -z",
            pos=T[:3, 3].tolist(),
            quat_wxyz=quat_wxyz(R_gl),
            xyaxes=np.concatenate([R_gl[:, 0], R_gl[:, 1]]).tolist(),
            fovy_deg=fy,
            resolution=[w, h],
        ),
        reprojection_px=err,
        corners_used=ncorn,
        frames_used=len(sols),
        position_spread_mm=(pos.std(axis=0) * 1000).tolist(),
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    res["sim_config_overhead_camera"] = dict(
        pos=m_pos(T), xyaxes=res["mujoco"]["xyaxes"], fovy=round(fy, 3)
    )
    os.makedirs(RESDIR, exist_ok=True)
    p = os.path.join(RESDIR, f"{a.name}_extrinsics.json")
    json.dump(res, open(p, "w"), indent=2)
    vis = frame.copy()
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, a.square_mm / 1000 * 3, 4)
    cv2.imwrite(os.path.join(RESDIR, f"{a.name}_extrinsics_check.jpg"), vis)
    m = res["mujoco"]
    print(
        f"{a.name} pose in the {res['frame']} frame  ({ncorn} corners, reprojection {err:.2f} px, {len(sols)} frames)"
    )
    print(
        f"  position (m): {np.round(T[:3, 3], 4).tolist()}   spread over frames (mm): {np.round(res['position_spread_mm'], 2).tolist()}"
    )
    print(f"  OpenCV quat (wxyz): {np.round(res['opencv']['quat_wxyz'], 5).tolist()}")
    print(
        f'  MuJoCo: <camera name="{a.name}" pos="{" ".join(f"{v:.4f}" for v in m["pos"])}" '
        f'quat="{" ".join(f"{v:.5f}" for v in m["quat_wxyz"])}" fovy="{fy:.2f}"/>'
    )
    print(
        "  rl-so101 mounts/config.json overhead_camera: "
        + json.dumps(res["sim_config_overhead_camera"])
    )
    print(f"saved {p}  (+ {a.name}_extrinsics_check.jpg with the board axes drawn)")


# ---------------------------------------------------------------- selftest
def cmd_selftest(a):
    """Render the board through a known camera, then check calibration and pose recovery."""
    rng = np.random.default_rng(0)
    board, px = make_board(), 200
    img = board.generateImage((SQ_X * px, SQ_Y * px), marginSize=0, borderBits=1)
    s = SQUARE_MM / 1000 / px  # metres per board-image pixel
    w, h = 1280, 720
    K = np.array([[900.0, 0, 655.0], [0, 905.0, 350.0], [0, 0, 1]])
    views, truth = [], []
    for i in range(40):
        rv = np.array(
            [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.5, 0.5)]
        )
        R = cv2.Rodrigues(rv)[0]
        centre = np.array([SQ_X, SQ_Y, 0]) * SQUARE_MM / 2000
        t = (
            np.array(
                [
                    rng.uniform(-0.12, 0.12),
                    rng.uniform(-0.07, 0.07),
                    rng.uniform(0.3, 0.55),
                ]
            )
            - R @ centre
        )
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.diag([s, s, 1])
        view = cv2.warpPerspective(img, H, (w, h), borderValue=128)
        c, ids = detect(view, board)
        if c is not None and len(ids) >= 8:
            views.append((str(i), c, ids))
            truth.append((R, t))
    r = calibrate(views, board, (w, h))
    dK = np.abs(r["K"] - K)
    print(
        f"intrinsics: {len(views)} views, rms {r['rms']:.3f}px, errors fx {dK[0, 0]:.2f} fy {dK[1, 1]:.2f} cx {dK[0, 2]:.2f} cy {dK[1, 2]:.2f} px"
    )
    ok = (
        r["rms"] < 0.5
        and dK[0, 0] < 5
        and dK[1, 1] < 5
        and dK[0, 2] < 5
        and dK[1, 2] < 5
    )
    # pose: camera 0.6 m above the table looking straight down at the board centre, image x aligned with board x
    R = np.eye(3)
    cam_table = np.array([0.20, 0.05, 0.60])
    # looking straight down with image-x = table-x: board-cv axes equal camera axes
    cam_boardcv = np.array(
        [cam_table[0], SQ_Y * SQUARE_MM / 1000 - cam_table[1], -cam_table[2]]
    )
    t = -R @ cam_boardcv
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.diag([s, s, 1])
    view = cv2.warpPerspective(img, H, (w, h), borderValue=128)
    sol = solve_pose(view, board, K, np.zeros(5))
    T = camera_pose(sol[0], sol[1], SQUARE_MM, (0, 0, 0, 0))
    perr = np.linalg.norm(T[:3, 3] - cam_table) * 1000
    R_gl = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    print(
        f"extrinsics: camera recovered at {np.round(T[:3, 3], 4).tolist()} (truth {cam_table.tolist()}), error {perr:.2f} mm; "
        f"MuJoCo quat {np.round(quat_wxyz(R_gl), 3).tolist()} (expect [1,0,0,0])"
    )
    ok = ok and perr < 8 and abs(quat_wxyz(R_gl)[0]) > 0.999
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cam(p):
        p.add_argument(
            "--name",
            required=True,
            help="label for this camera, e.g. cam1_birdseye or cam2_wrist",
        )
        p.add_argument(
            "--device",
            default=DEFAULT_DEVICE,
            help="camera index on macOS (see 'calib.py cameras'), /dev/videoN on Linux",
        )
        # 640x480 is the sim's resolution. Calibrate in the same mode the policy uses:
        # other modes crop the sensor differently, so their lens values do not carry over.
        p.add_argument("--width", type=int, default=640)
        p.add_argument("--height", type=int, default=480)
        p.add_argument(
            "--square-mm",
            type=float,
            default=SQUARE_MM,
            help="measured size of one printed square",
        )

    sub.add_parser("board").set_defaults(fn=cmd_board)
    p = sub.add_parser("cameras", help="save one frame per camera index to see which is which")
    p.add_argument("--max-index", type=int, default=3)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.set_defaults(fn=cmd_cameras)
    p = sub.add_parser("capture")
    cam(p)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--min-corners", type=int, default=12)
    p.add_argument("--port", type=int, default=8001)
    p.set_defaults(fn=cmd_capture)
    p = sub.add_parser("intrinsics")
    p.add_argument("--name", required=True)
    p.add_argument("--square-mm", type=float, default=SQUARE_MM)
    p.add_argument(
        "--rational",
        action="store_true",
        help="8-coefficient distortion model for very wide lenses",
    )
    p.add_argument(
        "--free-k3",
        action="store_true",
        help="also fit k3 (default keeps k3 = 0; only use when the board covered the image corners)",
    )
    p.set_defaults(fn=cmd_intrinsics)
    p = sub.add_parser("extrinsics")
    cam(p)
    p.add_argument(
        "--intrinsics", help="name of the intrinsics to use (default: same as --name)"
    )
    p.add_argument(
        "--image",
        nargs="+",
        help="use saved image(s) instead of the live camera, e.g. captures/overhead_extrinsics/*.png",
    )
    p.add_argument("--frames", type=int, default=15)
    p.add_argument(
        "--board-in-world",
        dest="board_in_base",
        type=float,
        nargs=4,
        default=[0, 0, 0, 0],
        metavar=("X", "Y", "Z", "YAW_DEG"),
        help="pose of the board origin in the sim world frame (metres; yaw in degrees about +z, "
        "0 = board +x along world +x)",
    )
    p.set_defaults(fn=cmd_extrinsics)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
