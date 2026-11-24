#!/usr/bin/env python3
"""Capture calibration views through a browser tab (for when Python cannot open the camera itself).

The browser opens the camera at its native 640x480 mode and sends frames here; this server runs
the same board detection and view selection as `calib.py capture` and saves the views to the same
folders, so `calib.py intrinsics` and `calib.py extrinsics --image ...` work on them unchanged.

  python lerobot-tabletop/camera_calibration/browser_capture.py --name overhead
  then open http://localhost:8001 in a browser that has camera access (e.g. Chrome)

Two modes on the page:
  Intrinsics: move the board around; a view is saved each time it is held still in a new pose.
  Extrinsics: with the board lying flat, "Save extrinsics frames" saves 15 frames to
              captures/<name>_extrinsics/ for  calib.py extrinsics --image <those files>.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib import CAPDIR, RESDIR, SQ_Y, SQUARE_MM, detect, make_board, signature  # noqa: E402

PAGE = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calibration capture</title>
<style>
 body{margin:0;background:#111;color:#eee;font:15px system-ui,sans-serif}
 header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:8px 12px}
 select,button{font:inherit;padding:4px 10px}
 #wrap{position:relative;max-width:960px;margin:0 auto}
 video,canvas{width:100%;display:block}
 #ov{position:absolute;inset:0}
 #status{font-size:20px;font-weight:700;padding:8px 12px}
 #warn{color:#ff8080;padding:0 12px}
</style>
<header>
 <b>__NAME__</b>
 <select id="dev"></select>
 <label><input type="radio" name="mode" value="intrinsics" checked> Intrinsics (move the board)</label>
 <label><input type="radio" name="mode" value="extrinsics"> Extrinsics (board flat)</label>
 <button id="ext" disabled>Save extrinsics frames</button>
 <span id="mode"></span>
</header>
<div id="status">starting...</div><div id="warn"></div>
<div id="wrap"><video id="v" autoplay playsinline muted></video><canvas id="ov" width="640" height="480"></canvas></div>
<script>
const v=document.getElementById("v"), ov=document.getElementById("ov"), octx=ov.getContext("2d"),
      dev=document.getElementById("dev"), st=document.getElementById("status"),
      warn=document.getElementById("warn"), ext=document.getElementById("ext");
const grab=document.createElement("canvas"); grab.width=640; grab.height=480;
const gctx=grab.getContext("2d");
let stream, busy=false, flash=0, extLeft=0;

async function start(){
  if(stream) stream.getTracks().forEach(t=>t.stop());
  const video={width:{exact:640},height:{exact:480},resizeMode:"none"};
  if(dev.value) video.deviceId={exact:dev.value};
  try{ stream=await navigator.mediaDevices.getUserMedia({video,audio:false}); }
  catch(e){ st.textContent="Camera error: "+e.name+" "+e.message; return; }
  v.srcObject=stream;
  const s=stream.getVideoTracks()[0].getSettings();
  document.getElementById("mode").textContent=`${stream.getVideoTracks()[0].label} ${s.width}x${s.height} resize=${s.resizeMode}`;
  warn.textContent=(s.width!==640||s.height!==480)?"Not 640x480 - pick the right camera.":
                   (s.resizeMode && s.resizeMode!=="none")?"Browser is rescaling the image - calibration would not match the camera's own 640x480 mode.":"";
  const cams=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==="videoinput");
  dev.innerHTML=cams.map(c=>`<option value="${c.deviceId}">${c.label||"camera"}</option>`).join("");
  dev.value=s.deviceId;
  return cams;
}
dev.onchange=start;
document.querySelectorAll("input[name=mode]").forEach(r=>r.onchange=()=>{ext.disabled=r.value!=="extrinsics"||!r.checked;});
ext.onclick=()=>{extLeft=15; ext.disabled=true;};

// Coverage = union of saved views, drawn once as a single light layer (stacking each view's
// translucent hull turned it solid green and hid the board and the live corners).
const cov=document.createElement("canvas"); cov.width=640; cov.height=480;
const cctx=cov.getContext("2d"); let covCount=-1;
function draw(r){
  octx.clearRect(0,0,640,480);
  if(r.hull_saved){
    if(r.hull_saved.length!==covCount){
      covCount=r.hull_saved.length; cctx.clearRect(0,0,640,480); cctx.fillStyle="#0c0";
      r.hull_saved.forEach(h=>{cctx.beginPath();h.forEach(([x,y],i)=>i?cctx.lineTo(x,y):cctx.moveTo(x,y));cctx.closePath();cctx.fill();});
    }
    octx.globalAlpha=0.22; octx.drawImage(cov,0,0); octx.globalAlpha=1;
  }
  if(r.corners && !r.lines){
    octx.fillStyle="#ff00ff"; octx.strokeStyle="#fff"; octx.lineWidth=1.5;
    r.corners.forEach(([x,y])=>{octx.beginPath();octx.arc(x,y,5,0,7);octx.fill();octx.stroke();});
  }
  if(r.lines){
    r.lines.forEach(l=>{
      if(l.dot){ const [x,y]=l.pts[0]; octx.fillStyle=l.color; octx.beginPath(); octx.arc(x,y,l.dot,0,7); octx.fill(); return; }
      if(l.pts.length<2) return;
      octx.strokeStyle="#000"; octx.lineWidth=l.w+2; octx.beginPath();
      l.pts.forEach(([x,y],i)=>i?octx.lineTo(x,y):octx.moveTo(x,y)); octx.stroke();
      octx.strokeStyle=l.color; octx.lineWidth=l.w; octx.beginPath();
      l.pts.forEach(([x,y],i)=>i?octx.lineTo(x,y):octx.moveTo(x,y)); octx.stroke();
    });
    octx.font="bold 13px system-ui"; octx.lineWidth=3;
    (r.texts||[]).forEach(t=>{ octx.strokeStyle="#000"; octx.strokeText(t.t,t.x,t.y); octx.fillStyle=t.color; octx.fillText(t.t,t.x,t.y); });
  }
  if(Date.now()-flash<300){ octx.strokeStyle="#0f0"; octx.lineWidth=16; octx.strokeRect(0,0,640,480); }
}

async function loop(){
  if(!busy && v.videoWidth===640 && v.videoHeight===480){
    busy=true;
    gctx.drawImage(v,0,0,640,480);
    const blob=await new Promise(r=>grab.toBlob(r,"image/png"));
    const mode=document.querySelector("input[name=mode]:checked").value;
    const save=mode==="extrinsics" && extLeft>0;
    try{
      const res=await fetch(`/frame?mode=${mode}&save=${save?1:0}`,{method:"POST",body:blob});
      const r=await res.json();
      if(r.saved) flash=Date.now();
      if(save && r.saved){ extLeft--; if(!extLeft) ext.disabled=false; }
      st.textContent=r.status; draw(r);
    }catch(e){ st.textContent="Server not reachable: "+e; }
    busy=false;
  }
  setTimeout(loop,150);
}

(async()=>{
  const cams=await start();
  const inno=(cams||[]).find(c=>/innomaker|U20CAM|UVC/i.test(c.label));
  if(inno && inno.deviceId!==dev.value){ dev.value=inno.deviceId; await start(); }
  loop();
})();
</script>
"""


class Capture:
    def __init__(self, a):
        self.a = a
        self.board = make_board(a.square_mm)
        self.out = os.path.join(CAPDIR, a.name)
        self.ext_out = os.path.join(CAPDIR, a.name + "_extrinsics")
        os.makedirs(self.out, exist_ok=True)
        self.n = len(glob.glob(os.path.join(self.out, "*.png")))
        self.sigs, self.hulls, self.last, self.last_save = [], [], None, 0.0
        self.covered = np.zeros((480, 640), np.uint8)
        # Views already on disk count towards coverage and novelty.
        for f in sorted(glob.glob(os.path.join(self.out, "*.png"))):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            c, ids = detect(g, self.board)
            if c is not None:
                self._remember(c, g.shape[1])
        self.lock = threading.Lock()

    def _remember(self, c, w):
        self.sigs.append(signature(c, w))
        hull = cv2.convexHull(c.reshape(-1, 2).astype(np.int32))
        cv2.fillConvexPoly(self.covered, hull, 255)
        self.hulls.append(hull.reshape(-1, 2).tolist())

    def frame(self, png, mode, save):
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "bad frame"}
        h, w = img.shape[:2]
        if (w, h) != (640, 480):
            return {"status": f"frame is {w}x{h}, expected 640x480"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c, ids = detect(gray, self.board)
        with self.lock:
            if mode == "extrinsics":
                return self._extrinsics(img, c, ids, save)
            return self._intrinsics(img, c, ids, w)

    def _intrinsics(self, img, c, ids, w):
        a = self.a
        res = {"hull_saved": self.hulls, "saved": False}
        if self.n >= a.count:
            res["status"] = f"Done: {self.n} views. Next: calib.py intrinsics --name {a.name}"
            return res
        if c is None:
            self.last = None
            res["status"] = f"{self.n}/{a.count} saved | no board"
            return res
        res["corners"] = c.reshape(-1, 2).tolist()
        cur = {int(i): p for i, p in zip(ids.ravel(), c.reshape(-1, 2))}
        still = False
        if self.last:
            common = [k for k in cur if k in self.last]
            if len(common) >= 6:
                still = np.mean([np.linalg.norm(cur[k] - self.last[k]) for k in common]) < w / 900
        self.last = cur
        sig = signature(c, w)
        novel = all(np.linalg.norm(sig - s) > 0.07 for s in self.sigs)
        enough = len(ids) >= a.min_corners
        if enough and still and novel and time.time() - self.last_save > 0.7:
            cv2.imwrite(os.path.join(self.out, f"{self.n:03d}.png"), img)
            self.n += 1
            self._remember(c, w)
            self.last_save = time.time()
            res["saved"] = True
            print(f"  saved view {self.n}/{a.count}  ({len(ids)} corners, "
                  f"image coverage {self.covered.mean() / 2.55:.0f}%)", flush=True)
        res["status"] = (
            f"{self.n}/{a.count} saved | {len(ids)} corners"
            + ("" if enough else " (too few)")
            + ("" if still else " - hold still")
            + ("" if novel else " - move to a new pose")
            + f" | coverage {self.covered.mean() / 2.55:.0f}%"
        )
        return res

    def _guides(self, c, ids):
        """Measuring guides drawn on the live view: where to measure forward and left from the origin dot.

        Board frame (OpenCV): origin at the pattern's top-left corner (far from the arm), +x to the
        arm's right, +y toward the arm, on the board surface. The origin dot is at (0, 5 squares).
        """
        p = os.path.join(RESDIR, f"{self.a.name}_intrinsics.json")
        if not os.path.exists(p):
            return {}
        d = json.load(open(p))
        K, D = np.array(d["camera_matrix"]), np.array(d["distortion"])
        obj, imgp = self.board.matchImagePoints(c, ids)
        ok, rv, tv = cv2.solvePnP(obj, imgp, K, D)
        if not ok:
            return {}
        sq = self.a.square_mm / 1000.0
        oy = SQ_Y * sq  # origin dot

        def proj(pts):
            pts = np.asarray(pts, float)
            cam = (cv2.Rodrigues(rv)[0] @ pts.T + tv).T
            uv = cv2.projectPoints(pts, rv, tv, K, D)[0].reshape(-1, 2)
            # Only inside the image: the lens model is not valid beyond its edges.
            keep = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
            return uv, keep

        lines, texts = [], []
        # Forward ruler: from the origin dot straight toward the arm, ticks every 10 mm.
        L = 0.60
        uv, keep = proj([[0, oy + t, 0] for t in np.linspace(0, L, 121)])
        lines.append({"pts": uv[keep].tolist(), "color": "#00e5ff", "w": 3})
        last_mm = 0
        for mm in range(10, int(L * 1000) + 1, 10):
            big = mm % 50 == 0
            h = 0.010 if big else 0.004
            seg, k = proj([[-h, oy + mm / 1000, 0], [h, oy + mm / 1000, 0]])
            if k.all():
                lines.append({"pts": seg.tolist(), "color": "#00e5ff", "w": 3 if big else 1.5})
                if big:
                    texts.append({"x": seg[1][0] + 6, "y": seg[1][1] + 5, "t": f"{mm}", "color": "#00e5ff"})
                    if seg[0][1] < 440:  # leave room below for the left ruler's labels
                        last_mm = mm
        texts.append({"x": uv[0][0] + 10, "y": uv[0][1] - 10, "t": "origin dot (0)", "color": "#ffe600"})
        lines.append({"pts": proj([[0, oy, 0]])[0].tolist(), "color": "#ffe600", "w": 0, "dot": 7})
        if last_mm:
            # Left ruler across the bottom of the view, square to the forward line (+ = arm's left).
            y = oy + last_mm / 1000
            uv2, k2 = proj([[x, y, 0] for x in np.linspace(-0.40, 0.40, 161)])
            lines.append({"pts": uv2[k2].tolist(), "color": "#ff9a00", "w": 3})
            for mm in range(-400, 401, 50):
                if mm == 0:
                    continue
                seg, k = proj([[-mm / 1000, y - 0.008, 0], [-mm / 1000, y + 0.008, 0]])
                if k.all():
                    lines.append({"pts": seg.tolist(), "color": "#ff9a00", "w": 2})
                    texts.append({"x": seg[0][0] - 14, "y": seg[1][1] + 14, "t": f"{mm:+d}", "color": "#ff9a00"})
            texts.append({"x": 10, "y": 40, "t": f"orange: LEFT ruler at {last_mm} mm forward (+ = left, - = right)", "color": "#ff9a00"})
        texts.append({"x": 10, "y": 20, "t": "cyan: FORWARD ruler, mm from the origin dot toward the arm", "color": "#00e5ff"})
        return {"lines": lines, "texts": texts}

    def _extrinsics(self, img, c, ids, save):
        res = {"saved": False}
        if c is None:
            res["status"] = "Extrinsics: board not found (whole board visible, flat, nothing covering it)"
            return res
        res["corners"] = c.reshape(-1, 2).tolist()
        res.update(self._guides(c, ids))
        total = (self.board.getChessboardSize()[0] - 1) * (self.board.getChessboardSize()[1] - 1)
        res["status"] = f"Extrinsics: {len(ids)}/{total} corners seen"
        if save:
            if len(ids) < total:
                res["status"] += " - not saving, some corners are hidden"
                return res
            os.makedirs(self.ext_out, exist_ok=True)
            k = len(glob.glob(os.path.join(self.ext_out, "*.png")))
            p = os.path.join(self.ext_out, f"{k:03d}.png")
            cv2.imwrite(p, img)
            res["saved"] = True
            res["status"] += f" - saved {p}"
            print(f"  saved extrinsics frame {p}", flush=True)
        return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--square-mm", type=float, default=SQUARE_MM)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    cap = Capture(a)
    page = PAGE.replace("__NAME__", a.name).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page)
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if not self.path.startswith("/frame"):
                return self._send(404, "text/plain", b"not found")
            q = dict(p.split("=", 1) for p in self.path.partition("?")[2].split("&") if "=" in p)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            r = cap.frame(body, q.get("mode", "intrinsics"), q.get("save") == "1")
            self._send(200, "application/json", json.dumps(r).encode())

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"Capturing '{a.name}' -> {cap.out} ({cap.n} views already there)\n"
          f"Open http://localhost:{a.port} in Chrome. Ctrl+C to stop.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
