"""Run Tony's frozen rollout suites (dataset-so101-physical-v1-200ep, evaluation/) with our PPO/BC policies.

Tony's runner (lerobot-tabletop/evaluate_suite.py) loads ACT checkpoints only. This replays the same
cases in So101PickPlaceEnv so our observation/action pipeline (relative actions, resync mapping,
sensing delays) is unchanged, and scores them with Tony's own definitions:

* start: the case's archived physics state (arm pose, cube) and bin position, restored exactly;
* appearance: recording/visual_randomization.apply_profile + augment_image from the release's
  training-tools-overlay.zip (case "secondary_light" as in learning/visual_test_suite.py). Images are
  rendered at the policy's size (as in training) and the image effects applied there with blur
  radius and noise sigma scaled by width/640. Checked against rendering at 640x480, augmenting and
  area-downsampling: mean difference 0.2-1.9 grey levels, the same as native vs area-downsampled
  rendering alone (the effects themselves change 3-15 levels), at 1/13 of the cost;
* task success: Tony's EvaluationEnv.placement_valid (cube wholly inside the bin, settled, released)
  for 0.5 s, within the suite deadline (900 steps);
* prohibited contact: recording.self_collision.RecordingCollisionChecker (self-contact, upright,
  curtains) at every physics substep; collision-free success = success with no such contact.

    python eval_suite.py --suite <release>/evaluation/validation --runs physA_nominal_s1 ... \
        --steps 500000 600000 --delays 0:0:0 2:0:1 3:0:2 --name physA_validation
    python eval_suite.py --suite <release>/evaluation/final-test --checkpoints <selected.json> ...

Results: outputs/suite_eval/<name>/results.jsonl (one row per run/step/delay/case; resumable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs" / "suite_eval"
# Extracted training-tools-overlay.zip from the release (it holds visual_randomization.py).
OVERLAY = Path(os.environ.get("SO101_TOOLS_OVERLAY", Path.home() / "datasets" / "so101-physical-v1-200ep" / "training-tools-overlay"))


def parse_delay(text: str) -> dict:
    a, j, i = (int(v) for v in text.split(":"))
    return {"latency_steps": a, "obs_latency_steps": j, "image_latency_steps": i}


def placement_valid(world) -> bool:
    """Tony's learning/tabletop.EvaluationEnv.placement_valid on our world."""
    model, data = world.model, world.data
    cube = world.cube_body
    geom = int(model.body_geomadr[cube])
    rotation = data.xmat[world.box_body].reshape(3, 3)
    local = (data.geom_xpos[geom] - data.xpos[world.box_body]) @ rotation
    extent = np.abs(rotation.T @ data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
    inside = bool(np.all(np.abs(local[:2]) + extent[:2] < 0.045) and local[2] - extent[2] >= 0.002
                  and local[2] + extent[2] <= 0.037)
    settled = bool(np.linalg.norm(data.cvel[cube, 3:]) < 0.02 and np.linalg.norm(data.cvel[cube, :3]) < 0.5)
    for contact in data.contact:
        bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
        if cube in bodies and any(b not in (cube, world.box_body, 0) for b in bodies):
            return False
    return inside and settled


VISUAL_FIELDS = ("mat_specular", "mat_shininess", "mat_rgba", "geom_rgba", "light_dir", "light_pos",
                 "light_diffuse", "light_specular")


class SuiteRunner:
    """One env + checker per worker process; cases are run one at a time."""

    def __init__(self, suite: Path, policy_path: Path, delay: dict | None, device: str = "cpu"):
        import importlib.util

        sys.path.insert(0, str(HERE.parent / "lerobot-tabletop"))
        import mujoco
        from recording.self_collision import RecordingCollisionChecker

        # visual_randomization.py ships in the release overlay only; load it by path (the repo's own
        # recording package would shadow the overlay's folder).
        spec = importlib.util.spec_from_file_location(
            "visual_randomization", OVERLAY / "lerobot-tabletop" / "recording" / "visual_randomization.py")
        vr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vr)
        apply_profile, augment_image = vr.apply_profile, vr.augment_image

        import deploy
        from envs import So101PickPlaceEnv

        self.mujoco, self.apply_profile, self.augment_image = mujoco, apply_profile, augment_image
        self.suite = suite
        self.manifest = json.loads((suite / "manifest.json").read_text())
        with np.load(suite / "states.npz") as z:
            self.states, self.bins = z["states"].copy(), z["bin_positions"].copy()
        self.policy = deploy.load_policy(policy_path, device=device)
        extra = {"fixed_factors": {"actuator": delay}} if delay else {}
        self.env = So101PickPlaceEnv(**self.policy.env_kwargs(**extra))
        self.env.episode_steps = int(self.manifest["deadline"])
        w = self.env.world
        self.ref = mujoco.MjModel.from_binary_path(str(suite / "model.mjb"))
        self.ref_data = mujoco.MjData(self.ref)
        self._check_scene()
        self.checker = RecordingCollisionChecker(w.model)
        self.visual0 = {f: getattr(w.model, f).copy() for f in VISUAL_FIELDS}
        self.head0 = (w.model.vis.headlight.diffuse.copy(), w.model.vis.headlight.ambient.copy())
        self.size = (int(self.manifest["height"]), int(self.manifest["width"]))
        self._render = w.render
        self.profile, self._scaled = None, None
        w.render = self._sensor_render
        self.contacts: list = []
        w.step = self._step_checked

    def _check_scene(self) -> None:
        mine, ref = self.env.world.model, self.ref
        names = lambda m: [m.joint(i).name for i in range(m.njnt)]  # noqa: E731
        if (ref.nq, ref.nv, ref.nu, names(ref)) != (mine.nq, mine.nv, mine.nu, names(mine)):
            raise RuntimeError("suite model layout differs from the env's model")
        if not np.isclose(ref.opt.timestep, mine.opt.timestep):
            raise RuntimeError("suite physics timestep differs from the env's")
        # Our scene adds one body/geom, the hidden reach-target marker; everything else must match.
        marker = mine.body("reach_target").id
        geoms = [g for g in range(mine.ngeom) if mine.geom_bodyid[g] != marker]
        bodies = [i for i in range(mine.nbody) if i != marker]
        box = ref.body("box").id
        checks = {"geom_size": geoms, "geom_pos": geoms, "geom_friction": geoms, "body_mass": bodies,
                  "body_pos": bodies, "cam_pos": None, "cam_quat": None, "cam_fovy": None,
                  "actuator_gainprm": None, "actuator_biasprm": None, "dof_damping": None,
                  "dof_armature": None, "light_pos": None, "jnt_range": None}
        for field, keep in checks.items():
            a, b = getattr(mine, field), getattr(ref, field)
            if keep is not None:
                a = a[keep]
            if field == "body_pos":  # the bin is moved per case
                a, b = np.delete(a, box, 0), np.delete(b, box, 0)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
                raise RuntimeError(f"suite scene differs from the env's in {field}")

    def _sensor_render(self, camera: str, height: int, width: int) -> np.ndarray:
        image = self._render(camera, height, width)
        if self.profile is not None:
            index = 0 if camera == self.env.camera else 1  # Tony's camera order: overhead, wrist
            image = self.augment_image(image, self._scaled_profile(width), index, self.env._steps)
        return image

    def _scaled_profile(self, width: int) -> dict:
        k = width / self.size[1]
        if self._scaled is None or self._scaled[0] != k:
            p = json.loads(json.dumps(self.profile))
            p["camera_effects"]["blur_radius_px"] *= k
            p["camera_effects"]["noise_sigma_255"] *= k
            self._scaled = (k, p)
        return self._scaled[1]

    def _step_checked(self, ctrl: np.ndarray) -> None:
        w = self.env.world
        w.data.ctrl[w.act] = np.clip(ctrl, w.lo, w.hi)
        for _ in range(w.substeps):  # Tony checks prohibited contact at every physics substep
            self.mujoco.mj_step(w.model, w.data)
            contact = self.checker.check(w.data)
            if contact and len(self.contacts) < 50:
                self.contacts.append({**contact, "step": self.env._steps})

    def _appearance(self, case: dict) -> None:
        m = self.env.world.model
        for f, v in self.visual0.items():
            getattr(m, f)[:] = v
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = self.head0
        profile = case.get("appearance_profile")
        self.profile = profile if profile and profile.get("variant", 0) != 0 else None
        self._scaled = None
        if profile is not None:
            self.apply_profile(m, profile)
        extra = case.get("secondary_light")
        if extra:
            m.light_pos[1], m.light_dir[1] = extra["position"], extra["direction"]
            m.light_diffuse[1], m.light_specular[1] = extra["diffuse"], extra["specular"]

    def run_case(self, index: int) -> dict:
        mj, env, w = self.mujoco, self.env, self.env.world
        case = self.manifest["cases"][index]
        mj.mj_setState(self.ref, self.ref_data, self.states[index], mj.mjtState.mjSTATE_INTEGRATION)
        cube_xy = self.ref_data.qpos[w.cube_qadr: w.cube_qadr + 2].copy()
        env.reset(seed=1000, options={"layout": {"cube_xy": cube_xy, "bin_xy": self.bins[index][:2]}})
        # Exact archived start (as Tony's ScenarioEnv.reset): bin, then the full integration state.
        w.model.body_pos[w.box_body] = self.bins[index]
        d, r = w.data, self.ref_data
        d.time, d.qpos[:], d.qvel[:], d.act[:], d.ctrl[:] = r.time, r.qpos, r.qvel, r.act, r.ctrl
        d.qacc_warmstart[:], d.qfrc_applied[:] = r.qacc_warmstart, r.qfrc_applied
        mj.mj_forward(w.model, d)
        self._appearance(case)
        q = w.ctrl.copy()
        env._target_ctrl, env._held = q.copy(), q.copy()
        env._pending.clear()
        if env.mapper is not None:
            env.mapper.reset(q)  # the last command sent is the archived hold target
        env._in_box_steps = 0
        env._phi, _ = env._potential()
        self.contacts = []
        initial = self.checker.check(d)
        if initial:
            self.contacts.append({**initial, "step": -1})
        obs = env._observe(fresh=True)
        settled, success, lifted, our_success, steps = 0, False, False, False, 0
        need = int(np.ceil(self.manifest["fps"] * 0.5))
        for steps in range(1, env.episode_steps + 1):
            action, _ = self.policy.model.predict(obs, deterministic=True)
            obs, _, our_done, _, info = env.step(action)
            lifted |= bool(info.get("cube_lifted"))
            our_success |= bool(info.get("is_success"))
            settled = settled + 1 if placement_valid(w) else 0
            if settled >= need:
                success = True
                break
        cube = w.cube_pos()
        return {
            "case": case["id"], "family": case["family"], "primary": case.get("primary", True),
            "pair": case.get("pair"), "stress_factor": case.get("stress_factor"),
            "task_success": success, "collision_free_success": success and not self.contacts,
            "prohibited_contact": bool(self.contacts),
            "first_contact": self.contacts[0] if self.contacts else None,
            "lifted": lifted, "env_success": our_success, "steps": steps,
            "cube_final": cube.tolist(),
            "failure": "" if success else ("cube_off_table" if cube[2] < -0.03 else "timeout"),
        }


def work(job: dict) -> list[dict]:
    """All remaining cases of one (checkpoint, delay) in one env, closed afterwards. Workers are also
    recycled after each task (maxtasksperchild=1): keeping envs across checkpoints leaked ~8 GB per
    worker and exhausted the commit limit beside training."""
    import torch

    torch.set_num_threads(1)
    runner = SuiteRunner(Path(job["suite"]), Path(job["policy"]),
                         parse_delay(job["delay"]) if job["delay"] != "0:0:0" else None, job["device"])
    rows = []
    try:
        for index in job["indices"]:
            start = time.time()
            row = runner.run_case(index)
            rows.append({**{k: job[k] for k in ("run", "condition", "seed", "step", "delay")}, **row,
                         "seconds": round(time.time() - start, 1)})
    finally:
        runner.env.close()
    return rows


def checkpoint(run: str, step: int | str) -> Path:
    run_dir = HERE / "outputs" / "ppo" / run
    if str(step) == "final":
        return run_dir / "model_final.zip"
    return run_dir / f"model_{int(step)}.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, required=True, help="evaluation/validation or evaluation/final-test")
    parser.add_argument("--runs", nargs="+", help="PPO run names under outputs/ppo")
    parser.add_argument("--steps", nargs="+", default=["final"], help="checkpoint steps (or 'final')")
    parser.add_argument("--checkpoints", type=Path, help="JSON {run: step} (the selected checkpoints)")
    parser.add_argument("--delays", nargs="+", default=["0:0:0"], help="commands:joints:camera steps late")
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cpu", help="policy device per worker; the default keeps "
                        "the GPU free for training (6 GPU workers beside two PPO runs ran out of memory)")
    parser.add_argument("--overlay", type=Path, help="extracted training-tools-overlay.zip "
                        "(default: $SO101_TOOLS_OVERLAY)")
    args = parser.parse_args()
    global OVERLAY
    if args.overlay:
        OVERLAY = args.overlay
    os.environ["SO101_TOOLS_OVERLAY"] = str(OVERLAY)
    if not (OVERLAY / "lerobot-tabletop" / "recording" / "visual_randomization.py").is_file():
        sys.exit("--overlay must point at the extracted training-tools-overlay.zip")
    manifest_path = args.suite / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, key in (("states.npz", "states_sha256"), ("model.mjb", "model_sha256")):
        if hashlib.sha256((args.suite / name).read_bytes()).hexdigest() != manifest[key]:
            sys.exit(f"{name} does not match the suite manifest")
    if args.checkpoints:
        selected = json.loads(args.checkpoints.read_text())
        pairs = [(run, step) for run, step in selected.items()]
    else:
        pairs = [(run, step) for run in args.runs for step in args.steps]
    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = set()
    if results.is_file():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            done.add((r["run"], str(r["step"]), r["delay"], r["case"]))
    config = {"suite": str(args.suite), "suite_name": manifest["name"], "role": manifest["role"],
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "checkpoints": [[r, str(s)] for r, s in pairs], "delays": args.delays,
              "deterministic_actions": True, "deadline": manifest["deadline"],
              "render": "policy resolution; image effects scaled by width/640"}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    jobs = []
    for run, step in pairs:
        path = checkpoint(run, step)
        if not path.is_file():
            sys.exit(f"missing checkpoint {path}")
        seed = re.search(r"_s(\d+)$", run)
        condition = "randomized" if "randomized" in run else "nominal"
        for delay in args.delays:
            todo = [i for i, case in enumerate(manifest["cases"]) if (run, str(step), delay, case["id"]) not in done]
            if todo:
                jobs.append({"suite": str(args.suite), "policy": str(path), "run": run, "step": str(step),
                             "condition": condition, "seed": int(seed.group(1)) if seed else None,
                             "delay": delay, "indices": todo, "device": args.device})
    total = sum(len(j["indices"]) for j in jobs)
    print(f"{total} episodes in {len(jobs)} groups to run ({len(done)} already done) -> {results}", flush=True)
    start, k = time.time(), 0
    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, results.open("a") as f:
        for rows in pool.imap_unordered(work, jobs, chunksize=1):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            k += len(rows)
            print(f"[{time.time() - start:6.0f}s] {k}/{total}  ({rows[0]['run']} {rows[0]['step']} "
                  f"{rows[0]['delay']}: {sum(r['task_success'] for r in rows)}/{len(rows)})", flush=True)
    summarize(results, out / "summary.md")


def summarize(results: Path, path: Path) -> None:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["run"], r["step"], r["delay"], r["family"]), []).append(r)
    lines = ["| run | step | delay | family | cases | task success | collision-free success | lifted |",
             "|---|---|---|---|---|---|---|---|"]
    for (run, step, delay, family), rs in sorted(groups.items()):
        n = len(rs)
        s = sum(r["task_success"] for r in rs)
        c = sum(r["collision_free_success"] for r in rs)
        lf = sum(r["lifted"] for r in rs)
        lines.append(f"| {run} | {step} | {delay} | {family} | {n} | {s}/{n} | {c}/{n} | {lf}/{n} |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
