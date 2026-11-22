"""Frozen paired scenarios for local ACT checkpoint selection."""

import hashlib
import json
from pathlib import Path

import env as tabletop
import gymnasium as gym
import mujoco
import numpy as np
from recording.archive import capture_state, load_episode_archive, restore_state
from recording.schema import JOINTS
from recording.self_collision import SelfCollisionChecker

from learning.tabletop import EvaluationEnv


def _joint_ids(model):
    joints = [model.joint(n).id for n in JOINTS]
    return model.jnt_qposadr[joints], model.jnt_dofadr[joints]


def prepare_suite(dataset, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Suite already frozen; reuse it rather than regenerating")
    manifest, archive, model = load_episode_archive(Path(dataset), 45)
    data = mujoco.MjData(model)
    restore_state(model, data, archive["states"][0])
    qadr, dofs = _joint_ids(model)
    actuators = np.array([model.actuator(n).id for n in JOINTS])
    lo, hi = model.actuator_ctrlrange[actuators].T
    cube, box = model.body("red_cube").id, model.body("box").id
    cube_qa = int(model.jnt_qposadr[model.body_jntadr[cube]])
    cube0, bin0 = data.xpos[cube].copy(), data.xpos[box].copy()
    separation = bin0[:2] - cube0[:2]
    away = separation / np.linalg.norm(separation)
    sideways = np.array([-away[1], away[0]])
    baseline_body_pos = model.body_pos.copy()
    checker = SelfCollisionChecker(model)
    anchor_offset = np.array([0.0, 2.0, -2.0, 0.0, 0.0, 0.0])
    candidates = []

    def add(
        name,
        family,
        robot=None,
        cube_shift=(0, 0),
        bin_away=0.0,
        bin_side=0.0,
        legacy=False,
    ):
        candidates.append(
            dict(
                id=name,
                family=family,
                robot_delta_deg=list(np.zeros(6) if robot is None else robot),
                cube_delta_m=list(cube_shift),
                bin_away_m=bin_away,
                bin_side_m=bin_side,
                legacy=legacy,
            )
        )

    add("legacy_original45", "historical_control", legacy=True)
    add("anchor", "control")
    robot_offsets = [
        np.array(x, float)
        for x in (
            [2, 0, 0, 0, 0, 0],
            [-2, 0, 0, 0, 0, 0],
            [2, 1, 1, 2, 2, 0],
            [-2, -1, -1, -2, -2, 0],
        )
    ]
    for i, delta in enumerate(robot_offsets):
        add(f"robot_{i + 1}", "robot", robot=delta)
    for cm in (1, 2, 4, 6):
        for axis in (0, 1):
            for sign in (-1, 1):
                offset = np.zeros(2)
                offset[axis] = sign * cm / 100
                add(
                    f"cube_{'xy'[axis]}_{'plus' if sign > 0 else 'minus'}{cm}cm",
                    "cube",
                    cube_shift=offset,
                )
    # Conservative bin limits: <=4 cm away and <=4 cm sideways, then IK checked.
    for cm in (2, 4):
        add(f"bin_away{cm}cm", "bin", bin_away=cm / 100)
    for cm in (2, 4):
        for sign in (-1, 1):
            add(
                f"bin_away2_side{'plus' if sign > 0 else 'minus'}{cm}cm",
                "bin",
                bin_away=0.02,
                bin_side=sign * cm / 100,
            )
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        add(
            f"combined_{i + 1}",
            "combined",
            robot=robot_offsets[2 if sx > 0 else 3],
            cube_shift=(sx * 0.02, sy * 0.02),
            bin_away=0.02,
            bin_side=sx * 0.02,
        )

    states, bins, cases, rejected = [], [], [], []
    for case in candidates:
        model.body_pos[:] = baseline_body_pos
        restore_state(model, data, archive["states"][0])
        if not case["legacy"]:
            delta = anchor_offset + np.asarray(case["robot_delta_deg"])
            data.qpos[qadr[:5]] += np.deg2rad(delta[:5])
            data.qvel[:] = 0
            data.qacc_warmstart[:] = 0
            data.ctrl[actuators] = data.qpos[qadr]
        bin_position = bin0.copy()
        bin_position[:2] += case["bin_away_m"] * away + case["bin_side_m"] * sideways
        model.body_pos[box] = bin_position
        data.qpos[cube_qa : cube_qa + 2] += case["cube_delta_m"]
        mujoco.mj_forward(model, data)
        collision = checker.check(data)
        reason = None
        if np.any(data.qpos[qadr] < lo - 1e-8) or np.any(data.qpos[qadr] > hi + 1e-8):
            reason = "joint_limit"
        elif collision and not case["legacy"]:
            reason = "initial_self_collision"
        if np.any(np.abs(data.xpos[cube][:2] - [0.2, 0]) + 0.015 >= [0.4, 0.4]):
            reason = "cube_off_table"
        if np.any(np.abs(bin_position[:2] - [0.2, 0]) + 0.049 >= [0.4, 0.4]):
            reason = "bin_off_table"
        # Full initial geometric contacts with the cube/bin, except table support, are invalid.
        for c in data.contact:
            bodies = {int(model.geom_bodyid[g]) for g in c.geom}
            if cube in bodies and box in bodies and c.dist < 0:
                reason = "cube_bin_overlap"
        if reason:
            rejected.append({**case, "reason": reason, "contact": collision})
            continue
        case.update(
            index=len(states),
            bin_position=bin_position.tolist(),
            cube_position=data.xpos[cube].tolist(),
            initial_contact=collision,
            primary=not case["legacy"],
        )
        states.append(capture_state(model, data))
        bins.append(bin_position)
        cases.append(case)
    model.body_pos[:] = baseline_body_pos
    mujoco.mj_saveModel(model, str(destination / "model.mjb"))
    np.savez_compressed(destination / "states.npz", states=states, bin_positions=bins)
    result = {
        "schema_version": 1,
        "name": "so101-robustness-v1",
        "source_episode_uuid": manifest["episode_uuid"],
        "source_original_episode": 45,
        "source_scene": manifest["scene_id"],
        "fps": 30,
        "deadline": 900,
        "height": 480,
        "width": 640,
        "substeps": manifest["metadata"]["substeps"],
        "instruction": manifest["metadata"]["task"],
        "anchor_offset_deg": anchor_offset.tolist(),
        "bin_axis_away": away.tolist(),
        "bin_axis_sideways": sideways.tolist(),
        "cases": cases,
        "rejected": rejected,
        "notes": [
            "Primary cases use collision-free nearby anchor; historical control remains labeled separately.",
            "All scenario states are frozen before any checkpoint is run.",
            "This is a checkpoint-selection development suite, not an independent generalization estimate.",
        ],
    }
    result["states_sha256"] = hashlib.sha256(
        (destination / "states.npz").read_bytes()
    ).hexdigest()
    result["model_sha256"] = hashlib.sha256(
        (destination / "model.mjb").read_bytes()
    ).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(result, indent=2))
    return preflight_suite(dataset, destination)


class ScenarioEnv(EvaluationEnv):
    """The native task/success checker, with resets from a frozen benchmark bundle."""

    def __init__(self, bundle, case_index):
        self.bundle = Path(bundle)
        self.suite = json.loads((self.bundle / "manifest.json").read_text())
        self.case = self.suite["cases"][case_index]
        with np.load(self.bundle / "states.npz") as z:
            self.initial_state = z["states"][case_index].copy()
            self.bin_position = z["bin_positions"][case_index].copy()
        super().__init__(
            fps=self.suite["fps"],
            episode_length=self.suite["deadline"],
            observation_height=self.suite["height"],
            observation_width=self.suite["width"],
            cameras={"overhead": "overhead", "wrist": "wrist_cam"},
            terminate_on_success=False,
        )
        self.reset_metrics()

    def reset_metrics(self):
        self.targets = []
        self.positions = []
        self.velocities = []
        self.contacts = []
        self.contact_ticks = set()
        self.first_done = False
        self.outcome = {}

    def _ensure_world(self):
        if self._world is None:
            w = tabletop._World.__new__(tabletop._World)
            w.model = mujoco.MjModel.from_binary_path(str(self.bundle / "model.mjb"))
            w.data = mujoco.MjData(w.model)
            w.qadr, w.dofs = _joint_ids(w.model)
            w.actuators = np.array([w.model.actuator(n).id for n in JOINTS])
            w.lo, w.hi = w.model.actuator_ctrlrange[w.actuators].T.copy()
            w.cube_bodies = [w.model.body("red_cube").id]
            w.box_body = w.model.body("box").id
            w.substeps = self.suite["substeps"]
            w.instruction = self.suite["instruction"]
            w.renderer = mujoco.Renderer(w.model, self.height, self.width)
            self._world = w
            self.checker = SelfCollisionChecker(w.model)
        return self._world

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        w = self._ensure_world()
        w.model.body_pos[w.box_body] = self.bin_position
        restore_state(w.model, w.data, self.initial_state)
        self._steps = 0
        self._settled_steps = 0
        self.reset_metrics()
        self.positions.append(w.joint_state())
        self.velocities.append(self._velocity())
        initial = self.checker.check(w.data)
        if initial:
            self.contacts.append({**initial, "step": -1})
        self.physics_step_callback = self._contact_observer
        return self._obs(w), {"is_success": False, "scenario_id": self.case["id"]}

    def _velocity(self):
        w = self._ensure_world()
        v = np.rad2deg(w.data.qvel[w.dofs])
        v[5] = w.data.qvel[w.dofs[5]] / (w.hi[5] - w.lo[5]) * 100
        return v.copy()

    def _contact_observer(self, model, data):
        contact = self.checker.check(data)
        if contact:
            self.contact_ticks.add(len(self.targets))
            if len(self.contacts) < 1000:
                self.contacts.append({**contact, "step": self._steps})

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        w = self._ensure_world()
        self._contact_observer(w.model, w.data)
        target = np.rad2deg(w.data.ctrl[w.actuators])
        target[5] = (w.data.ctrl[w.actuators[5]] - w.lo[5]) / (w.hi[5] - w.lo[5]) * 100
        self.targets.append(target)
        self.positions.append(w.joint_state())
        self.velocities.append(self._velocity())
        if terminated or truncated:
            cube = w.data.xpos[w.cube_bodies[0]]
            self.outcome = {
                "task_success": bool(info["is_success"]),
                "collision_free_success": bool(
                    info["is_success"] and not self.contacts
                ),
                "steps": self._steps,
                "sim_seconds": self._steps / self.fps,
                "contact_control_ticks": len(self.contact_ticks),
                "first_contact": self.contacts[0] if self.contacts else None,
                "contact_pairs": sorted({tuple(x["bodies"]) for x in self.contacts}),
                "cube_final_position": cube.tolist(),
                "failure": ""
                if info["is_success"]
                else ("cube_off_table" if cube[2] < -0.03 else "timeout"),
            }
        return obs, reward, terminated, truncated, info


def preflight_suite(dataset, destination):
    """Freeze verified endpoints; reject invalid starts before checkpoint selection."""
    from learning.reachability import find_reachability_witness, select_reference_pose

    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "states.npz") as saved:
        states = saved["states"].copy()
        bins = saved["bin_positions"].copy()
    source_manifest, archive, model = load_episode_archive(Path(dataset), 45)
    np.savez_compressed(root / "source_archive.npz", **archive)
    (root / "source_archive_manifest.json").write_text(
        json.dumps(source_manifest, indent=2)
    )
    data = mujoco.MjData(model)
    box = model.body("box").id
    base_bin = model.body_pos[box].copy()
    reference = select_reference_pose(model, archive["states"], base_bin)
    manifest["reachability_reference"] = reference
    # Version 1's demonstrated cube approach reference, recorded explicitly.
    restore_state(model, data, archive["states"][125])
    cube_seed = data.qpos.copy()
    cube_site = data.site_xpos[model.site("gripperframe").id].copy()
    manifest["cube_reachability_reference"] = {
        "source_state_index": 125,
        "site_position": cube_site.tolist(),
    }
    accepted, new_states, new_bins = [], [], []
    bin_checks = {}
    for index, case in enumerate(manifest["cases"]):
        model.body_pos[box] = bins[index]
        bin_shift = bins[index] - base_bin
        key = tuple(np.round(bin_shift, 8))
        if np.any(np.abs(bin_shift) > 1e-9):
            if key not in bin_checks:
                bin_checks[key] = find_reachability_witness(
                    model,
                    reference["qpos"],
                    np.array(reference["position"]) + bin_shift,
                )
            witness = bin_checks[key]
            if not witness["reachable"]:
                manifest["rejected"].append(
                    {
                        **case,
                        "reason": "bin_endpoint_unverified",
                        "reachability": witness,
                    }
                )
                continue
            case["bin_reachability"] = witness
        else:
            case["bin_reachability"] = {
                "reachable": True,
                "reason": "demonstrated_original_destination",
                "scope": "Known demonstrated bin destination",
            }
        restore_state(model, data, states[index])
        checker = SelfCollisionChecker(model)
        bad_contact = any(
            contact.dist <= 0
            and any(
                int(model.geom_bodyid[g]) in checker.robot_bodies for g in contact.geom
            )
            and any(
                int(model.geom_bodyid[g]) not in checker.robot_bodies
                for g in contact.geom
            )
            for contact in data.contact
        )
        if bad_contact and case["primary"]:
            manifest["rejected"].append(
                {**case, "reason": "initial_robot_object_contact"}
            )
            continue
        cube_shift = np.array([*case["cube_delta_m"], 0.0])
        if np.any(cube_shift):
            witness = find_reachability_witness(
                model, cube_seed, cube_site + cube_shift
            )
            if not witness["reachable"]:
                manifest["rejected"].append(
                    {
                        **case,
                        "reason": "cube_endpoint_unverified",
                        "reachability": witness,
                    }
                )
                continue
            case["cube_reachability"] = witness
        case["index"] = len(new_states)
        accepted.append(case)
        new_states.append(states[index])
        new_bins.append(bins[index])
    manifest["cases"] = accepted
    manifest["bin_reachability_checked"] = True
    manifest["cube_reachability_checked"] = True
    manifest["bin_reachability_note"] = (
        "Modified bins have joint-limited, self-contact-free release endpoints within 5mm and 15 degrees tool-axis tolerance. This does not prove collision-free approach paths."
    )
    np.savez_compressed(root / "states.npz", states=new_states, bin_positions=new_bins)
    manifest["states_sha256"] = hashlib.sha256(
        (root / "states.npz").read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_suite(args.dataset, args.output)
    print("Frozen", len(result["cases"]), "cases;", len(result["rejected"]), "rejected")
