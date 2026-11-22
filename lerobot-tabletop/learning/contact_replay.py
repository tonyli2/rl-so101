"""Replay identical TE targets with and without one shoulder–gripper contact pair.

This is a diagnostic counterfactual, not a proposed simulation configuration.
No MjModel field or checkpoint is changed; the process-local contact callback is
restored even if integration fails.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from recording.archive import load_episode_archive, restore_state
from recording.schema import JOINTS


def _body_id(model, basename):
    """Resolve actual archived body names, allowing an attached-model prefix."""
    matches = [
        i
        for i in range(model.nbody)
        if model.body(i).name.rsplit("/", 1)[-1] == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {basename!r} body; found {matches}")
    return matches[0]


def _measure(model, data, qadr, dofs, gripper_range):
    position = np.rad2deg(data.qpos[qadr])
    velocity = np.rad2deg(data.qvel[dofs])
    low, high = gripper_range
    position[5] = (data.qpos[qadr[5]] - low) / (high - low) * 100
    velocity[5] = data.qvel[dofs[5]] / (high - low) * 100
    return position.copy(), velocity.copy()


def _statistics(positions, velocities, targets, limit=None):
    n = len(targets) if limit is None else min(limit, len(targets))
    delta = np.diff(positions[: n + 1], axis=0)
    # Do not let the initial reset-to-command transient dominate the snap metric.
    delta = delta[min(10, len(delta)) :]
    return {
        "steps": n,
        "max_abs_position_increment_after_tick10": np.max(
            np.abs(delta), axis=0
        ).tolist()
        if len(delta)
        else None,
        "max_abs_tracking_error": np.max(
            np.abs(positions[1 : n + 1] - targets[:n]), axis=0
        ).tolist(),
        "max_abs_velocity": np.max(np.abs(velocities[1 : n + 1]), axis=0).tolist(),
    }


def replay(trace_path, dataset, episode, output):
    """Write replay.npz and result.json; refuse to replace an existing result."""
    with np.load(trace_path, allow_pickle=False) as trace:
        targets = np.asarray(trace["applied"], dtype=float)
        measured_positions = np.asarray(trace["positions"], dtype=float)
        measured_velocities = np.asarray(trace["velocities"], dtype=float)
    n = len(targets)
    if (
        targets.shape != (n, 6)
        or measured_positions.shape != (n + 1, 6)
        or measured_velocities.shape != (n + 1, 6)
    ):
        raise ValueError("Expected applied (T,6), positions/velocities (T+1,6)")
    if not all(
        np.isfinite(x).all() for x in (targets, measured_positions, measured_velocities)
    ):
        raise ValueError("Trace contains nonfinite values")
    manifest, archive, model = load_episode_archive(dataset, episode)
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    dofs = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    actuators = np.array([model.actuator(name).id for name in JOINTS])
    low, high = model.actuator_ctrlrange[actuators].T
    pair = {_body_id(model, "shoulder"), _body_id(model, "gripper")}
    substeps = manifest["metadata"]["substeps"]
    original_filter = mujoco.get_mjcb_contactfilter()

    def filtered_contact(mod, data, geom1, geom2):
        if {int(mod.geom_bodyid[geom1]), int(mod.geom_bodyid[geom2])} == pair:
            return 1
        return (
            original_filter(mod, data, geom1, geom2)
            if original_filter is not None
            else 0
        )

    payload = {
        "applied": targets,
        "measured_positions": measured_positions,
        "measured_velocities": measured_velocities,
    }
    metrics = {
        "trace": str(trace_path.resolve()),
        "dataset": str(dataset.resolve()),
        "episode": episode,
        "joint_names": list(JOINTS),
        "units": ["degrees"] * 5 + ["percentage_points"],
        "suppressed_body_pair": [model.body(i).name for i in sorted(pair)],
        "substeps": substeps,
        "physics_timestep": model.opt.timestep,
        "note": "Only this contact pair is suppressed. Targets and initial integration state are identical. This is a causal diagnostic, not a production fix.",
    }
    for label, callback in (
        ("baseline", original_filter),
        ("pair_filtered", filtered_contact),
    ):
        try:
            mujoco.set_mjcb_contactfilter(callback)
            data = mujoco.MjData(model)
            restore_state(model, data, archive["states"][0])
            q, v = _measure(model, data, qadr, dofs, (low[5], high[5]))
            positions, velocities = [q], [v]
            pair_contacts, pair_forces, torques = [], [], []
            for target in targets:
                control = np.deg2rad(target)
                control[5] = low[5] + target[5] / 100 * (high[5] - low[5])
                data.ctrl[actuators] = np.clip(control, low, high)
                mujoco.mj_step(model, data, nstep=substeps)
                mujoco.mj_forward(model, data)
                q, v = _measure(model, data, qadr, dofs, (low[5], high[5]))
                positions.append(q)
                velocities.append(v)
                contact_forces = []
                for index, contact in enumerate(data.contact):
                    if {int(model.geom_bodyid[g]) for g in contact.geom} == pair:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        contact_forces.append(float(force[0]))
                pair_contacts.append(len(contact_forces))
                pair_forces.append(sum(contact_forces))
                torques.append(data.actuator_force[actuators].copy())
        finally:
            mujoco.set_mjcb_contactfilter(original_filter)
        positions, velocities = np.asarray(positions), np.asarray(velocities)
        payload.update(
            {
                f"{label}_positions": positions,
                f"{label}_velocities": velocities,
                f"{label}_pair_contacts": np.asarray(pair_contacts),
                f"{label}_pair_normal_force": np.asarray(pair_forces),
                f"{label}_actuator_force": np.asarray(torques),
            }
        )
        metrics[label] = {
            "full_rollout": _statistics(positions, velocities, targets),
            "first150": _statistics(positions, velocities, targets, 150),
            "pair_contact_steps": np.flatnonzero(pair_contacts).tolist(),
            "max_pair_normal_force": max(pair_forces),
        }
        if label == "baseline":
            metrics["baseline_alignment"] = {
                "max_abs_position_error": np.max(
                    np.abs(positions - measured_positions), axis=0
                ).tolist(),
                "max_abs_velocity_error": np.max(
                    np.abs(velocities - measured_velocities), axis=0
                ).tolist(),
                "first150_max_abs_position_error": np.max(
                    np.abs(positions[:151] - measured_positions[:151]), axis=0
                ).tolist(),
                "note": "Applied targets were saved as float32. Tiny reconstruction differences may grow over long contact-rich rollouts; inspect early alignment before interpreting the counterfactual.",
            }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "replay.npz", **payload)
    (output / "result.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.trace, args.dataset, args.episode, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
