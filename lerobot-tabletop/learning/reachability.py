"""Conservative endpoint IK witnesses for fixed-scene robustness cases.

A witness is a joint-limited, self-contact-free endpoint near a reference tool
orientation. It does not establish a collision-free path, a feasible grasp, or
that an ACT policy can reach that endpoint.
"""

import mujoco
import numpy as np
from recording.archive import restore_state
from recording.schema import JOINTS
from recording.self_collision import SelfCollisionChecker


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _obstacle_contact(model, data, checker):
    """Disallow robot contact with table/world or bin; cube contact may be grasp."""
    box = model.body("box").id
    for contact in data.contact:
        if contact.dist > 0:
            continue
        bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
        if any(b in checker.robot_bodies for b in bodies) and any(
            b in (0, box) for b in bodies
        ):
            return True
    return False


def select_reference_pose(model, states, bin_center, site_name="gripperframe"):
    """Pick a clear, elevated demonstrated pose near the bin as an IK seed.

    The score favors site positions close to the bin's XY center and approximately
    10 cm above the table. Stored source state index makes the choice reviewable.
    """
    data = mujoco.MjData(model)
    checker = SelfCollisionChecker(model)
    site = model.site(site_name).id
    center = np.asarray(bin_center, dtype=float)
    candidates = []
    for index, state in enumerate(states):
        restore_state(model, data, state)
        position = data.site_xpos[site].copy()
        xy_error = float(np.linalg.norm(position[:2] - center[:2]))
        if not (0.05 <= position[2] <= 0.18 and xy_error <= 0.08):
            continue
        if checker.check(data) is not None or _obstacle_contact(model, data, checker):
            continue
        score = xy_error + 0.3 * abs(float(position[2]) - 0.10)
        candidates.append(
            (
                score,
                index,
                data.qpos.copy(),
                position,
                data.site_xmat[site].reshape(3, 3)[:, 2].copy(),
            )
        )
    if not candidates:
        raise ValueError("No clear demonstrated elevated pose near the bin")
    _, index, qpos, position, axis = min(candidates, key=lambda item: item[0])
    return {
        "qpos": qpos.tolist(),
        "position": position.tolist(),
        "axis": axis.tolist(),
        "state_index": index,
    }


def find_reachability_witness(
    model,
    reference_qpos,
    target_position,
    site_name="gripperframe",
    tolerance=0.005,
    max_iterations=200,
):
    """Find a 5-arm-DOF position/tool-axis endpoint witness using damped IK.

    Gripper opening and all non-arm qpos stay fixed. The two independent
    tool-axis constraints retain the demonstrated approach without imposing
    an impossible full six-dimensional pose on a five-DOF arm. Up to 15 degrees
    tool-axis error is accepted. A local failure is reported as 'unverified',
    never proof that a task is physically impossible.
    """
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Positive tolerance and iteration count are required")
    reference = np.asarray(reference_qpos, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if reference.shape != (model.nq,) or target.shape != (3,):
        raise ValueError("Expected full model qpos and three-value target position")
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError("IK inputs must be finite")
    site = model.site(site_name).id
    joint_ids = np.array([model.joint(name).id for name in JOINTS[:5]])
    qadr = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    checker = SelfCollisionChecker(model)
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    mujoco.mj_forward(model, data)
    target_axis = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
    position_jac = np.zeros((3, model.nv))
    rotation_jac = np.zeros((3, model.nv))
    axis_weight = 0.04
    best = None
    offsets = ([0, 0, 0, 0, 0], [0, 0.08, -0.08, 0, 0], [0, -0.08, 0.08, 0, 0])
    for restart, offset in enumerate(offsets):
        data.qpos[:] = reference
        data.qpos[qadr] = np.clip(reference[qadr] + offset, limits[:, 0], limits[:, 1])
        for iteration in range(max_iterations):
            mujoco.mj_forward(model, data)
            position = data.site_xpos[site].copy()
            axis = data.site_xmat[site].reshape(3, 3)[:, 2]
            error = target - position
            distance = float(np.linalg.norm(error))
            axis_error = float(np.arccos(np.clip(np.dot(axis, target_axis), -1, 1)))
            self_contact = checker.check(data)
            obstacle_contact = _obstacle_contact(model, data, checker)
            within_limits = bool(
                np.all(data.qpos[qadr] >= limits[:, 0] - 1e-8)
                and np.all(data.qpos[qadr] <= limits[:, 1] + 1e-8)
            )
            accepted = bool(
                distance <= tolerance
                and axis_error <= np.deg2rad(15)
                and self_contact is None
                and not obstacle_contact
                and within_limits
            )
            candidate = {
                "reachable": accepted,
                "position_error_m": distance,
                "axis_error_rad": axis_error,
                "qpos": data.qpos.tolist(),
                "arm_joint_degrees": np.rad2deg(data.qpos[qadr]).tolist(),
                "position": position.tolist(),
                "target_position": target.tolist(),
                "self_contact": self_contact,
                "obstacle_contact": obstacle_contact,
                "within_limits": within_limits,
                "restart": restart,
                "iterations": iteration + 1,
                "reason": "endpoint_witness" if accepted else "unverified_local_ik",
                "scope": "Endpoint only; not a collision-free trajectory or grasp proof",
            }
            score = (
                distance
                + 0.04 * axis_error
                + (0.1 if self_contact or obstacle_contact else 0)
            )
            if best is None or score < best[0]:
                best = (score, candidate)
            if accepted:
                return candidate
            mujoco.mj_jacSite(model, data, position_jac, rotation_jac, site)
            axis_jac = -_skew(axis) @ rotation_jac[:, dofs]
            jac = np.vstack((position_jac[:, dofs], axis_weight * axis_jac))
            residual = np.concatenate((error, axis_weight * (target_axis - axis)))
            delta = np.linalg.solve(
                jac.T @ jac + 0.005**2 * np.eye(5),
                jac.T @ residual + 1e-6 * (reference[qadr] - data.qpos[qadr]),
            )
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > 0.12:
                delta *= 0.12 / max_delta
            data.qpos[qadr] = np.clip(
                data.qpos[qadr] + delta, limits[:, 0], limits[:, 1]
            )
    return best[1]
