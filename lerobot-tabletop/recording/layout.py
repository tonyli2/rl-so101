"""Rejection-sampled task layouts, using existing endpoint IK and MuJoCo rendering."""

from pathlib import Path

import mujoco
import numpy as np
from learning.reachability import find_reachability_witness, select_reference_pose
from mounts.enclosure import update_paper_scene

from .archive import load_episode_archive, restore_state
from .self_collision import SelfCollisionChecker


class LayoutSampler:
    protocol = {
        "version": 1,
        "cube_xy_half_range_m": 0.04,
        "bin_away_range_m": [0.0, 0.04],
        "bin_side_half_range_m": 0.04,
        "visibility": "overhead segmentation: >=16 pixels, 2px frame margin",
        "reachability": "collision-checked endpoints only, not path guarantees",
    }

    def __init__(self, world, reference_root, seed, width, height, start_episode=0):
        self.world = world
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, start_episode]))
        self.width, self.height = width, height
        model = world.model
        self.cube = model.body("red_cube").id
        self.box = model.body("box").id
        self.qa = int(model.jnt_qposadr[model.body_jntadr[self.cube]])
        self.cube0 = model.body_pos[self.cube].copy()
        self.bin0 = model.body_pos[self.box].copy()
        direction = self.bin0[:2] - self.cube0[:2]
        self.away = direction / np.linalg.norm(direction)
        self.side = np.array([-self.away[1], self.away[0]])
        _, arrays, reference_model = load_episode_archive(Path(reference_root), 0)
        if reference_model.nq != model.nq or reference_model.nv != model.nv:
            raise ValueError("Reference dataset does not match collection model layout")
        # Choose seeds in their original model, not the changed arm/base frame.
        # Then translate demonstrated tool offsets to the current task positions.
        reference_data = mujoco.MjData(reference_model)
        restore_state(reference_model, reference_data, arrays["states"][0])
        self.cube_reference = select_reference_pose(
            reference_model,
            arrays["states"],
            reference_data.xpos[reference_model.body("red_cube").id],
        )
        self.bin_reference = select_reference_pose(
            reference_model,
            arrays["states"],
            reference_data.xpos[reference_model.body("box").id],
        )
        for reference, current, name in (
            (self.cube_reference, self.cube0, "red_cube"),
            (self.bin_reference, self.bin0, "box"),
        ):
            reference["position"] = (
                np.asarray(reference["position"])
                + current
                - reference_data.xpos[reference_model.body(name).id]
            ).tolist()
        self.checker = SelfCollisionChecker(model)
        self.current = None
        self.renderer = mujoco.Renderer(world.model, height=height, width=width)
        self.renderer.enable_segmentation_rendering()

    def visible(self):
        update_paper_scene(self.renderer, self.world.model, self.world.data, "overhead")
        image = self.renderer.render()
        model = self.world.model
        geom_ids = np.flatnonzero(model.geom_bodyid == self.cube)
        mask = (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            image[:, :, 0], geom_ids
        )
        y, x = np.where(mask)
        return bool(
            len(x) >= 16
            and x.min() >= 2
            and y.min() >= 2
            and x.max() < self.width - 2
            and y.max() < self.height - 2
        )

    def validate_start(self):
        data, model = self.world.data, self.world.model
        mujoco.mj_forward(model, data)
        if not self.visible():
            return "Red cube is not sufficiently visible in overhead camera; move the arm or reset"
        # A preview motion must not move the sampled task before recording starts.
        if (
            self.current is not None
            and np.linalg.norm(
                data.xpos[self.cube][:2] - np.array(self.current["cube_position"])[:2]
            )
            > 0.002
        ):
            return "Cube moved during preview; reset before recording"
        for contact in data.contact:
            bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
            if (
                contact.dist <= 0
                and bodies.intersection(self.checker.robot_bodies)
                and bodies.intersection({self.cube, self.box})
            ):
                return "Robot touches cube/bin at recording start; move clear or reset"
        return None

    def sample(self):
        model, data = self.world.model, self.world.data
        for attempt in range(300):
            cube_shift = self.rng.uniform(-0.04, 0.04, 2)
            away, side = self.rng.uniform(0, 0.04), self.rng.uniform(-0.04, 0.04)
            bin_shift = away * self.away + side * self.side
            cube, box = self.cube0.copy(), self.bin0.copy()
            cube[:2] += cube_shift
            box[:2] += bin_shift
            # Conservative axis-aligned outer bin bounds plus cube and 1cm clearance.
            if np.all(np.abs(cube[:2] - box[:2]) < 0.049 + 0.015 + 0.01):
                continue
            if np.any(np.abs(cube[:2] - [0.2, 0]) + 0.015 >= 0.4) or np.any(
                np.abs(box[:2] - [0.2, 0]) + 0.049 >= 0.4
            ):
                continue
            model.body_pos[self.box] = box
            data.qpos[self.qa : self.qa + 3] = cube
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            if not self.visible():
                continue
            witnesses = []
            for reference, shift in (
                (self.cube_reference, cube_shift),
                (self.bin_reference, bin_shift),
            ):
                target = np.array(reference["position"])
                target[:2] += shift
                witnesses.append(
                    find_reachability_witness(model, reference["qpos"], target)
                )
            if not all(w["reachable"] for w in witnesses):
                continue
            self.current = {
                "protocol": self.protocol,
                "cube_position": cube.tolist(),
                "bin_position": box.tolist(),
                "cube_delta_m": cube_shift.tolist(),
                "bin_away_m": float(away),
                "bin_side_m": float(side),
                "rng_state": self.rng.bit_generator.state,
                "endpoint_witnesses": witnesses,
            }
            return self.current
        raise RuntimeError(
            "No visible, separated, endpoint-verified layout after 300 attempts"
        )

    def close(self):
        self.renderer.close()
