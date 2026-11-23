"""Convert Tony's demonstrations into So101PickPlaceEnv observations for behavior cloning.

Source: the ``sim-baseline-clean-50-v1`` release (GitHub release on tonyli2/rl-so101), extracted
to ``lerobot-tabletop/recordings/sim-baseline``. Every frame there has an exact MuJoCo integration
state (``sim_archive/episodes/*.npz``) and the compiled model it was recorded with.

Each state is restored in its archived model and copied, joint by joint, into the env's mounted
world (identical layout; the env only adds an invisible reach marker). The env then produces the
observation the policy would see at that moment: normalized joints, velocities and current targets,
plus the overhead and wrist cameras at the env's image size. The label is the control applied from that frame,
in the env's ``absolute`` action units ([-1, 1] over each actuator's range).

Output ``outputs/bc_data/sim-baseline.npz`` holds single frames; ``train_bc.py`` stacks the
3-step history within each episode, exactly as the env does at run time.

    python make_bc_dataset.py
    python make_bc_dataset.py --check-replay 5     # re-simulate 5 episodes against the archive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import mujoco
import numpy as np

from envs import So101PickPlaceEnv, prepare_mounted

HERE = Path(__file__).parent
DATASET = HERE.parent / "lerobot-tabletop" / "recordings" / "sim-baseline"
OUT = HERE / "outputs" / "bc_data" / "sim-baseline.npz"
PERTURBED = HERE / "outputs" / "bc_data" / "sim-baseline-perturbed.npz"
STATE_MASK = mujoco.mjtState.mjSTATE_INTEGRATION
TRAIN_EPISODES = range(0, 45)  # Tony's recommended split (release notes / dataset manifest)
VAL_EPISODES = range(45, 50)


class ArchiveReplayer:
    """Restores archived integration states and mirrors them into the env's world."""

    def __init__(self, env: So101PickPlaceEnv):
        self.env = env
        self.world = env.world
        self.models: dict[str, tuple[mujoco.MjModel, mujoco.MjData]] = {}

    def archived(self, scene_id: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if scene_id not in self.models:
            path = DATASET / "sim_archive" / "scenes" / scene_id / "model.mjb"
            model = mujoco.MjModel.from_binary_path(str(path))
            self._check_layout(model)
            self.models[scene_id] = (model, mujoco.MjData(model))
        return self.models[scene_id]

    def _check_layout(self, archived: mujoco.MjModel) -> None:
        mine = self.world.model
        names = lambda m: [m.joint(i).name for i in range(m.njnt)]  # noqa: E731
        if (archived.nq, archived.nv, archived.nu, names(archived)) != (mine.nq, mine.nv, mine.nu, names(mine)):
            raise RuntimeError("Archived model layout differs from the env's mounted model")
        if not np.isclose(archived.opt.timestep, mine.opt.timestep):
            raise RuntimeError("Archived physics timestep differs from the env's")

    def load(self, scene_id: str, state: np.ndarray) -> None:
        """Set the env world to an archived integration state."""
        model, data = self.archived(scene_id)
        mujoco.mj_setState(model, data, state, STATE_MASK)
        w = self.world.data
        w.time = data.time
        w.qpos[:] = data.qpos
        w.qvel[:] = data.qvel
        w.act[:] = data.act
        w.ctrl[:] = data.ctrl
        w.qacc_warmstart[:] = data.qacc_warmstart
        w.qfrc_applied[:] = data.qfrc_applied
        w.xfrc_applied[: model.nbody] = data.xfrc_applied
        self.world.set_target((0.45, 0.30, -0.2))  # the env hides the reach marker in this task
        mujoco.mj_forward(self.world.model, w)
        self.env._target_ctrl = self.world.ctrl  # the commanded targets are part of the observation

    def cube_touches_fingers(self) -> bool:
        w = self.world
        fingers = set(w.finger_geoms.tolist())
        for c in w.data.contact[: w.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if w.cube_geom in pair and pair & fingers:
                return True
        return False

    def perturb_arm(self, delta: np.ndarray) -> bool:
        """Offset the five arm joints by ``delta`` (radians) in the loaded state, for rendering only.

        A cube touching the fingers is carried rigidly with the gripper, so a held cube stays in the
        hand. Returns whether the cube was carried.
        """
        w = self.world
        m, d = w.model, w.data
        carried = self.cube_touches_fingers()
        if carried:
            g = m.body("gripper").id
            rot = d.xmat[g].reshape(3, 3).copy()
            p_rel = rot.T @ (d.xpos[w.cube_body] - d.xpos[g])
            q_inv, q_rel = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(q_inv, d.xquat[g])
            mujoco.mju_mulQuat(q_rel, q_inv, d.xquat[w.cube_body])
        q = d.qpos[w.qadr[:5]] + delta
        d.qpos[w.qadr[:5]] = np.clip(q, w.lo[:5], w.hi[:5])
        mujoco.mj_kinematics(m, d)
        if carried:
            a = w.cube_qadr
            d.qpos[a : a + 3] = d.xmat[g].reshape(3, 3) @ p_rel + d.xpos[g]
            mujoco.mju_mulQuat(d.qpos[a + 3 : a + 7], d.xquat[g], q_rel)
        mujoco.mj_forward(m, d)
        return carried


def episodes() -> list[dict]:
    items = []
    for meta_path in sorted((DATASET / "sim_archive" / "episodes").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if not meta["metadata"].get("policy_eligible", False):
            continue
        if meta["state_mask"] != int(STATE_MASK):
            raise RuntimeError(f"{meta_path.name}: state mask {meta['state_mask']} != {int(STATE_MASK)}")
        items.append({
            "index": int(meta["episode_index"]),
            "uuid": meta["episode_uuid"],
            "scene": meta["scene_id"],
            "npz": meta_path.with_suffix(".npz"),
        })
    items.sort(key=lambda e: e["index"])
    if [e["index"] for e in items] != list(range(50)):
        raise RuntimeError(f"Expected episodes 0..49, found {[e['index'] for e in items]}")
    return items


def to_action(world, controls: np.ndarray) -> np.ndarray:
    """Applied actuator controls (radians) -> the env's absolute action in [-1, 1]."""
    return np.clip(2 * (controls - world.lo) / (world.hi - world.lo) - 1, -1, 1).astype(np.float32)


def check_replay(replayer: ArchiveReplayer, items: list[dict], count: int) -> list[dict]:
    """Re-simulate episodes in the env world from their first state with the archived controls."""
    results = []
    w = replayer.world
    for item in items[:: max(1, len(items) // count)][:count]:
        arrays = np.load(item["npz"])
        states, controls = arrays["states"], arrays["controls"]
        replayer.load(item["scene"], states[0])
        model, data = replayer.archived(item["scene"])
        drift = []
        for t, ctrl in enumerate(controls):
            w.step(ctrl)
            mujoco.mj_setState(model, data, states[t + 1], STATE_MASK)
            drift.append(np.abs(w.data.qpos - data.qpos).max())
        results.append({
            "episode": item["index"], "steps": len(controls),
            "max_qpos_drift": float(max(drift)), "final_cube_error_m": float(
                np.linalg.norm(w.cube_pos() - data.qpos[w.cube_qadr : w.cube_qadr + 3])),
        })
        print(f"replay episode {item['index']:2d}: {len(controls)} steps, max qpos drift "
              f"{results[-1]['max_qpos_drift']:.2e}, final cube error {results[-1]['final_cube_error_m'] * 1000:.3f} mm")
    return results


def make_perturbed(replayer: ArchiveReplayer, items: list[dict], args) -> None:
    """Noise augmentation (DART-style): nudged arm poses labelled with the demonstrated targets.

    Actions are absolute joint targets on the demonstrated path, so from a nudged pose the original
    label means "steer back onto the path": the recovery behavior plain BC never sees.
    Training episodes only; validation stays clean.
    """
    env, rng = replayer.env, np.random.default_rng(args.seed)
    sigma = np.deg2rad(args.perturb_deg)
    states, images, wrist_images, actions, episode, frame, offsets = [], [], [], [], [], [], []
    carried_count, start = 0, time.time()
    for item in items:
        if item["index"] not in TRAIN_EPISODES:
            continue
        arrays = np.load(item["npz"])
        ep_states, controls = arrays["states"], arrays["controls"]
        for t in range(len(controls)):
            label = to_action(env.world, controls[t])
            for _ in range(args.perturb):
                replayer.load(item["scene"], ep_states[t])
                delta = rng.normal(0.0, sigma, 5)
                carried_count += replayer.perturb_arm(delta)
                obs = env._observe(fresh=True)
                states.append(obs["state"])
                images.append(obs["front"])
                wrist_images.append(obs["wrist"])
                actions.append(label)
                episode.append(item["index"])
                frame.append(t)
                offsets.append(delta)
        print(f"episode {item['index']:2d}: {len(controls) * args.perturb} perturbed frames", flush=True)
    PERTURBED.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PERTURBED,
        states=np.asarray(states, np.float32), images=np.asarray(images, np.uint8),
        wrist_images=np.asarray(wrist_images, np.uint8), actions=np.asarray(actions, np.float32),
        episode=np.asarray(episode, np.int16), frame=np.asarray(frame, np.int32),
        offsets=np.asarray(offsets, np.float32),
    )
    info = {"copies_per_frame": args.perturb, "perturb_deg_std": args.perturb_deg, "seed": args.seed,
            "frames": len(actions), "frames_with_cube_carried": int(carried_count),
            "episodes": sorted(set(episode)), "seconds": round(time.time() - start, 1)}
    PERTURBED.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\n{len(actions)} perturbed frames ({carried_count} with the cube carried) -> {PERTURBED} "
          f"({info['seconds']} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-size", type=int, nargs=2, default=(96, 128), metavar=("H", "W"))
    parser.add_argument("--check-replay", type=int, default=3, help="episodes to re-simulate (0 to skip)")
    parser.add_argument("--perturb", type=int, default=0, metavar="K",
                        help="noise augmentation: write K perturbed copies of every training frame to "
                             f"{PERTURBED.name} instead of the clean dataset")
    parser.add_argument("--perturb-deg", type=float, default=2.5, help="std of the arm-joint offsets (degrees)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not DATASET.is_dir():
        sys.exit(f"Dataset not found at {DATASET}; download and extract the sim-baseline-clean-50-v1 release")

    prepare_mounted(rebuild=True)
    env = So101PickPlaceEnv(action_mode="absolute", image=True, image_size=tuple(args.image_size), history=1,
                            wrist=True)
    replayer = ArchiveReplayer(env)
    items = episodes()

    if args.perturb:
        make_perturbed(replayer, items, args)
        env.close()
        return

    replay = check_replay(replayer, items, args.check_replay) if args.check_replay else []

    states, images, wrist_images, actions, episode, frame = [], [], [], [], [], []
    start = time.time()
    for item in items:
        arrays = np.load(item["npz"])
        ep_states, controls = arrays["states"], arrays["controls"]
        for t in range(len(controls)):  # the terminal state has no action
            replayer.load(item["scene"], ep_states[t])
            obs = env._observe(fresh=True)
            states.append(obs["state"])
            images.append(obs["front"])
            wrist_images.append(obs["wrist"])
            actions.append(to_action(env.world, controls[t]))
            episode.append(item["index"])
            frame.append(t)
        print(f"episode {item['index']:2d}: {len(controls)} frames", flush=True)
    env.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        states=np.asarray(states, np.float32), images=np.asarray(images, np.uint8),
        wrist_images=np.asarray(wrist_images, np.uint8),
        actions=np.asarray(actions, np.float32), episode=np.asarray(episode, np.int16),
        frame=np.asarray(frame, np.int32),
    )
    info = {
        "source": str(DATASET), "release": "sim-baseline-clean-50-v1", "frames": len(actions),
        "episodes": len(items), "train_episodes": list(TRAIN_EPISODES), "val_episodes": list(VAL_EPISODES),
        "image_size": list(args.image_size), "camera": env.camera, "wrist_camera": env.WRIST_CAMERA,
        "action_mode": "absolute",
        "state_dim": int(np.asarray(states).shape[1]), "replay_check": replay,
        "seconds": round(time.time() - start, 1),
        "episode_uuid": {e["index"]: e["uuid"] for e in items},
    }
    OUT.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\n{len(actions)} frames from {len(items)} episodes -> {OUT} ({info['seconds']} s)")


if __name__ == "__main__":
    main()
