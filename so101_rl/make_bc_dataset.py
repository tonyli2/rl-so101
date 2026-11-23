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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-size", type=int, nargs=2, default=(96, 128), metavar=("H", "W"))
    parser.add_argument("--check-replay", type=int, default=3, help="episodes to re-simulate (0 to skip)")
    args = parser.parse_args()
    if not DATASET.is_dir():
        sys.exit(f"Dataset not found at {DATASET}; download and extract the sim-baseline-clean-50-v1 release")

    prepare_mounted(rebuild=True)
    env = So101PickPlaceEnv(action_mode="absolute", image=True, image_size=tuple(args.image_size), history=1,
                            wrist=True)
    replayer = ArchiveReplayer(env)
    items = episodes()

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
