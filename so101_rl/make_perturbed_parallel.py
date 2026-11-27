"""Noise-augmented copies (make_bc_dataset.py --perturb) rendered by several processes at once.

Rendering archived states runs at ~5 frames/s per process, so a 50-demo release takes hours in
one process. This splits the training episodes into --workers groups, renders each group in its own
process with make_bc_dataset.make_perturbed, then merges the parts (same seed per group offset).
--clean renders the clean dataset (all episodes, make_bc_dataset.render_clean) the same way.

    python make_perturbed_parallel.py --source <release root> --name physical-v1 --perturb 3 --workers 6
    python make_perturbed_parallel.py --source <release root> --name physical-v1 --clean --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "wgl")

import numpy as np

KEYS = ("states", "images", "wrist_images", "actions", "episode", "frame", "offsets")
CLEAN_KEYS = KEYS[:-1]


def work(job: dict) -> str:
    import make_bc_dataset as M
    from envs import So101PickPlaceEnv

    M.configure(Path(job["source"]), job["name"])
    M.TRAIN_EPISODES = job["episodes"]
    M.PERTURBED = Path(job["out"])
    env = So101PickPlaceEnv(action_mode="absolute", image=True, image_size=(96, 128), history=1, wrist=True)
    replayer = M.ArchiveReplayer(env)
    items = [i for i in M.episodes() if i["index"] in job["episodes"]]
    if job.get("clean"):
        np.savez(job["out"], **M.render_clean(replayer, items))
        env.close()
        return job["out"]
    args = SimpleNamespace(perturb=job["perturb"], perturb_deg=job["perturb_deg"], seed=job["seed"])
    M.make_perturbed(replayer, items, args)
    env.close()
    return job["out"]


def main() -> None:
    import make_bc_dataset as M
    from envs import prepare_mounted

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--perturb", type=int, default=3)
    parser.add_argument("--perturb-deg", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="render the clean dataset instead")
    args = parser.parse_args()

    prepare_mounted(rebuild=True)
    M.configure(args.source, args.name)
    if args.clean:
        train = sorted(i["index"] for i in M.episodes())
    else:
        train = list(M.TRAIN_EPISODES)
    groups = [train[k:: args.workers] for k in range(args.workers)]
    out_dir = M.PERTURBED.parent
    stem = args.name if args.clean else f"{args.name}-perturbed"
    keys = CLEAN_KEYS if args.clean else KEYS
    jobs = [{"source": str(args.source), "name": args.name, "episodes": g, "perturb": args.perturb,
             "perturb_deg": args.perturb_deg, "seed": args.seed + 1000 * k, "clean": args.clean,
             "out": str(out_dir / f"{stem}.part{k}.npz")} for k, g in enumerate(groups)]
    start = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(work, jobs)
    merged = {key: [] for key in keys}
    for part in parts:
        with np.load(part) as d:  # closed afterwards, so Windows lets us delete the part files
            for key in keys:
                merged[key].append(d[key])
    if args.clean:  # back in recorded order: episode, then frame
        order = np.lexsort((np.concatenate(merged["frame"]), np.concatenate(merged["episode"])))
        data = {key: np.concatenate(v)[order] for key, v in merged.items()}
        final = out_dir / f"{args.name}.npz"
        np.savez(final, **data)
        info_path = final.with_suffix(".json")
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        info.update({"source": str(M.DATASET), "release": str(M.DATASET.parent.name), "frames": len(order),
                     "layout": M.LAYOUT, "episodes": len(train), "train_episodes": list(M.TRAIN_EPISODES),
                     "val_episodes": list(M.VAL_EPISODES), "action_mode": "absolute",
                     "state_dim": int(data["states"].shape[1]), "seconds": round(time.time() - start, 1),
                     "workers": args.workers, "render": "opaque paper curtains (sensor view)"})
        info_path.write_text(json.dumps(info, indent=2))
        for part in parts:
            Path(part).unlink()
        print(f"{len(order)} clean frames -> {final}")
        return
    final = out_dir / f"{args.name}-perturbed.npz"
    np.savez(final, **{key: np.concatenate(v) for key, v in merged.items()})
    info = {"copies_per_frame": args.perturb, "perturb_deg_std": args.perturb_deg, "seed": args.seed,
            "workers": args.workers, "frames": int(sum(len(v) for v in merged["frame"])),
            "episodes": train}
    final.with_suffix(".json").write_text(json.dumps(info, indent=2))
    for part in parts:
        Path(part).unlink()
        Path(part).with_suffix(".json").unlink(missing_ok=True)
    print(f"{info['frames']} perturbed frames -> {final}")


if __name__ == "__main__":
    main()
