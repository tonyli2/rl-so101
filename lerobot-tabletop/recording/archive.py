"""Versioned MuJoCo snapshots alongside native LeRobot learning datasets.

LeRobot owns its videos, numeric tables and episode metadata. This module only
stores the simulator information required to restore or re-render those rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

SCHEMA_VERSION = 1
STATE_MASK = int(mujoco.mjtState.mjSTATE_INTEGRATION)
TABLETOP_ROOT = Path(__file__).resolve().parents[1]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value) -> bytes:
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")

    return (
        json.dumps(value, indent=2, sort_keys=True, default=convert, allow_nan=False)
        + "\n"
    ).encode()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".pending-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _archive_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "sim_archive"


def _episode_name(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            "Episode UUID must contain only letters, digits, underscores and hyphens"
        )
    return value


def _layout(model) -> dict:
    return {
        "nq": model.nq,
        "nv": model.nv,
        "na": model.na,
        "nu": model.nu,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "joint_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
        ],
        "joint_types": model.jnt_type.tolist(),
        "joint_qposadr": model.jnt_qposadr.tolist(),
        "actuator_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ],
        "state_size": mujoco.mj_stateSize(model, STATE_MASK),
    }


def capture_state(model, data) -> np.ndarray:
    """Return an owned float64 integration-state snapshot, including controls."""
    state = np.empty(mujoco.mj_stateSize(model, STATE_MASK), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_MASK)
    if not np.isfinite(state).all():
        raise ValueError("Cannot archive a non-finite simulator state")
    return state


def restore_state(model, data, state: np.ndarray) -> None:
    """Restore a snapshot and refresh derived poses before rendering or checking it."""
    state = np.asarray(state, dtype=np.float64)
    expected = mujoco.mj_stateSize(model, STATE_MASK)
    if state.shape != (expected,) or not np.isfinite(state).all():
        raise ValueError(
            f"Expected a finite integration-state vector of length {expected}"
        )
    mujoco.mj_setState(model, data, state, STATE_MASK)
    mujoco.mj_forward(model, data)


def _source_files() -> dict[str, bytes]:
    """Preserve code/config and stock assets once, without caches or calibration secrets."""
    result = {}
    candidates = list(TABLETOP_ROOT.glob("*.py")) + [
        TABLETOP_ROOT / "so101_tabletop.xml"
    ]
    for relative in ("mounts", "robotstudio_so101", "textures", "recording"):
        directory = TABLETOP_ROOT / relative
        if directory.exists():
            candidates.extend(directory.rglob("*"))
    allowed = {".py", ".xml", ".json", ".stl", ".png"}
    for path in sorted(set(candidates)):
        if not path.is_file() or "__pycache__" in path.parts or ".cache" in path.parts:
            continue
        if path.suffix.lower() not in allowed and path.name != "LICENSE":
            continue
        result[path.relative_to(TABLETOP_ROOT).as_posix()] = path.read_bytes()
    return result


def _portable_xml(name: str, payload: bytes, names: set[str]) -> bytes:
    """Rewrite generated absolute mesh/texture references to bundled relative paths."""
    if not name.endswith(".xml"):
        return payload
    root = ET.fromstring(payload)
    for node in root.iter():
        filename = node.get("file")
        if filename and Path(filename).is_absolute():
            try:
                relative = (
                    Path(filename).resolve().relative_to(TABLETOP_ROOT).as_posix()
                )
            except ValueError as error:
                raise ValueError(
                    f"Scene asset lies outside project snapshot: {filename}"
                ) from error
            if relative not in names:
                raise ValueError(f"Missing source asset in snapshot: {filename}")
            node.set("file", os.path.relpath(relative, start=str(Path(name).parent)))
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def snapshot_scene(world, root: str | Path) -> str:
    """Save the actual runtime model and portable source bundle; return its content ID.

    Call after reset and timestep/camera configuration. If later episodes mutate
    MjModel parameters, snapshot again: those changes are not in mj_getState.
    Only ``world.model`` is required, so a SimpleNamespace(model=...) also works.
    """
    model = world.model
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    payload = binary.tobytes()
    sources = _source_files()
    source_hashes = {name: _sha256(data) for name, data in sources.items()}
    identity = {
        "mujoco_version": mujoco.__version__,
        "model_sha256": _sha256(payload),
        "source_hashes": source_hashes,
    }
    scene_id = _sha256(_json_bytes(identity))
    scenes = _archive_root(root) / "scenes"
    destination = scenes / scene_id
    if destination.is_dir():
        return scene_id
    scenes.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=scenes, prefix=".pending-"))
    try:
        (temporary / "model.mjb").write_bytes(payload)
        bundle_hashes = {}
        for name, content in sources.items():
            target = temporary / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            bundled = _portable_xml(name, content, set(sources))
            target.write_bytes(bundled)
            bundle_hashes[name] = _sha256(bundled)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id,
            **identity,
            "bundle_hashes": bundle_hashes,
            "state_mask": STATE_MASK,
            "state_mask_name": "mjSTATE_INTEGRATION",
            "layout": _layout(model),
            "timestep": float(model.opt.timestep),
        }
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=TABLETOP_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            metadata["git_commit"] = result.stdout.strip()
            result = subprocess.run(
                ["git", "diff", "--binary", "HEAD", "--", "."],
                cwd=TABLETOP_ROOT,
                capture_output=True,
                check=True,
            )
            (temporary / "tracked_changes.patch").write_bytes(result.stdout)
        except (OSError, subprocess.CalledProcessError):
            metadata["git_commit"] = None
        _atomic_json(temporary / "scene.json", metadata)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return scene_id


def _scene_metadata(root: str | Path, scene_id: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", scene_id):
        raise ValueError("Invalid scene content ID")
    directory = _archive_root(root) / "scenes" / scene_id
    metadata = json.loads((directory / "scene.json").read_text())
    if (
        metadata.get("scene_id") != scene_id
        or metadata.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported or inconsistent scene manifest")
    if metadata.get("mujoco_version") != mujoco.__version__:
        raise ValueError(
            f"Archive requires MuJoCo {metadata.get('mujoco_version')}; installed {mujoco.__version__}"
        )
    if metadata.get("state_mask") != STATE_MASK:
        raise ValueError(
            "Archive integration-state mask differs from this implementation"
        )
    if _sha256((directory / "model.mjb").read_bytes()) != metadata["model_sha256"]:
        raise ValueError("Compiled scene checksum mismatch")
    return directory, metadata


def _validate_arrays(arrays: dict, state_size: int) -> tuple[dict, int]:
    required = {"states", "raw_actions", "applied_actions", "controls"}
    if not required.issubset(arrays):
        raise ValueError(f"Missing archive arrays: {sorted(required - arrays.keys())}")
    result = {}
    for name, values in arrays.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid array key: {name}")
        array = np.asarray(values)
        if array.dtype.kind not in "biuf":
            raise ValueError(
                f"Archive arrays must be numeric, not {array.dtype}: {name}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in archive array {name}")
        result[name] = array
    states = np.asarray(result["states"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != state_size or len(states) < 2:
        raise ValueError(
            "states must contain T+1 integration snapshots, including terminal state"
        )
    count = len(states) - 1
    result["states"] = states
    for name in required - {"states"}:
        if result[name].shape != (count, 6):
            raise ValueError(f"{name} must have shape ({count}, 6)")
    for name in set(result) - required:
        if result[name].ndim == 0 or len(result[name]) not in (count, count + 1):
            raise ValueError(f"Auxiliary array {name} must have T or T+1 rows")
    return result, count


def write_episode_archive(
    root: str | Path,
    episode_uuid: str,
    episode_index: int,
    scene_id: str,
    arrays: dict,
    metadata: dict,
) -> Path:
    """Write recoverable sidecars before native dataset commit; never overwrite an episode."""
    identifier = _episode_name(episode_uuid)
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise ValueError("episode_index must be a nonnegative integer")
    _, scene = _scene_metadata(root, scene_id)
    arrays, count = _validate_arrays(arrays, scene["layout"]["state_size"])
    directory = _archive_root(root) / "episodes"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    data_path = path.with_suffix(".npz")
    if path.exists() or data_path.exists():
        raise FileExistsError(f"Episode archive already exists: {identifier}")
    for existing in directory.glob("*.json"):
        prior = json.loads(existing.read_text())
        if prior.get("episode_index") == episode_index:
            raise ValueError(
                f"Episode index {episode_index} already has an archive; reconcile it before resuming"
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode_uuid": identifier,
        "episode_index": episode_index,
        "scene_id": scene_id,
        "mujoco_version": mujoco.__version__,
        "state_mask": STATE_MASK,
        "state_size": scene["layout"]["state_size"],
        "transition_count": count,
        "metadata": metadata,
    }
    # Persist intent first so an interrupted NPZ write is discoverable.
    _atomic_json(path, manifest)
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".pending-", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, data_path)
    manifest["arrays_sha256"] = _sha256(data_path.read_bytes())
    manifest["arrays"] = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }
    _atomic_json(path, manifest)
    return path


def _find_episode(root: str | Path, episode: str | int) -> Path:
    directory = _archive_root(root) / "episodes"
    if isinstance(episode, int):
        matches = [
            path
            for path in directory.glob("*.json")
            if json.loads(path.read_text()).get("episode_index") == episode
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one archive for episode index {episode}, found {len(matches)}"
            )
        return matches[0]
    return directory / f"{_episode_name(str(episode))}.json"


def load_episode_archive(
    root: str | Path, episode: str | int, *, allow_pending: bool = False
):
    """Return (manifest, arrays, exact saved MjModel); reject incomplete/incompatible data."""
    path = _find_episode(root, episode)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported episode archive schema")
    if manifest.get("status") != "committed" and not (
        allow_pending and manifest.get("status") == "pending"
    ):
        raise ValueError(f"Episode {manifest.get('episode_uuid')} is not committed")
    directory, scene = _scene_metadata(root, manifest["scene_id"])
    if (
        manifest.get("mujoco_version") != mujoco.__version__
        or manifest.get("state_mask") != STATE_MASK
    ):
        raise ValueError("Episode MuJoCo version or state mask mismatch")
    data_path = path.with_suffix(".npz")
    if not data_path.is_file() or _sha256(data_path.read_bytes()) != manifest.get(
        "arrays_sha256"
    ):
        raise ValueError("Episode data missing, incomplete, or checksum mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays, count = _validate_arrays(arrays, scene["layout"]["state_size"])
    if count != manifest.get("transition_count") or scene["layout"][
        "state_size"
    ] != manifest.get("state_size"):
        raise ValueError("Episode row count or state layout mismatch")
    for name, array in arrays.items():
        if manifest.get("arrays", {}).get(name) != {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }:
            raise ValueError(f"Array metadata mismatch: {name}")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    if _layout(model) != scene["layout"]:
        raise ValueError("Compiled model layout differs from the archived layout")
    return manifest, arrays, model


def mark_committed(root: str | Path, episode_uuid: str) -> None:
    """Mark sidecar usable only after the native LeRobot episode save succeeds."""
    path = _find_episode(root, episode_uuid)
    manifest, _, _ = load_episode_archive(root, episode_uuid, allow_pending=True)
    manifest["status"] = "committed"
    _atomic_json(path, manifest)
