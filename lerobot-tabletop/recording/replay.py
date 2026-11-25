"""Native-dataset validation and MuJoCo state-based replay/re-rendering."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from mounts.enclosure import update_paper_scene

from .archive import (
    load_episode_archive,
    mark_committed,
    restore_state,
    snapshot_scene,
    write_episode_archive,
)


def _episodes(root, episode=None, *, allow_pending=False):
    directory = root / "sim_archive" / "episodes"
    if not directory.is_dir():
        raise FileNotFoundError(f"No simulator archive at {directory}")
    manifests = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    if not allow_pending and any(
        item.get("status") != "committed" for item in manifests
    ):
        raise ValueError(
            "Pending or incomplete episode archives exist; reconcile them before using this dataset"
        )
    allowed = {"committed", "pending"} if allow_pending else {"committed"}
    selected = [item for item in manifests if item.get("status") in allowed]
    if episode is not None:
        selected = [
            item
            for item in selected
            if item["episode_uuid"] == str(episode)
            or str(item["episode_index"]) == str(episode)
        ]
    if not selected:
        raise ValueError("No matching committed episodes")
    return sorted(selected, key=lambda item: item["episode_index"])


def _metadata(manifest):
    metadata = manifest["metadata"]
    required = ("fps", "height", "width", "camera_map", "task", "substeps")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Episode {manifest['episode_uuid']} lacks metadata {missing}")
    return metadata


def _reader(root):
    from lerobot.datasets import LeRobotDataset

    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a finalized local LeRobot dataset: {root}")
    return LeRobotDataset(
        repo_id=f"local/{root.name}", root=root, return_uint8=True, video_backend="pyav"
    )


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def _joint_state(model, data):
    from env import JOINTS

    values = np.array([data.qpos[model.joint(name).qposadr[0]] for name in JOINTS])
    result = np.rad2deg(values)
    low, high = model.actuator_ctrlrange[model.actuator("gripper").id]
    result[5] = (values[5] - low) / (high - low) * 100
    return result.astype(np.float32)


def _observation(model, data, renderer, metadata):
    images = {}
    for key, name in metadata["camera_map"].items():
        update_paper_scene(renderer, model, data, name)
        images[key] = renderer.render().copy()
    return {"agent_pos": _joint_state(model, data), "pixels": images}


def check_append_metadata(root):
    """Cheap append checks; full video/state validation remains an explicit tool."""
    for marker in ("COLLECTION_INCOMPLETE.json", "RERENDER_INCOMPLETE.json"):
        if (root / marker).exists():
            raise ValueError(
                f"Incomplete operation: {marker} exists; inspect before resuming"
            )
    info = json.loads((root / "meta/info.json").read_text())
    manifests = _episodes(root)
    indices = [item["episode_index"] for item in manifests]
    if indices != list(range(info["total_episodes"])):
        raise ValueError("Dataset/archive episode indices differ")
    if sum(item["transition_count"] for item in manifests) != info["total_frames"]:
        raise ValueError("Dataset/archive frame counts differ")
    for item in manifests:
        archive = root / "sim_archive/episodes" / f"{item['episode_uuid']}.npz"
        if not archive.is_file():
            raise ValueError(f"Missing episode archive: {archive}")


def validate(root: Path, episode=None, *, compare_images=False, allow_pending=False):
    """Check native decoding and row/archive labels; compare images only on request."""
    if not allow_pending and (root / "COLLECTION_INCOMPLETE.json").exists():
        raise ValueError(
            "Incomplete collection attempt: COLLECTION_INCOMPLETE.json exists; inspect before resuming"
        )
    if not allow_pending and (root / "RERENDER_INCOMPLETE.json").exists():
        raise ValueError(
            "Incomplete re-render export: RERENDER_INCOMPLETE.json exists; inspect before resuming"
        )
    dataset = _reader(root)
    all_manifests = _episodes(root, allow_pending=allow_pending)
    if sum(item["transition_count"] for item in all_manifests) != len(dataset):
        raise ValueError(
            "Dataset frame count differs from committed archive transition count"
        )
    if len(all_manifests) != dataset.num_episodes:
        raise ValueError(
            "Dataset episode count differs from committed archive episode count"
        )
    episodes = np.asarray(dataset.hf_dataset["episode_index"])
    reports = []
    for item in _episodes(root, episode, allow_pending=allow_pending):
        manifest, arrays, model = load_episode_archive(
            root, item["episode_uuid"], allow_pending=allow_pending
        )
        metadata = _metadata(manifest)
        indices = np.flatnonzero(episodes == manifest["episode_index"])
        count = manifest["transition_count"]
        if len(indices) != count:
            raise ValueError(
                f"Episode {manifest['episode_index']}: native/archive frame count mismatch"
            )
        if float(dataset.fps) != float(metadata["fps"]):
            raise ValueError("Dataset/archive FPS mismatch")
        data = mujoco.MjData(model)
        renderer = (
            mujoco.Renderer(model, metadata["height"], metadata["width"])
            if compare_images
            else None
        )
        errors = {key: [] for key in metadata["camera_map"]}
        try:
            for frame_index, native_index in enumerate(indices):
                frame = dataset[int(native_index)]
                if int(_numpy(frame["frame_index"]).item()) != frame_index:
                    raise ValueError("Non-contiguous native frame index")
                if not np.allclose(
                    _numpy(frame["action"]),
                    arrays["applied_actions"][frame_index],
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Action mismatch at episode/frame {manifest['episode_index']}/{frame_index}"
                    )
                restore_state(model, data, arrays["states"][frame_index])
                if not np.allclose(
                    _numpy(frame["observation.state"]),
                    _joint_state(model, data),
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Observation state mismatch at frame {frame_index}"
                    )
                for key, name in metadata["camera_map"].items():
                    image = _numpy(frame[f"observation.images.{key}"])
                    if image.shape == (3, metadata["height"], metadata["width"]):
                        image = image.transpose(1, 2, 0)
                    if image.shape != (metadata["height"], metadata["width"], 3):
                        raise ValueError(
                            f"Camera {key} has incorrect decoded dimensions {image.shape}"
                        )
                    if not np.isfinite(image).all():
                        raise ValueError(f"Camera {key} has non-finite pixels")
                    if renderer is not None:
                        update_paper_scene(renderer, model, data, name)
                        reference = renderer.render()
                        # MP4 is lossy; report the difference, do not require pixel identity.
                        errors[key].append(
                            float(
                                np.abs(
                                    reference.astype(float) - image.astype(float)
                                ).mean()
                            )
                        )
            restore_state(model, data, arrays["states"][-1])
        finally:
            if renderer is not None:
                renderer.close()
        report = {
            "episode_uuid": manifest["episode_uuid"],
            "frames": count,
            "status": "PASS",
        }
        if compare_images:
            report["mean_absolute_pixel_error_0_255"] = {
                key: float(np.mean(value)) for key, value in errors.items()
            }
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def replay(root: Path, episode=None, *, mode="states", headless=False, tolerance=1e-4):
    """State replay shows archived motion; command replay measures dynamics drift."""
    reports = []
    for item in _episodes(root, episode):
        manifest, arrays, model = load_episode_archive(root, item["episode_uuid"])
        metadata = _metadata(manifest)
        data, reference = mujoco.MjData(model), mujoco.MjData(model)
        restore_state(model, data, arrays["states"][0])
        viewer = None
        try:
            if not headless:
                from mujoco import viewer as viewer_module

                viewer = viewer_module.launch_passive(model, data)
                viewer.opt.geomgroup[5] = 1
            maximum_error = 0.0
            replayed = 0
            for index in range(manifest["transition_count"]):
                if viewer is not None and not viewer.is_running():
                    break
                start = time.monotonic()
                if mode == "states":
                    if viewer is None:
                        restore_state(model, data, arrays["states"][index + 1])
                    else:
                        with viewer.lock():
                            restore_state(model, data, arrays["states"][index + 1])
                else:
                    from env import JOINTS

                    actuator_ids = [model.actuator(name).id for name in JOINTS]
                    if viewer is None:
                        data.ctrl[actuator_ids] = arrays["controls"][index]
                        mujoco.mj_step(model, data, nstep=int(metadata["substeps"]))
                        mujoco.mj_forward(model, data)
                    else:
                        with viewer.lock():
                            data.ctrl[actuator_ids] = arrays["controls"][index]
                            mujoco.mj_step(model, data, nstep=int(metadata["substeps"]))
                            mujoco.mj_forward(model, data)
                    restore_state(model, reference, arrays["states"][index + 1])
                    maximum_error = max(
                        maximum_error, float(np.max(np.abs(data.qpos - reference.qpos)))
                    )
                if not np.isfinite(data.qpos).all():
                    raise ValueError("Non-finite replay physics state")
                replayed += 1
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0, 1 / metadata["fps"] - (time.monotonic() - start)))
            report = {
                "episode_uuid": manifest["episode_uuid"],
                "mode": mode,
                "maximum_qpos_absolute_error": maximum_error,
                "frames_replayed": replayed,
                "status": "PASS"
                if replayed == manifest["transition_count"]
                else "STOPPED",
            }
            if mode == "commands" and maximum_error > tolerance:
                raise ValueError(
                    f"Command replay qpos error {maximum_error:.6g} exceeds tolerance {tolerance}"
                )
            reports.append(report)
            print(json.dumps(report), flush=True)
        finally:
            if viewer is not None:
                viewer.close()
    return reports


def _camera_overrides(model, overrides):
    for name, settings in overrides.items():
        try:
            camera = model.camera(name)
        except KeyError as error:
            raise ValueError(f"Unknown camera {name!r}") from error
        if not isinstance(settings, dict):
            raise ValueError(f"Camera {name}: settings must be an object")
        unknown = set(settings) - {"pos", "quat", "fovy"}
        if unknown:
            raise ValueError(
                f"Camera {name}: unsupported override keys {sorted(unknown)}"
            )
        for field, shape in (("pos", (3,)), ("quat", (4,))):
            if field in settings:
                value = np.asarray(settings[field], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Camera {name}: invalid {field}")
                if field == "quat":
                    length = np.linalg.norm(value)
                    if length == 0:
                        raise ValueError(f"Camera {name}: zero quaternion")
                    value = value / length
                getattr(camera, field)[:] = value
        if "fovy" in settings:
            value = float(settings["fovy"])
            if not np.isfinite(value) or not 0 < value < 180:
                raise ValueError(
                    f"Camera {name}: fovy must be between 0 and 180 degrees"
                )
            camera.fovy[0] = value


def rerender(
    root: Path,
    output: Path,
    episode=None,
    overrides_path=None,
    *,
    height=None,
    width=None,
):
    """Restore archived states into a NEW local LeRobot dataset; never overwrite source."""
    from lerobot.datasets import VideoEncodingManager

    from .schema import check_encoder, create_dataset, learning_frame

    if output.exists():
        raise ValueError(f"Output already exists; choose a new dataset root: {output}")
    selected = _episodes(root, episode)
    overrides = json.loads(Path(overrides_path).read_text()) if overrides_path else {}
    if not isinstance(overrides, dict):
        raise ValueError(
            "Camera overrides must be a JSON object keyed by MuJoCo camera name"
        )
    first, _, _ = load_episode_archive(root, selected[0]["episode_uuid"])
    settings = _metadata(first)
    output_height = int(height if height is not None else settings["height"])
    output_width = int(width if width is not None else settings["width"])
    dataset = None
    completed = []
    try:
        for output_index, item in enumerate(selected):
            dataset = create_dataset(
                output,
                f"local/{output.name}",
                int(settings["fps"]),
                output_height,
                output_width,
                resume=output_index > 0,
            )
            incomplete = output / "RERENDER_INCOMPLETE.json"
            incomplete.write_text(
                json.dumps(
                    {"source": str(root), "camera_overrides": overrides}, indent=2
                )
            )
            with VideoEncodingManager(dataset):
                manifest, arrays, model = load_episode_archive(
                    root, item["episode_uuid"]
                )
                metadata = _metadata(manifest)
                for key in ("fps", "height", "width", "camera_map"):
                    if metadata[key] != settings[key]:
                        raise ValueError(
                            f"Episode schema differs for {key}; export compatible groups separately"
                        )
                _camera_overrides(model, overrides)
                model.vis.global_.offheight = max(
                    model.vis.global_.offheight, output_height
                )
                model.vis.global_.offwidth = max(
                    model.vis.global_.offwidth, output_width
                )
                scene_id = snapshot_scene(SimpleNamespace(model=model), output)
                data = mujoco.MjData(model)
                renderer = mujoco.Renderer(model, output_height, output_width)
                try:
                    for index in range(manifest["transition_count"]):
                        restore_state(model, data, arrays["states"][index])
                        observation = _observation(model, data, renderer, metadata)
                        dataset.add_frame(
                            learning_frame(
                                observation,
                                arrays["applied_actions"][index],
                                metadata["task"],
                            )
                        )
                    derived = {
                        **metadata,
                        "source_episode_uuid": manifest["episode_uuid"],
                        "source_episode_index": manifest["episode_index"],
                        "source_dataset": str(root),
                        "source_scene_id": manifest["scene_id"],
                        "camera_overrides": overrides,
                        "height": output_height,
                        "width": output_width,
                    }
                    write_episode_archive(
                        output,
                        manifest["episode_uuid"],
                        output_index,
                        scene_id,
                        arrays,
                        derived,
                    )
                    check_encoder(dataset)
                    dataset.save_episode()
                finally:
                    renderer.close()
            dataset.finalize()
            validate(output, manifest["episode_uuid"], allow_pending=True)
            mark_committed(output, manifest["episode_uuid"])
            completed.append(manifest["episode_uuid"])
        incomplete.unlink()
        print(f"Re-rendered {len(completed)} episode(s) to {output}", flush=True)
    except BaseException:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        raise
    return output
