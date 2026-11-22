"""Split flagged episodes with native LeRobot tools, preserving exact simulator archives.

Preparation is non-destructive. Commit only renames verified directories, keeping
an untouched original dataset outside recordings for historical evaluation.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import split_dataset
from recording.replay import check_append_metadata, validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifests(root):
    result = {}
    for path in (root / "sim_archive/episodes").glob("*.json"):
        item = json.loads(path.read_text())
        if item["episode_index"] in result:
            raise ValueError("Duplicate archive episode index")
        result[item["episode_index"]] = (path, item)
    return result


def _copy_archives(source, target, selected, *, quarantined, backup, audit):
    originals = _manifests(source)
    episodes = target / "sim_archive/episodes"
    episodes.mkdir(parents=True)
    scenes = set()
    mapping = []
    for new, old in enumerate(sorted(selected)):
        path, original = originals[old]
        item = json.loads(json.dumps(original))
        item["original_episode_index"] = old
        item["episode_index"] = new
        metadata = item["metadata"]
        metadata["original_episode_index"] = old
        metadata["original_repo_id"] = metadata.get("repo_id")
        metadata["repo_id"] = f"local/{target.name}"
        metadata["collision_review"] = {
            "status": "quarantined"
            if quarantined
            else "passed_original_model_substep_audit",
            "scope": "all nonadjacent robot body pairs at all physics substeps under the original archived model",
            "audit_path": str(audit),
            "audit_sha256": digest(audit),
            "self_contact_detected": quarantined,
            "scope_limitation": "This certifies the audited saved dynamics/model/rule, not changed scenes or physical hardware",
        }
        if quarantined:
            metadata["policy_eligible"] = False
            metadata["quarantine_reason"] = (
                "robot self-contact detected by the full original-model physics-substep audit"
            )
        (episodes / path.name).write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(path.with_suffix(".npz"), episodes / path.with_suffix(".npz").name)
        scenes.add(item["scene_id"])
        mapping.append(
            {
                "original_episode_index": old,
                "new_episode_index": new,
                "episode_uuid": item["episode_uuid"],
                "frames": item["transition_count"],
                "quarantined": quarantined,
            }
        )
    for scene in scenes:
        shutil.copytree(
            source / "sim_archive/scenes" / scene, target / "sim_archive/scenes" / scene
        )
    report = {
        "schema_version": 1,
        "source_root": str(source),
        "original_backup_root": str(backup),
        "source_total_episodes": len(originals),
        "source_total_frames": sum(
            x[1]["transition_count"] for x in originals.values()
        ),
        "split": "quarantined" if quarantined else "retained",
        "original_to_new": {
            str(x["original_episode_index"]): x["new_episode_index"] for x in mapping
        },
        "episodes": mapping,
    }
    (target / "episode_migration.json").write_text(json.dumps(report, indent=2) + "\n")
    if quarantined:
        (target / "QUARANTINED.json").write_text(
            json.dumps(
                {
                    "reason": "Simulation self-contact; excluded from the active training dataset",
                    "policy_eligible": False,
                    "original_episode_indices": sorted(selected),
                },
                indent=2,
            )
            + "\n"
        )
        (target / "README.md").write_text(
            "# Quarantined simulation demonstrations\n\n"
            "These episodes contain detected robot self-contact and were removed from sim-baseline.\n"
            "Do not use this directory for training without reviewing the collision issue.\n"
            "Native LeRobot records, videos and exact MuJoCo archives remain readable.\n"
            "Original-to-current episode numbers and UUIDs are in episode_migration.json.\n"
        )
    return report


def _verify_split(source, target, mapping):
    """Check all numeric records and archives, decode all samples once, and hash copied video bytes."""
    check_append_metadata(target)
    source_meta = LeRobotDataset("local/source", root=source, video_backend="pyav").meta
    target_ds = LeRobotDataset("local/split", root=target, video_backend="pyav")
    target_meta = target_ds.meta
    expected_frames = sum(row["frames"] for row in mapping["episodes"])
    if (
        target_meta.total_episodes != len(mapping["episodes"])
        or len(target_ds) != expected_frames
    ):
        raise ValueError("Split episode/frame totals differ")
    video_hashes = []
    for row in mapping["episodes"]:
        old, new = row["original_episode_index"], row["new_episode_index"]
        for key in source_meta.video_keys:
            src = source / source_meta.get_video_file_path(old, key)
            dst = target / target_meta.get_video_file_path(new, key)
            if digest(src) != digest(dst):
                raise ValueError(
                    f"Video was changed or reencoded: episode {old}, {key}"
                )
            video_hashes.append(str(dst.relative_to(target)))
    # Archive validator checks every native action/state and decodes both cameras at every frame.
    reports = validate(target)
    return {
        "episodes": len(reports),
        "frames": expected_frames,
        "unchanged_videos": len(set(video_hashes)),
        "native_archive_validation": "PASS",
    }


def prepare(source, quarantine, staging, backup, indices, audit):
    source, quarantine, staging, backup = [
        p.expanduser().resolve() for p in (source, quarantine, staging, backup)
    ]
    if staging.exists() or quarantine.exists() or backup.exists():
        raise FileExistsError(
            "Staging, quarantine and original backup destinations must be unused"
        )
    check_append_metadata(source)
    audit = audit.expanduser().resolve()
    audit_rows = json.loads(audit.read_text())
    audited_bad = sorted(
        row["episode"]
        for row in audit_rows
        if row["self_contact_transitions"] or row["saved_state_contact_indices"]
    )
    info = json.loads((source / "meta/info.json").read_text())
    flagged = sorted(set(indices))
    if (
        sorted(row["episode"] for row in audit_rows)
        != list(range(info["total_episodes"]))
        or sum(row["transitions"] for row in audit_rows) != info["total_frames"]
        or flagged != audited_bad
    ):
        raise ValueError(
            "Selection does not match the complete authoritative substep audit"
        )
    if any(row["max_replay_qpos_error_native"] != 0 for row in audit_rows):
        raise ValueError(
            "Substep audit did not reproduce the archived dynamics exactly"
        )
    if not flagged or any(i < 0 or i >= info["total_episodes"] for i in flagged):
        raise ValueError("Invalid episode selection")
    retained = sorted(set(range(info["total_episodes"])) - set(flagged))
    if not retained:
        raise ValueError("Cannot quarantine every episode")
    dataset = LeRobotDataset("local/sim-baseline", root=source, video_backend="pyav")
    # Copy-only condition: each source MP4 must belong to exactly one episode.
    for key in dataset.meta.video_keys:
        paths = [
            str(dataset.meta.get_video_file_path(ep, key))
            for ep in range(info["total_episodes"])
        ]
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Source has shared videos; refusing a migration that could reencode"
            )
    split_dataset(
        dataset, {source.name: retained, quarantine.name: flagged}, output_dir=staging
    )
    reports = {}
    for name, selected, bad in (
        (source.name, retained, False),
        (quarantine.name, flagged, True),
    ):
        target = staging / name
        mapping = _copy_archives(
            source, target, selected, quarantined=bad, backup=backup, audit=audit
        )
        reports[name] = _verify_split(source, target, mapping)
    report = {
        "schema_version": 1,
        "source": str(source),
        "quarantine": str(quarantine),
        "staging": str(staging),
        "original_backup": str(backup),
        "flagged_original_indices": flagged,
        "substep_audit": str(audit),
        "substep_audit_sha256": digest(audit),
        "source_info_sha256": digest(source / "meta/info.json"),
        "verification": reports,
        "status": "PREPARED_NOT_COMMITTED",
    }
    (staging / "migration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def commit(staging):
    staging = staging.expanduser().resolve()
    manifest_path = staging / "migration.json"
    report = json.loads(manifest_path.read_text())
    if report["status"] != "PREPARED_NOT_COMMITTED":
        raise ValueError("Migration not in prepared state")
    source, quarantine, backup = [
        Path(report[k]) for k in ("source", "quarantine", "original_backup")
    ]
    if quarantine.exists() or backup.exists():
        raise FileExistsError("Refusing to overwrite an existing destination")
    check_append_metadata(source)
    if digest(source / "meta/info.json") != report["source_info_sha256"]:
        raise ValueError("Source changed since preparation")
    for name in (source.name, quarantine.name):
        check_append_metadata(staging / name)
    backup.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem renames preserve every original byte in the historical backup.
    os.rename(source, backup)
    try:
        os.rename(staging / source.name, source)
        os.rename(staging / quarantine.name, quarantine)
    except BaseException:
        if source.exists():
            os.rename(source, staging / source.name)
        os.rename(backup, source)
        raise
    report["status"] = "COMMITTED"
    final_report = source.parent / "sim-baseline-quarantine-migration.json"
    final_report.write_text(json.dumps(report, indent=2) + "\n")
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--quarantine", type=Path, required=True)
    prep.add_argument("--staging", type=Path, required=True)
    prep.add_argument("--backup", type=Path, required=True)
    prep.add_argument("--indices", nargs="+", type=int, required=True)
    prep.add_argument("--audit", type=Path, required=True)
    apply = sub.add_parser("commit")
    apply.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(
            args.source,
            args.quarantine,
            args.staging,
            args.backup,
            args.indices,
            args.audit,
        )
    else:
        commit(args.staging)


if __name__ == "__main__":
    main()
