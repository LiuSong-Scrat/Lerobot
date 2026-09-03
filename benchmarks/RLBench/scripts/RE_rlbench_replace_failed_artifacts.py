#!/usr/bin/env python3
"""Safely replace failed RLBench artifact slots with replay-validated candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path


COMPATIBILITY_KEYS = (
    "variation",
    "num_points",
    "gripper_points",
    "gripper_max_width",
    "gripper_template",
    "gripper_template_version",
    "virtual_gripper_width_normalization_max_m",
    "virtual_gripper_geometry_max_width_m",
    "virtual_gripper_opening_max_width_m",
    "virtual_gripper_len_m",
    "image_size",
    "fps",
    "action_semantics_version",
    "action_label_mode",
    "action_alignment",
    "scene_bounds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-result-root", type=Path, required=True)
    parser.add_argument("--target-replay-summary", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        required=True,
        metavar="TARGET_TASK:EP=CANDIDATE_TASK:EP",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object: " + str(path))
    return value


def parse_slot(text: str) -> tuple[str, int]:
    task, separator, episode = text.rpartition(":")
    if not separator or not task:
        raise ValueError("Expected TASK:EP, got " + repr(text))
    return task, int(episode)


def parse_mapping(text: str) -> tuple[tuple[str, int], tuple[str, int]]:
    target, separator, candidate = text.partition("=")
    if not separator:
        raise ValueError("Expected TARGET=CANDIDATE, got " + repr(text))
    return parse_slot(target), parse_slot(candidate)


def episode_name(task: str, episode: int) -> str:
    return task + "__episode_" + str(episode).zfill(5)


def record_by_slot(manifest: dict) -> dict[tuple[str, int], dict]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Manifest records must be a list")
    return {
        (str(record["task"]), int(record["local_episode_index"])): record
        for record in records
    }


def validate_artifact(path: Path, task: str, episode: int) -> dict:
    arrays = path / "arrays.npz"
    point_clouds = path / "point_clouds.zarr"
    record_path = path / "record.json"
    if not arrays.is_file() or not point_clouds.is_dir() or not record_path.is_file():
        raise RuntimeError("Incomplete artifact: " + str(path))
    record = load_json(record_path)
    if str(record.get("task")) != task or int(record.get("local_episode_index", -1)) != episode:
        raise RuntimeError("Artifact record slot mismatch: " + str(record_path))
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def newest_candidate_result(root: Path, task: str, episode: int) -> tuple[Path, dict]:
    pattern = task + "_episode_" + str(episode).zfill(3) + "_eef0_planning_*/result.json"
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError("Missing candidate replay result for " + task + ":" + str(episode))
    path = matches[-1]
    result = load_json(path)
    required = {
        "task": task,
        "episode": episode,
        "success": True,
        "gripper_mode": "delta_width_initial_sync",
        "gripper_delta_alignment": "current_minus_previous",
        "controller_profile": "pointact_eval",
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(
                "Candidate replay gate failed at " + str(path) + ": "
                + key + "=" + repr(result.get(key)) + ", expected " + repr(expected)
            )
    if abs(float(result.get("gripper_delta_threshold_m", -1.0)) - 0.003) > 1e-12:
        raise RuntimeError("Candidate was not validated with delta threshold 0.003: " + str(path))
    return path, result


def validate_target_failures(summary: dict, targets: set[tuple[str, int]]) -> None:
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise RuntimeError("Target replay summary must contain all 100 completed records")
    indexed = {(str(row["task"]), int(row["episode"])): row for row in records}
    for slot in targets:
        row = indexed.get(slot)
        if row is None:
            raise RuntimeError("Target replay summary has no record for " + repr(slot))
        if row.get("primary_state") != "completed" or bool(row.get("success")):
            raise RuntimeError("Target slot is not a completed replay failure: " + repr(slot))


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = parse_args()
    target_root = args.target_artifact_root.expanduser().resolve()
    candidate_root = args.candidate_artifact_root.expanduser().resolve()
    backup_root = args.backup_root.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    mappings = [parse_mapping(text) for text in args.replace]
    targets = [target for target, _ in mappings]
    candidates = [candidate for _, candidate in mappings]
    if len(set(targets)) != len(targets) or len(set(candidates)) != len(candidates):
        raise RuntimeError("Target and candidate slots must each be unique")

    target_manifest_path = target_root / "manifest.json"
    candidate_manifest_path = candidate_root / "manifest.json"
    target_manifest = load_json(target_manifest_path)
    candidate_manifest = load_json(candidate_manifest_path)
    target_config = target_manifest.get("config", {})
    candidate_config = candidate_manifest.get("config", {})
    for key in COMPATIBILITY_KEYS:
        if target_config.get(key) != candidate_config.get(key):
            raise RuntimeError(
                "Artifact collection settings differ for " + key + ": target="
                + repr(target_config.get(key)) + " candidate=" + repr(candidate_config.get(key))
            )
    target_records = record_by_slot(target_manifest)
    candidate_records = record_by_slot(candidate_manifest)
    validate_target_failures(load_json(args.target_replay_summary), set(targets))

    checks = []
    for target, candidate in mappings:
        if target[0] != candidate[0]:
            raise RuntimeError("A replacement must use the same RLBench task: " + repr((target, candidate)))
        if target not in target_records or candidate not in candidate_records:
            raise RuntimeError("Manifest does not contain replacement mapping " + repr((target, candidate)))
        target_path = target_root / episode_name(*target)
        candidate_path = candidate_root / episode_name(*candidate)
        target_record = validate_artifact(target_path, *target)
        candidate_record = validate_artifact(candidate_path, *candidate)
        result_path, result = newest_candidate_result(
            args.candidate_result_root.expanduser().resolve(), *candidate
        )
        checks.append(
            {
                "target": {"task": target[0], "episode": target[1]},
                "candidate": {"task": candidate[0], "episode": candidate[1]},
                "target_frames": int(target_record["frames"]),
                "candidate_frames": int(candidate_record["frames"]),
                "target_arrays_sha256": sha256(target_path / "arrays.npz"),
                "candidate_arrays_sha256": sha256(candidate_path / "arrays.npz"),
                "candidate_replay_result": str(result_path),
                "candidate_replay_success": bool(result["success"]),
            }
        )

    audit = {
        "schema": "rlbench_replay_gated_artifact_replacement_v1",
        "created_unix_s": time.time(),
        "target_artifact_root": str(target_root),
        "candidate_artifact_root": str(candidate_root),
        "target_replay_summary": str(args.target_replay_summary.resolve()),
        "candidate_result_root": str(args.candidate_result_root.resolve()),
        "gate": {
            "success": True,
            "controller_profile": "pointact_eval",
            "gripper_mode": "delta_width_initial_sync",
            "gripper_delta_threshold_m": 0.003,
            "gripper_delta_alignment": "current_minus_previous",
        },
        "replacements": checks,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        atomic_json(audit_output, audit)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    if backup_root.exists():
        raise FileExistsError("Backup root already exists: " + str(backup_root))
    backup_root.mkdir(parents=True)
    staging_root = target_root / (".replacement_staging_" + uuid.uuid4().hex)
    staging_root.mkdir()
    staged = []
    for target, candidate in mappings:
        candidate_path = candidate_root / episode_name(*candidate)
        staged_path = staging_root / episode_name(*target)
        shutil.copytree(candidate_path, staged_path, copy_function=hardlink_or_copy)
        record = load_json(staged_path / "record.json")
        record["task"] = target[0]
        record["local_episode_index"] = target[1]
        record["replacement_provenance"] = {
            "candidate_task": candidate[0],
            "candidate_local_episode_index": candidate[1],
            "candidate_artifact_root": str(candidate_root),
            "candidate_replay_result": checks[len(staged)]["candidate_replay_result"],
        }
        atomic_json(staged_path / "record.json", record)
        staged.append((target, candidate, staged_path))

    completed = []
    try:
        for target, candidate, staged_path in staged:
            target_path = target_root / episode_name(*target)
            backup_path = backup_root / target_path.name
            os.replace(target_path, backup_path)
            os.replace(staged_path, target_path)
            completed.append((target_path, backup_path, staged_path))

        replacement_by_target = dict(mappings)
        new_records = []
        for record in target_manifest["records"]:
            target = (str(record["task"]), int(record["local_episode_index"]))
            candidate = replacement_by_target.get(target)
            if candidate is None:
                new_records.append(record)
                continue
            replacement_record = dict(candidate_records[candidate])
            replacement_record["task"] = target[0]
            replacement_record["local_episode_index"] = target[1]
            replacement_record["replacement_provenance"] = {
                "candidate_task": candidate[0],
                "candidate_local_episode_index": candidate[1],
                "candidate_artifact_root": str(candidate_root),
            }
            new_records.append(replacement_record)
        target_manifest["records"] = new_records
        atomic_json(target_manifest_path, target_manifest)
        audit["dry_run"] = False
        audit["backup_root"] = str(backup_root)
        atomic_json(audit_output, audit)
    except Exception:
        for target_path, backup_path, staged_path in reversed(completed):
            if target_path.exists():
                os.replace(target_path, staged_path)
            if backup_path.exists():
                os.replace(backup_path, target_path)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

