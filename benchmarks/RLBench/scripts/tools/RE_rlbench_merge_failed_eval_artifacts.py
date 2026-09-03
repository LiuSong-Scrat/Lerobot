#!/usr/bin/env python3
"""Merge failure-only RLBench evaluation artifacts into one flat bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TASKS = ("phone_on_base", "take_frame_off_hanger", "water_plants")


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    output_root.mkdir(parents=True)
    (output_root / "source_configs").mkdir()
    manifest: dict[str, object] = {
        "source_root": str(source_root),
        "selection": "success == false",
        "tasks": {},
        "episodes": [],
    }

    for task in TASKS:
        task_parent = source_root / task
        candidates = sorted(task_parent.glob(f"{task}_*%"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one finalized directory for {task}, got {candidates}")
        task_dir = candidates[0]
        summary_path = task_dir / "summary.json"
        config_path = task_dir / "config.json"
        summary = json.loads(summary_path.read_text())
        failures = [result for result in summary["results"] if not result["success"]]

        shutil.copy2(config_path, output_root / "source_configs" / f"{task}.json")
        task_manifest = {
            "source_summary": str(summary_path),
            "evaluated": int(summary["episodes"]),
            "successes_removed": int(summary["successes"]),
            "failures_kept": len(failures),
            "failed_episode_indices": [int(result["episode_index"]) for result in failures],
        }
        manifest["tasks"][task] = task_manifest

        for result in failures:
            episode_index = int(result["episode_index"])
            episode_name = f"{task}_episode_{episode_index:03d}"
            episode_dir = output_root / episode_name
            episode_dir.mkdir()

            copy_required(
                task_dir / "videos" / f"episode_{episode_index:03d}.mp4",
                episode_dir / f"{episode_name}.mp4",
            )
            copy_required(
                task_dir / "action_visualizations" / f"episode_{episode_index:03d}",
                episode_dir / "action_visualizations",
            )
            copy_required(
                task_dir / "action_chunks" / f"episode_{episode_index:03d}",
                episode_dir / "action_chunks",
            )
            copy_required(
                task_dir / "actions" / f"episode_{episode_index:03d}_actions.npy",
                episode_dir / "actions.npy",
            )
            copy_required(
                task_dir / "model_chunks" / f"episode_{episode_index:03d}_model_chunks.npy",
                episode_dir / "model_chunks.npy",
            )
            copy_required(
                task_dir
                / "executed_action_alignment"
                / f"episode_{episode_index:03d}_executed_model_actions_relative10.npy",
                episode_dir / "executed_model_actions_relative10.npy",
            )

            compact_result = {
                key: result.get(key)
                for key in (
                    "task",
                    "episode_index",
                    "seed_episode_index",
                    "language",
                    "success",
                    "end_reason",
                    "model_calls",
                    "policy_action_steps_attempted",
                    "environment_actions",
                    "physics_frames",
                    "video_frames",
                    "mover_targets",
                    "mover_reached_targets",
                    "mover_unreached_targets",
                    "mover_retries",
                    "controller_continue_errors",
                    "error",
                )
            }
            compact_result["local_artifacts"] = {
                "video": f"{episode_name}.mp4",
                "action_visualizations": "action_visualizations/",
                "action_chunks": "action_chunks/",
                "actions": "actions.npy",
                "model_chunks": "model_chunks.npy",
                "executed_action_alignment": "executed_model_actions_relative10.npy",
            }
            (episode_dir / "result.json").write_text(
                json.dumps(compact_result, indent=2, ensure_ascii=False) + "\n"
            )
            manifest["episodes"].append(
                {
                    "name": episode_name,
                    "task": task,
                    "episode_index": episode_index,
                    "seed_episode_index": int(result["seed_episode_index"]),
                    "end_reason": result["end_reason"],
                }
            )

    manifest["total_failures_kept"] = len(manifest["episodes"])
    manifest["total_successes_removed"] = sum(
        task_info["successes_removed"] for task_info in manifest["tasks"].values()
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (output_root / "README.md").write_text(
        "# RLBench 103000 failure-only merged bundle\n\n"
        "This directory contains only failed episodes from the three-task failed-seed rerun.\n"
        "Successful episodes were not copied, and the original evaluation directories were not modified.\n\n"
        f"- Failed episodes kept: {manifest['total_failures_kept']}\n"
        f"- Successful episodes excluded: {manifest['total_successes_removed']}\n"
        "- Evaluator: checkpoint 103000, 20,000 points, 250 REAP gripper points, "
        "delta 0.0025, exec 24, planner 10 ms, max model calls 20\n"
    )


if __name__ == "__main__":
    main()
