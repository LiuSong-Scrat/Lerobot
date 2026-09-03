#!/usr/bin/env python3
"""Extract failed RLBench evaluation episodes into a reviewable package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


DEFAULT_FFMPEG = Path("/home/liusong/miniconda3/envs/rlbench/bin/ffmpeg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    return parser.parse_args()


def load_task_summary(path: Path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary.get("results"), list):
        raise ValueError(f"Expected task-level summary with results[]: {path}")
    return summary


def copy_array(source: Path, target: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)
    value = np.load(target, allow_pickle=False)
    return value


def overlay_frame_numbers(ffmpeg: Path, source: Path, target: Path, episode: int):
    text = f"EPISODE {episode:03d}   VIDEO FRAME %{{n}}"
    drawtext = (
        "drawtext="
        "font='DejaVu Sans Mono':"
        f"text='{text}':"
        "x=12:y=h-th-12:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.70:boxborderw=6"
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            drawtext,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-an",
            str(target),
        ],
        check=True,
    )


def video_frame_count(path: Path):
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count
    except Exception:
        return None


def main():
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    task_root = summary_path.parent
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)

    summary = load_task_summary(summary_path)
    failures = [row for row in summary["results"] if not bool(row.get("success"))]
    if not failures:
        raise RuntimeError("No failed episodes were found.")
    output_dir.mkdir(parents=True)

    manifest = []
    reduced_results = []
    for result in failures:
        episode = int(result["episode_index"])
        stem = f"episode_{episode:03d}"
        episode_dir = output_dir / stem
        episode_dir.mkdir()

        video_source = task_root / result.get("video", f"videos/{stem}.mp4")
        simulator_source = task_root / result.get(
            "executed_actions", f"actions/{stem}_actions.npy"
        )
        model_chunks_source = task_root / result.get(
            "model_chunks", f"model_chunks/{stem}_model_chunks.npy"
        )
        aligned_source = (
            task_root
            / "executed_action_alignment"
            / f"{stem}_executed_model_actions_relative10.npy"
        )

        original_video = episode_dir / "video_original.mp4"
        numbered_video = episode_dir / "video_with_frame_numbers.mp4"
        shutil.copy2(video_source, original_video)
        overlay_frame_numbers(args.ffmpeg, video_source, numbered_video, episode)

        simulator = copy_array(
            simulator_source, episode_dir / "executed_simulator_actions_8d.npy"
        )
        model_chunks = copy_array(model_chunks_source, episode_dir / "model_chunks_32x10.npy")
        executed_model = copy_array(
            aligned_source, episode_dir / "executed_model_actions_relative10.npy"
        )
        if len(simulator) != len(executed_model):
            raise RuntimeError(
                f"{stem}: simulator rows {len(simulator)} != executed-model rows "
                f"{len(executed_model)}"
            )

        # Requested easy-to-find action label.  It is deliberately a numeric,
        # pickle-free copy of the pose9+gripper model rows that were actually
        # selected for execution, not an unavailable online expert label.
        np.save(episode_dir / "action_labels.npy", executed_model, allow_pickle=False)
        columns = {
            "action_labels.npy": {
                "shape": list(executed_model.shape),
                "semantics": "model pose9+gripper action actually selected for each environment step",
                "columns": [
                    "translation_x",
                    "translation_y",
                    "translation_z",
                    "rotation_6d_0",
                    "rotation_6d_1",
                    "rotation_6d_2",
                    "rotation_6d_3",
                    "rotation_6d_4",
                    "rotation_6d_5",
                    "gripper_action9",
                ],
                "video_alignment": "row i is executed after video frame i and produces video frame i+1",
            },
            "executed_simulator_actions_8d.npy": {
                "shape": list(simulator.shape),
                "semantics": "final 7D pose plus gripper command sent to the RLBench simulator",
            },
            "model_chunks_32x10.npy": {
                "shape": list(model_chunks.shape),
                "semantics": "raw 32-row model output for every model call",
            },
        }
        (episode_dir / "action_columns.json").write_text(
            json.dumps(columns, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        episode_result = dict(result)
        episode_result["source_task_root"] = str(task_root)
        (episode_dir / "failure_metadata.json").write_text(
            json.dumps(episode_result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        frames = video_frame_count(numbered_video)
        manifest.append(
            {
                "episode": episode,
                "end_reason": result.get("end_reason", ""),
                "model_calls": result.get("model_calls", ""),
                "environment_actions": len(simulator),
                "video_frames": frames,
                "output_directory": stem,
            }
        )
        reduced_results.append(episode_result)
        print(
            f"[failed-eval-extract] {stem} video_frames={frames} "
            f"action_rows={len(executed_model)}",
            flush=True,
        )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "failure_summary.json").write_text(
        json.dumps(
            {
                "source_summary": str(summary_path),
                "failure_count": len(failures),
                "results": reduced_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    episode_list = ", ".join(f"{row['episode']:03d}" for row in manifest)
    readme = f"""# Failed RLBench episode review package

- Source: `{task_root}`
- Failed episodes: {episode_list}
- Failure count: {len(manifest)}

Each episode directory contains:

- `video_with_frame_numbers.mp4`: review video with zero-based video frame index.
- `video_original.mp4`: unmodified source video.
- `action_labels.npy`: `(T,10)` pose9+gripper model rows actually selected for execution.
- `executed_model_actions_relative10.npy`: the same canonical executed-model rows.
- `executed_simulator_actions_8d.npy`: `(T,8)` final command sent to RLBench.
- `model_chunks_32x10.npy`: raw `(model_calls,32,10)` policy chunks.
- `action_columns.json`: column meanings and frame alignment.
- `failure_metadata.json`: complete per-episode evaluation result.

There is no expert ground-truth action label for an online evaluation rollout.
Here `action_labels.npy` means the model action actually selected for execution.
Row `i` acts on video frame `i` and produces video frame `i+1`, so the video has
normally one more frame than the action arrays.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[failed-eval-extract-done] failures={len(manifest)} output={output_dir}")


if __name__ == "__main__":
    main()
