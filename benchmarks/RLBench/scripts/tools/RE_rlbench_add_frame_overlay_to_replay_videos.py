#!/usr/bin/env python3
"""Retrofit exact frame/action overlays onto existing RLBench replay videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RL_SCRIPTS = SCRIPT_DIR.parent
sys.path.insert(0, str(RL_SCRIPTS))

from RE_rlbench_dataset_action_replay import annotate_replay_video_frames


OVERLAY_SCHEMA = "rlbench_replay_frame_action_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, value):
    temporary = path.with_name(path.name + ".frame_overlay_tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    result_paths = sorted((root / "replays").glob("*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No replay results under {root / 'replays'}")

    records = []
    for index, result_path in enumerate(result_paths, start=1):
        result = read_json(result_path)
        prior_overlay = result.get("video_frame_overlay") or {}
        if prior_overlay.get("enabled") and not args.force:
            records.append(
                {
                    "result": str(result_path),
                    "state": "already_annotated",
                    "schema": prior_overlay.get("schema"),
                }
            )
            continue
        video_path = result_path.parent / "replay.mp4"
        labels_path = result_path.parent / "action_labels.npy"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if not labels_path.is_file():
            raise FileNotFoundError(labels_path)

        reader = imageio.get_reader(video_path)
        try:
            metadata = reader.get_meta_data()
            fps = float(metadata.get("fps", 20.0))
            raw_frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
        finally:
            reader.close()
        actions = np.load(labels_path)
        annotated = annotate_replay_video_frames(
            frames=raw_frames,
            execution_trace=result.get("execution_trace", []),
            actions=actions,
            task_name=result["task"],
            episode_index=int(result["episode"]),
            mode=result["mode"],
            success=bool(result["success"]),
        )
        temporary_video = video_path.with_name("replay.frame_overlay_tmp.mp4")
        imageio.mimsave(
            temporary_video,
            annotated,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        check_reader = imageio.get_reader(temporary_video)
        try:
            output_frames = int(check_reader.count_frames())
            if output_frames:
                check_reader.get_data(0)
        finally:
            check_reader.close()
        if output_frames != len(raw_frames):
            temporary_video.unlink(missing_ok=True)
            raise RuntimeError(
                f"Frame count changed for {video_path}: "
                f"{len(raw_frames)} -> {output_frames}"
            )
        os.replace(temporary_video, video_path)
        result["video_frame_overlay"] = {
            "enabled": True,
            "schema": OVERLAY_SCHEMA,
            "mapping_source": "execution_trace",
            "retrofitted": True,
            "video_frame_count": output_frames,
            "fields": [
                "task",
                "episode",
                "mode",
                "video_frame",
                "result",
                "dataset_frame",
                "action_index",
                "phase",
                "mover_attempt",
                "label_xyz",
                "label_gripper",
            ],
        }
        write_json_atomic(result_path, result)
        records.append(
            {
                "result": str(result_path),
                "video": str(video_path),
                "state": "annotated",
                "schema": OVERLAY_SCHEMA,
                "video_frame_count": output_frames,
            }
        )
        print(
            f"[overlay] {index}/{len(result_paths)} task={result['task']} "
            f"episode={int(result['episode']):03d} frames={output_frames}",
            flush=True,
        )

    summary = {
        "run_root": str(root),
        "requested": len(result_paths),
        "annotated": sum(row["state"] == "annotated" for row in records),
        "already_annotated": sum(
            row["state"] == "already_annotated" for row in records
        ),
        "schema": OVERLAY_SCHEMA,
        "records": records,
    }
    write_json_atomic(root / "frame_overlay_retrofit.json", summary)
    print(
        f"complete={len(records)}/{len(result_paths)} "
        f"annotated={summary['annotated']} already={summary['already_annotated']}"
    )


if __name__ == "__main__":
    main()
