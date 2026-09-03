#!/usr/bin/env python3
"""Remove successful-episode artifacts from a completed RLBench eval run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _inside(root: Path, candidate: Path) -> bool:
    root = root.resolve()
    candidate = candidate.resolve()
    return os.path.commonpath((str(root), str(candidate))) == str(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    removed_videos = 0
    removed_action_vis_dirs = 0
    for result in summary.get("results", []):
        if not result.get("success", False):
            continue

        video_rel = result.get("video")
        if video_rel:
            video_path = run_dir / video_rel
            if not _inside(run_dir, video_path):
                raise ValueError(f"Refusing path outside run directory: {video_path}")
            if video_path.is_file():
                video_path.unlink()
                removed_videos += 1
            result["video"] = None

        episode_index = int(result["episode_index"])
        action_vis_dir = run_dir / "action_visualizations" / f"episode_{episode_index:03d}"
        if not _inside(run_dir, action_vis_dir):
            raise ValueError(f"Refusing path outside run directory: {action_vis_dir}")
        if action_vis_dir.is_dir():
            shutil.rmtree(action_vis_dir)
            removed_action_vis_dirs += 1

    temporary_path = summary_path.with_suffix(".json.prune_tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, summary_path)
    print(
        f"[prune-success-artifacts] run_dir={run_dir} "
        f"videos={removed_videos} action_vis_dirs={removed_action_vis_dirs}"
    )


if __name__ == "__main__":
    main()
