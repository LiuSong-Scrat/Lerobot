#!/usr/bin/env python3
"""Join embedded LeRobot image frames into one labeled MP4 video."""

import argparse
import io
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--episodes-per-video", type=int, default=0)
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Optional comma-separated episode indices to export.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Preserve the full camera image without drawing task/episode text.",
    )
    return parser.parse_args()


def parse_episode_indices(value):
    if value is None:
        return None
    indices = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    if not indices:
        raise ValueError("--episode-indices must contain at least one episode")
    if any(index < 0 for index in indices):
        raise ValueError("--episode-indices must be non-negative")
    return tuple(dict.fromkeys(indices))


def load_tasks(dataset_root):
    rows = pq.read_table(dataset_root / "meta/tasks.parquet").to_pylist()
    return {int(row["task_index"]): row["__index_level_0__"] for row in rows}


def decode_image(value):
    image_bytes = value["bytes"]
    if image_bytes is None:
        image_bytes = (Path(value["path"])).read_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def add_label(image, task, episode, frame):
    text = f"{task} | episode {episode:03d} | frame {frame:03d}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def start_encoder(ffmpeg, output, size, fps, crf):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{size}x{size}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def close_encoder(process):
    if process is None:
        return
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main():
    args = parse_args()
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    fps = int(info["fps"])
    expected_frames = int(info["total_frames"])
    total_episodes = int(info["total_episodes"])
    tasks = load_tasks(args.dataset_root)
    parquet_files = sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {args.dataset_root / 'data'}")

    selected_episodes = parse_episode_indices(args.episode_indices)
    if selected_episodes is not None:
        episode_rows = []
        for episode_file in sorted(
            (args.dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")
        ):
            episode_rows.extend(
                pq.read_table(
                    episode_file,
                    columns=[
                        "episode_index",
                        "length",
                        "data/chunk_index",
                        "data/file_index",
                    ],
                ).to_pylist()
            )
        rows_by_episode = {int(row["episode_index"]): row for row in episode_rows}
        missing = sorted(set(selected_episodes) - set(rows_by_episode))
        if missing:
            raise ValueError(f"Unknown episode indices: {missing}")
        expected_frames = sum(
            int(rows_by_episode[index]["length"]) for index in selected_episodes
        )
        selected_data_files = {
            args.dataset_root
            / "data"
            / f"chunk-{int(rows_by_episode[index]['data/chunk_index']):03d}"
            / f"file-{int(rows_by_episode[index]['data/file_index']):03d}.parquet"
            for index in selected_episodes
        }
        missing_files = sorted(path for path in selected_data_files if not path.is_file())
        if missing_files:
            raise FileNotFoundError(f"Missing selected data files: {missing_files}")
        parquet_files = sorted(selected_data_files)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    split = int(args.episodes_per_video)
    if split < 0:
        raise ValueError("--episodes-per-video must be >= 0")
    if split and selected_episodes is not None:
        raise ValueError("Do not combine --episodes-per-video with --episode-indices")
    if split:
        args.output.mkdir(parents=True, exist_ok=True)
        process = None
    else:
        process = start_encoder(ffmpeg, args.output, args.size, fps, args.crf)
    current_group = None
    output_count = 0
    count = 0
    columns = ["observation.images.front", "episode_index", "frame_index", "task_index"]
    try:
        for parquet_file in parquet_files:
            for batch in pq.ParquetFile(parquet_file).iter_batches(batch_size=256, columns=columns):
                images = batch.column(0).to_pylist()
                episodes = batch.column(1).to_pylist()
                frames = batch.column(2).to_pylist()
                task_indices = batch.column(3).to_pylist()
                for value, episode, frame, task_index in zip(images, episodes, frames, task_indices, strict=True):
                    if selected_episodes is not None and int(episode) not in selected_episodes:
                        continue
                    if split:
                        group = int(episode) // split
                        if group != current_group:
                            close_encoder(process)
                            first = group * split
                            last = min(first + split - 1, total_episodes - 1)
                            output = args.output / f"episodes_{first:03d}_{last:03d}.mp4"
                            process = start_encoder(ffmpeg, output, args.size, fps, args.crf)
                            current_group = group
                            output_count += 1
                            print(f"[video] {output}", flush=True)
                    image = decode_image(value)
                    image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LANCZOS4)
                    if not args.no_label:
                        add_label(image, tasks[int(task_index)], int(episode), int(frame))
                    process.stdin.write(image.tobytes())
                    count += 1
                    if count % 1000 == 0:
                        print(f"[progress] {count}/{expected_frames}", flush=True)
    finally:
        close_encoder(process)
    if count != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} frames, wrote {count}")
    print(f"[done] frames={count} videos={output_count or 1} output={args.output}")


if __name__ == "__main__":
    main()
