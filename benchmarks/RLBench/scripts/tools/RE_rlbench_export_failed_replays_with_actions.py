#!/usr/bin/env python3
"""Bundle failed RLBench action replays with their exact dataset labels."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ACTION_COLUMNS = [
    "x_m",
    "y_m",
    "z_m",
    "rotation_col0_x",
    "rotation_col0_y",
    "rotation_col0_z",
    "rotation_col1_x",
    "rotation_col1_y",
    "rotation_col1_z",
    "gripper_width_m",
]
RAW_JOINT_COLUMNS = [
    "panda_joint1_rad",
    "panda_joint2_rad",
    "panda_joint3_rad",
    "panda_joint4_rad",
    "panda_joint5_rad",
    "panda_joint6_rad",
    "panda_joint7_rad",
    "gripper_open_bit",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def relocated_path(path_value, validation_root: Path):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file():
        return path
    try:
        shard_index = path.parts.index("shards")
    except ValueError:
        return None
    candidate = validation_root.joinpath(*path.parts[shard_index:])
    return candidate if candidate.is_file() else None


def load_episode_actions(parquet_dataset, global_episode: int):
    import pyarrow.dataset as pyarrow_dataset

    table = parquet_dataset.to_table(
        filter=pyarrow_dataset.field("episode_index") == int(global_episode),
        columns=["frame_index", "action", "observation.images.front"],
    )
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    encoded_images = table["observation.images.front"].to_pylist()
    order = np.argsort(frame_indices)
    return (
        frame_indices[order],
        actions[order],
        [encoded_images[index] for index in order],
    )


def write_array_csv(path: Path, frame_indices, array, columns):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action_index", "dataset_frame_index", *columns])
        for action_index, (frame_index, values) in enumerate(zip(frame_indices, array)):
            writer.writerow(
                [action_index, int(frame_index), *[f"{float(value):.9g}" for value in values]]
            )


def annotate_video(
    source: Path,
    destination: Path,
    actions: np.ndarray,
    font_path: Path,
):
    reader = imageio.get_reader(source)
    source_meta = reader.get_meta_data()
    fps = float(source_meta.get("fps", 20.0))
    font = ImageFont.truetype(str(font_path), 13)
    writer = imageio.get_writer(
        destination,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    frame_count = 0
    try:
        for video_frame, frame in enumerate(reader):
            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            if video_frame == 0:
                lines = [
                    "video_frame=0000 | RESET",
                    "action_index=none",
                    f"dataset_actions={len(actions)}",
                ]
            else:
                action_index = video_frame - 1
                if action_index < len(actions):
                    action = actions[action_index]
                    lines = [
                        f"video_frame={video_frame:04d}",
                        f"action_index={action_index:04d}/{len(actions) - 1:04d}",
                        "xyz=({:+.3f},{:+.3f},{:+.3f}) g={:.3f}".format(
                            float(action[0]),
                            float(action[1]),
                            float(action[2]),
                            float(action[9]),
                        ),
                    ]
                else:
                    lines = [
                        f"video_frame={video_frame:04d}",
                        f"action_index={action_index:04d} OUT_OF_RANGE",
                    ]
            line_height = 16
            box_height = 7 + line_height * len(lines)
            draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 170))
            for line_index, line in enumerate(lines):
                draw.text(
                    (5, 4 + line_index * line_height),
                    line,
                    font=font,
                    fill=(255, 255, 255, 255),
                )
            writer.append_data(np.asarray(image, dtype=np.uint8))
            frame_count += 1
    finally:
        reader.close()
        writer.close()
    return {"fps": fps, "video_frame_count": frame_count}


def decode_dataset_image(encoded):
    image_bytes = encoded.get("bytes", b"") if isinstance(encoded, dict) else b""
    if not image_bytes:
        raise RuntimeError("Dataset front image contains no encoded bytes.")
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def write_full_dataset_video(
    destination: Path,
    frame_indices: np.ndarray,
    encoded_images,
    actions: np.ndarray,
    fps: float,
    font_path: Path,
):
    """Write every stored dataset observation with its same-row action label."""
    if len(encoded_images) != len(actions):
        raise ValueError(
            f"Dataset image/action lengths differ: {len(encoded_images)} != {len(actions)}"
        )
    font = ImageFont.truetype(str(font_path), 12)
    writer = imageio.get_writer(
        destination,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for action_index, (dataset_frame, encoded, action) in enumerate(
            zip(frame_indices, encoded_images, actions)
        ):
            image = decode_dataset_image(encoded)
            draw = ImageDraw.Draw(image, "RGBA")
            lines = [
                f"dataset_frame={int(dataset_frame):04d}/{len(actions) - 1:04d}",
                f"action_index={action_index:04d} (command from this frame)",
                "xyz=({:+.3f},{:+.3f},{:+.3f}) g={:.3f}".format(
                    float(action[0]),
                    float(action[1]),
                    float(action[2]),
                    float(action[9]),
                ),
            ]
            line_height = 15
            box_height = 7 + line_height * len(lines)
            draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 170))
            for line_index, line in enumerate(lines):
                draw.text(
                    (4, 4 + line_index * line_height),
                    line,
                    font=font,
                    fill=(255, 255, 255, 255),
                )
            writer.append_data(np.asarray(image, dtype=np.uint8))
    finally:
        writer.close()
    return {
        "fps": float(fps),
        "video_frame_count": len(encoded_images),
        "complete_dataset_episode": True,
    }


def main():
    args = parse_args()
    validation_root = args.validation_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = validation_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not args.font.is_file():
        raise FileNotFoundError(args.font)

    summary = read_json(summary_path)
    dataset_root = Path(summary["dataset_root"]).expanduser().resolve()
    failed_records = [record for record in summary["records"] if not bool(record["success"])]
    dataset_info = read_json(dataset_root / "meta" / "info.json")
    dataset_fps = float(dataset_info.get("fps", 20.0))

    import pyarrow.dataset as pyarrow_dataset

    parquet_dataset = pyarrow_dataset.dataset(str(dataset_root / "data"), format="parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for record in failed_records:
        task = str(record["task"])
        local_episode = int(record["episode"])
        global_episode = int(record["global_episode"])
        trajectory_dir = output_dir / f"{task}_episode_{local_episode:03d}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        frame_indices, actions, encoded_images = load_episode_actions(
            parquet_dataset, global_episode
        )
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise RuntimeError(
                f"Expected action labels (T,10) for global episode {global_episode}, got {actions.shape}"
            )
        np.save(trajectory_dir / "action_labels_pose9_gripper.npy", actions)
        write_array_csv(
            trajectory_dir / "action_labels_pose9_gripper.csv",
            frame_indices,
            actions,
            ACTION_COLUMNS,
        )
        full_dataset_video_meta = write_full_dataset_video(
            trajectory_dir / "dataset_trajectory_full_with_frame_and_action_index.mp4",
            frame_indices,
            encoded_images,
            actions,
            dataset_fps,
            args.font,
        )

        raw_path = dataset_root / "raw_expert_actions" / f"episode_{global_episode:06d}.npy"
        raw_actions = None
        if raw_path.is_file():
            raw_actions = np.asarray(np.load(raw_path), dtype=np.float32)
            if raw_actions.shape != (len(actions), 8):
                raise RuntimeError(
                    f"Expected raw actions {(len(actions), 8)} for {raw_path}, got {raw_actions.shape}"
                )
            np.save(trajectory_dir / "raw_joint_actions.npy", raw_actions)
            write_array_csv(
                trajectory_dir / "raw_joint_actions.csv",
                frame_indices,
                raw_actions,
                RAW_JOINT_COLUMNS,
            )

        main_video = relocated_path(record.get("video"), validation_root)
        if main_video is None:
            raise FileNotFoundError(
                f"Main failure video missing for {task} local episode {local_episode}"
            )
        original_video = trajectory_dir / "replay_original.mp4"
        annotated_video = trajectory_dir / "replay_with_frame_and_action_index.mp4"
        shutil.copy2(main_video, original_video)
        video_meta = annotate_video(main_video, annotated_video, actions, args.font)

        raw_video = relocated_path(record.get("raw_joint_video"), validation_root)
        raw_video_meta = None
        if raw_video is not None:
            raw_original = trajectory_dir / "raw_joint_replay_original.mp4"
            raw_annotated = trajectory_dir / "raw_joint_replay_with_frame_and_action_index.mp4"
            shutil.copy2(raw_video, raw_original)
            raw_video_meta = annotate_video(raw_video, raw_annotated, actions, args.font)

        result_path = relocated_path(record.get("result_json"), validation_root)
        if result_path is not None:
            shutil.copy2(result_path, trajectory_dir / "replay_result.json")
        raw_result_path = relocated_path(record.get("raw_joint_result_json"), validation_root)
        if raw_result_path is not None:
            shutil.copy2(raw_result_path, trajectory_dir / "raw_joint_replay_result.json")

        trajectory_meta = {
            "task": task,
            "local_episode": local_episode,
            "global_episode": global_episode,
            "classification": record["classification"],
            "action_alignment": "transition",
            "action_label_semantics": "EEF0-relative target pose9 plus gripper width in metres",
            "video_mapping": {
                "video_frame_0": "restored initial observation; no action has executed",
                "video_frame_n_for_n_ge_1": "result after executing action_labels[n-1]",
            },
            "action_shape": list(actions.shape),
            "raw_joint_action_shape": None if raw_actions is None else list(raw_actions.shape),
            "main_video": video_meta,
            "full_dataset_video": full_dataset_video_meta,
            "raw_joint_video": raw_video_meta,
            "raw_joint_control_success": record.get("raw_joint_control_success"),
            "source_main_video": str(main_video),
        }
        with (trajectory_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(trajectory_meta, file, indent=2, ensure_ascii=False)

        manifest_rows.append(
            {
                "task": task,
                "local_episode": local_episode,
                "global_episode": global_episode,
                "classification": record["classification"],
                "action_rows": len(actions),
                "main_video_frames": video_meta["video_frame_count"],
                "full_dataset_video_frames": full_dataset_video_meta["video_frame_count"],
                "raw_joint_control_success": record.get("raw_joint_control_success"),
                "trajectory_dir": str(trajectory_dir),
            }
        )

    manifest_columns = list(manifest_rows[0]) if manifest_rows else []
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme_lines = [
        "# RLBench action replay 失败轨迹、视频与动作标签",
        "",
        f"- 来源：`{validation_root}`",
        f"- 数据集：`{dataset_root}`",
        f"- 失败轨迹：**{len(manifest_rows)}** 条",
        "- 每条轨迹保留原视频，并生成带视频帧号和 action 索引的标注视频。",
        "- `dataset_trajectory_full_with_frame_and_action_index.mp4` 覆盖该 episode 在 LeRobot 中存储的全部帧。",
        "- replay 视频只覆盖失败发生前实际执行到的部分，因此可能短于完整数据集轨迹。",
        "- 视频第 0 帧是恢复后的初始 observation，没有执行动作。",
        "- 视频第 n 帧（n>=1）是执行 `action_labels[n-1]` 后的 observation。",
        "",
        "每个轨迹目录包含：",
        "",
        "```text",
        "replay_original.mp4",
        "replay_with_frame_and_action_index.mp4",
        "dataset_trajectory_full_with_frame_and_action_index.mp4",
        "action_labels_pose9_gripper.npy",
        "action_labels_pose9_gripper.csv",
        "raw_joint_actions.npy",
        "raw_joint_actions.csv",
        "replay_result.json",
        "metadata.json",
        "```",
        "",
        "若 raw-joint 对照本身也失败，还包含 raw-joint 原始和标注视频。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"failed_trajectories={len(manifest_rows)}")


if __name__ == "__main__":
    main()
