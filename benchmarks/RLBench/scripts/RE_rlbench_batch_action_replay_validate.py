#!/usr/bin/env python3
"""Validate every collected RLBench action trajectory by exact-scene replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from RE_rlbench_gripper_control import (
    DELTA_WIDTH_INITIAL_SYNC,
    GRIPPER_CONTROL_MODES,
)


DEFAULT_TASKS = [
    "close_box",
    "close_fridge",
    "close_laptop_lid",
    "phone_on_base",
    "stack_wine",
    "sweep_to_dustpan",
    "take_frame_off_hanger",
    "take_umbrella_out_of_umbrella_stand",
    "toilet_seat_down",
    "water_plants",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay all dataset actions, retain failure videos, and summarize validity."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument(
        "--mode", choices=["eef0_planning", "eef0"], default="eef0_planning"
    )
    parser.add_argument(
        "--controller-profile",
        choices=["single_step", "pointact_eval"],
        default="single_step",
        help="Controller execution profile passed to each episode replay.",
    )
    parser.add_argument("--mover-max-tries", type=int, default=10)
    parser.add_argument("--mover-position-tolerance", type=float, default=0.05)
    parser.add_argument("--mover-gripper-position-tolerance", type=float, default=0.02)
    parser.add_argument("--waypoint-position-tolerance", type=float, default=0.002)
    parser.add_argument("--waypoint-rotation-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--clip-within-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gripper-after-reach",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pointact-pyrep-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--gripper-open-width", type=float, default=0.04)
    parser.add_argument(
        "--gripper-mode",
        choices=GRIPPER_CONTROL_MODES,
        default=DELTA_WIDTH_INITIAL_SYNC,
    )
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.002)
    parser.add_argument(
        "--gripper-delta-alignment",
        choices=["current_minus_previous", "next_minus_current"],
        default="current_minus_previous",
    )
    parser.add_argument(
        "--raw-joint-control-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay original joint targets after a converted EEF-label failure.",
    )
    parser.add_argument(
        "--water-plant-collision",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Match the water_plants plant-collision mode used during collection.",
    )
    parser.add_argument(
        "--water-drop-collision",
        choices=["original", "disabled"],
        default="original",
        help="Match the water_plants drop-collision mode used during collection.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def result_pattern(root: Path, task: str, episode: int, mode: str):
    return root.glob(
        f"{task}_episode_{episode:03d}_{mode}_*/result.json"
    )


def newest_result(root: Path, task: str, episode: int, mode: str):
    matches = sorted(result_pattern(root, task, episode, mode))
    return matches[-1] if matches else None


def read_result(path: Path | None):
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def replay_command(args, task, episode, mode, action_source, output_root):
    replay_script = Path(__file__).resolve().with_name(
        "RE_rlbench_dataset_action_replay.py"
    )
    artifact_root = args.artifact_root
    if artifact_root is None:
        artifact_root = Path(str(args.dataset_root.resolve()) + "_artifacts")
    command = [
        str(args.python),
        str(replay_script),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--artifact-root", str(artifact_root.resolve()),
        "--output-root", str(output_root.resolve()),
        "--task", task,
        "--episode", str(episode),
        "--mode", mode,
        "--action-source", action_source,
        "--variation", str(args.variation),
        "--seed", str(args.seed),
        "--image-size", str(args.image_size),
        "--video-fps", str(args.video_fps),
        "--video-policy", "failures",
        "--controller-profile", str(args.controller_profile),
        "--mover-max-tries", str(args.mover_max_tries),
        "--mover-position-tolerance", str(args.mover_position_tolerance),
        "--mover-gripper-position-tolerance", str(args.mover_gripper_position_tolerance),
        "--waypoint-position-tolerance", str(args.waypoint_position_tolerance),
        "--waypoint-rotation-tolerance", str(args.waypoint_rotation_tolerance),
        "--gripper-open-width", str(args.gripper_open_width),
        "--gripper-mode", str(args.gripper_mode),
        "--gripper-delta-threshold", str(args.gripper_delta_threshold),
        "--gripper-delta-alignment", str(args.gripper_delta_alignment),
        "--water-plant-collision", str(args.water_plant_collision),
        "--water-drop-collision", str(args.water_drop_collision),
        "--disable-point-cloud",
    ]
    command.append(
        "--clip-within-workspace"
        if args.clip_within_workspace
        else "--no-clip-within-workspace"
    )
    command.append(
        "--gripper-after-reach"
        if args.gripper_after_reach
        else "--no-gripper-after-reach"
    )
    command.append(
        "--pointact-pyrep-compat"
        if args.pointact_pyrep_compat
        else "--no-pointact-pyrep-compat"
    )
    return command


def run_one(args, task, episode, mode, action_source, output_root, log_root):
    existing = newest_result(output_root, task, episode, mode)
    if existing is not None:
        return read_result(existing), existing, "existing"

    log_path = log_root / f"{task}_episode_{episode:03d}_{mode}.log"
    command = replay_command(
        args, task, episode, mode, action_source, output_root
    )
    print(
        f"[replay-start] task={task} episode={episode} mode={mode}",
        flush=True,
    )
    env = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command=" + " ".join(command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    result_path = newest_result(output_root, task, episode, mode)
    result = read_result(result_path)
    state = "completed" if result is not None else f"process_error_{completed.returncode}"
    print(
        f"[replay-done] task={task} episode={episode} mode={mode} "
        f"success={None if result is None else result.get('success')} state={state}",
        flush=True,
    )
    # Give CoppeliaSim/Qt time to release the display before the next launch.
    time.sleep(1.0)
    return result, result_path, state


def classify(primary, control):
    if primary is None:
        return "primary_process_error"
    if bool(primary.get("success")):
        return "validated"
    if control is None:
        return "eef_replay_failed_unresolved"
    if bool(control.get("success")):
        return "eef_label_or_eef_controller_failure_raw_joint_passed"
    return "scene_restore_or_execution_failure_raw_joint_failed"


def nested_value(data, *keys):
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def write_reports(args, records):
    summary_json = args.output_root / "summary.json"
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "mode": args.mode,
                "controller_profile": args.controller_profile,
                "mover_max_tries": args.mover_max_tries,
                "clip_within_workspace": args.clip_within_workspace,
                "gripper_after_reach": args.gripper_after_reach,
                "pointact_pyrep_compat": args.pointact_pyrep_compat,
                "gripper_mode": args.gripper_mode,
                "gripper_delta_threshold_m": args.gripper_delta_threshold,
                "gripper_delta_alignment": args.gripper_delta_alignment,
                "water_plant_collision": args.water_plant_collision,
                "water_drop_collision": args.water_drop_collision,
                "planner_max_time_ms": int(
                    os.environ.get("RLBENCH_PLANNER_MAX_TIME_MS", "1000")
                ),
                "episodes_per_task": args.episodes_per_task,
                "tasks": args.tasks,
                "records": records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    columns = [
        "task", "episode", "global_episode", "success", "classification",
        "raw_joint_control_success", "scene_restore_method", "planner_max_time_ms",
        "first_rgb_mae",
        "object_position_error_max_m", "robot_position_error_m",
        "eef_tracking_position_error_max_m", "errors", "video", "result_json",
        "raw_joint_video", "raw_joint_result_json",
    ]
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in records)

    by_task = defaultdict(list)
    for row in records:
        by_task[row["task"]].append(row)
    total_success = sum(bool(row["success"]) for row in records)
    lines = [
        "# RLBench 数据集 action replay 验证报告",
        "",
        f"- 数据集：`{args.dataset_root.resolve()}`",
        f"- 主重放模式：`{args.mode}`，action 来源：LeRobot parquet",
        f"- 夹爪：`{args.gripper_mode}`，阈值 `{args.gripper_delta_threshold:.6f} m`，"
        f"对齐 `{args.gripper_delta_alignment}`",
        f"- 浇水物理：plant collision=`{args.water_plant_collision}`，"
        f"drop collision=`{args.water_drop_collision}`",
        f"- Planner 最大时间：`{os.environ.get('RLBENCH_PLANNER_MAX_TIME_MS', '1000')} ms`",
        f"- 总体：**{total_success}/{len(records)}**",
        "- 视频策略：只保存失败轨迹；主重放失败后运行 raw-joint 对照。",
        "",
        "## 分任务结果",
        "",
        "| 任务 | 成功 | 总数 | 成功率 |",
        "|---|---:|---:|---:|",
    ]
    for task in args.tasks:
        rows = by_task.get(task, [])
        passed = sum(bool(row["success"]) for row in rows)
        rate = 100.0 * passed / len(rows) if rows else 0.0
        lines.append(f"| {task} | {passed} | {len(rows)} | {rate:.1f}% |")

    failures = [row for row in records if not bool(row["success"])]
    lines.extend(["", "## 失败轨迹", ""])
    if not failures:
        lines.append("无。全部数据集动作标签均完成任务。")
    else:
        lines.extend([
            "| 任务 | episode | 分类 | raw joint | 视频 |",
            "|---|---:|---|---|---|",
        ])
        for row in failures:
            video = row.get("video") or "未生成"
            control = row.get("raw_joint_control_success")
            lines.append(
                f"| {row['task']} | {row['episode']} | {row['classification']} | "
                f"{control} | `{video}` |"
            )

    lines.extend([
        "",
        "## 判读口径",
        "",
        "- `validated`：恢复数据集第一帧后，parquet EEF action 顺序执行并成功。",
        "- `eef_label_or_eef_controller_failure_raw_joint_passed`：原始 joint target 成功，问题位于 EEF 标签转换或 EEF 控制执行链路。",
        "- `scene_restore_or_execution_failure_raw_joint_failed`：原始 joint 对照也失败，应先查场景/机器人恢复和仿真执行差异。",
        "- `primary_process_error`：程序或仿真器异常，没有形成有效测试结果。",
        "",
    ])
    (args.output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_json


def main():
    args = parse_args()
    if (
        args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
        and args.gripper_delta_alignment != "current_minus_previous"
    ):
        raise ValueError(
            "delta_width_initial_sync requires current_minus_previous alignment."
        )
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not (args.dataset_root / "data").is_dir():
        raise FileNotFoundError(f"LeRobot data directory is missing: {args.dataset_root / 'data'}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    primary_root = args.output_root / "primary_eef_replay"
    control_root = args.output_root / "raw_joint_failure_controls"
    log_root = args.output_root / "logs"
    primary_root.mkdir(exist_ok=True)
    control_root.mkdir(exist_ok=True)
    log_root.mkdir(exist_ok=True)

    records = []
    for task in args.tasks:
        for episode in range(args.episodes_per_task):
            primary, primary_path, primary_state = run_one(
                args, task, episode, args.mode, "parquet", primary_root, log_root
            )
            control = None
            control_path = None
            if (
                args.raw_joint_control_on_failure
                and primary is not None
                and not bool(primary.get("success"))
            ):
                control, control_path, _ = run_one(
                    args, task, episode, "raw_joint", "parquet",
                    control_root, log_root,
                )
            record = {
                "task": task,
                "episode": episode,
                "global_episode": None if primary is None else primary.get("global_episode"),
                "success": False if primary is None else bool(primary.get("success")),
                "classification": classify(primary, control),
                "primary_state": primary_state,
                "raw_joint_control_success": (
                    None if control is None else bool(control.get("success"))
                ),
                "scene_restore_method": None if primary is None else primary.get("scene_restore_method"),
                "planner_max_time_ms": (
                    None if primary is None else primary.get("planner_max_time_ms")
                ),
                "first_rgb_mae": None if primary is None else primary.get("restored_or_reset_vs_recorded_first_rgb_mae"),
                "object_position_error_max_m": nested_value(
                    primary, "initial_object_state_validation", "position_error_max_m"
                ),
                "robot_position_error_m": nested_value(
                    primary, "initial_robot_state_validation", "position_error_m"
                ),
                "eef_tracking_position_error_max_m": (
                    None if primary is None else primary.get("eef_tracking_position_error_max_m")
                ),
                "task_specific_diagnostics": (
                    None if primary is None else primary.get("task_specific_diagnostics")
                ),
                "errors": None if primary is None else primary.get("errors"),
                "video": None if primary is None else primary.get("video"),
                "result_json": None if primary_path is None else str(primary_path),
                "raw_joint_video": None if control is None else control.get("video"),
                "raw_joint_result_json": None if control_path is None else str(control_path),
            }
            records.append(record)
            write_reports(args, records)

    summary_path = write_reports(args, records)
    success_count = sum(bool(row["success"]) for row in records)
    print(
        f"[validation-complete] success={success_count}/{len(records)} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
