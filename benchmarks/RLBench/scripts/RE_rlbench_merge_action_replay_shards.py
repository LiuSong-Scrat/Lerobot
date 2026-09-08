#!/usr/bin/env python3
"""Merge parallel RLBench action-replay validation shards into one report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TASK_ORDER = [
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


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fmt(value, scale=1.0, digits=4):
    if value is None:
        return "—"
    return f"{float(value) * scale:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.validation_root.expanduser().resolve()
    dataset = args.dataset_root.expanduser().resolve()

    shard_paths = sorted((root / "shards").glob("shard_*/summary.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard summaries under {root / 'shards'}")
    shards = [load_json(path) for path in shard_paths]
    records = [record for shard in shards for record in shard["records"]]
    order = {task: index for index, task in enumerate(TASK_ORDER)}
    records.sort(key=lambda row: (order.get(row["task"], 999), int(row["episode"])))
    info = load_json(dataset / "meta" / "info.json")
    expected_episodes = int(info["total_episodes"])
    if len(records) != expected_episodes:
        raise RuntimeError(
            f"Expected {expected_episodes} replay records, got {len(records)}"
        )
    primary_success = sum(bool(row["success"]) for row in records)
    failures = [row for row in records if not bool(row["success"])]
    raw_pass = sum(row.get("raw_joint_control_success") is True for row in failures)
    raw_fail = sum(row.get("raw_joint_control_success") is False for row in failures)
    failure_videos = [
        video
        for row in failures
        for video in (row.get("video"), row.get("raw_joint_video"))
        if video
    ]
    recorded_planner_times = sorted({
        int(row["planner_max_time_ms"])
        for row in records
        if row.get("planner_max_time_ms") is not None
    })
    config = {
        "mode": shards[0].get("mode"),
        "gripper_mode": shards[0].get("gripper_mode"),
        "gripper_delta_threshold_m": shards[0].get("gripper_delta_threshold_m"),
        "gripper_delta_alignment": shards[0].get("gripper_delta_alignment"),
        "water_plant_collision": shards[0].get("water_plant_collision"),
        "water_drop_collision": shards[0].get("water_drop_collision"),
        "planner_max_time_ms_recorded": recorded_planner_times,
        "legacy_results_without_planner_field_assumed_ms": 1000,
        "action_source": "LeRobot parquet action",
        "video_policy": "failure_only",
    }
    resume_config_path = root / "RESUME_CONFIG.json"
    if resume_config_path.is_file():
        config["resume_phases"] = load_json(resume_config_path).get("phases", [])
    merged = {
        "dataset_root": str(dataset),
        "dataset_total_episodes": int(info["total_episodes"]),
        "dataset_total_frames": int(info["total_frames"]),
        "dataset_total_tasks": int(info["total_tasks"]),
        "replay_config": config,
        "primary_success": primary_success,
        "primary_total": len(records),
        "primary_success_rate": primary_success / len(records),
        "primary_failures": len(failures),
        "raw_joint_failure_controls_passed": raw_pass,
        "raw_joint_failure_controls_failed": raw_fail,
        "failure_video_count": len(failure_videos),
        "failure_videos": failure_videos,
        "records": records,
    }
    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(merged, file, indent=2, ensure_ascii=False)

    columns = sorted({key for row in records for key in row.keys()})
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )

    by_task = defaultdict(list)
    for row in records:
        by_task[row["task"]].append(row)
    non_water = [row for row in records if row["task"] != "water_plants"]
    non_water_success = sum(bool(row["success"]) for row in non_water)

    lines = [
        "# RLBench 十任务 LeRobot action replay 验证报告",
        "",
        "## 结论",
        "",
        f"- 严格主 replay 成功：**{primary_success}/{len(records)}（{100 * primary_success / len(records):.1f}%）**。",
        f"- 去掉具有动态水滴状态的 `water_plants`：**{non_water_success}/{len(non_water)}（{100 * non_water_success / len(non_water):.1f}%）**。",
        f"- 对 {len(failures)} 条主 replay 失败均执行 raw-joint 对照：{raw_pass} 条成功、{raw_fail} 条失败。",
        f"- 共保存 {len(failure_videos)} 个失败视频；成功轨迹不保存视频。",
        "",
        "主 replay 失败不能自动等同于数值标签错误：raw-joint 对照、场景恢复误差、"
        "EEF tracking 误差和任务内部 success-condition 诊断需要一起判读。",
        "",
        "## 数据集与 replay 配置",
        "",
        f"- 数据集：`{dataset}`",
        f"- episodes / frames / tasks：{info['total_episodes']} / {info['total_frames']} / {info['total_tasks']}",
        f"- 每任务 {info['total_episodes'] // info['total_tasks']} 条成功 RLBench expert 轨迹；"
        "20,000 总点、500 点 REAP 虚拟夹爪。",
        "- action：前 9 维为 `expert_target`，第 10 维来自同帧 `observation.state[9]`；"
        "`transition` 对齐；主 replay 读取 LeRobot parquet action。",
        "- 场景：恢复官方 demo RNG/reset、configuration tree 和逐对象 pose/joint snapshot。",
        "- 机器人：恢复第一帧世界 EEF 位姿与夹爪状态。",
        f"- 控制：`{config['mode']}`；实际记录的 planner 时间："
        f"`{config['planner_max_time_ms_recorded']}` ms。",
        f"- 夹爪：`{config['gripper_mode']}`，阈值 "
        f"{float(config['gripper_delta_threshold_m']):.6f} m，"
        f"`{config['gripper_delta_alignment']}`。",
        "- 视频：只保存失败；每条主失败额外运行原始 joint target 对照。",
        "",
        "## 分任务结果",
        "",
        "| 任务 | 成功 | 成功率 | 失败 episode | 最大物体恢复误差 | 最大机器人恢复误差 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for task in TASK_ORDER:
        rows = by_task[task]
        # Partial validation runs (for example, replay-gating only the tasks
        # that need replacement candidates) intentionally omit other tasks.
        if not rows:
            continue
        success = sum(bool(row["success"]) for row in rows)
        failed_eps = ", ".join(str(row["episode"]) for row in rows if not row["success"]) or "—"
        max_object = max(float(row.get("object_position_error_max_m") or 0.0) for row in rows)
        max_robot = max(float(row.get("robot_position_error_m") or 0.0) for row in rows)
        lines.append(
            f"| {task} | {success}/{len(rows)} | "
            f"{(100.0 * success / len(rows)) if rows else 0.0:.1f}% | {failed_eps} | "
            f"{fmt(max_object, 1000, 3)} mm | {fmt(max_robot, 1000, 4)} mm |"
        )

    lines.extend([
        "",
        "## 失败轨迹与视频",
        "",
        "| 任务 | ep | 主 replay | raw joint | 分类 | 主视频 | raw 视频 |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for row in failures:
        primary_video = row.get("video") or "—"
        raw_video = row.get("raw_joint_video") or "—"
        lines.append(
            f"| {row['task']} | {row['episode']} | 失败 | "
            f"{'成功' if row.get('raw_joint_control_success') else '失败'} | "
            f"`{row['classification']}` | `{primary_video}` | `{raw_video}` |"
        )

    lines.extend([
        "",
        "## 判读说明",
        "",
        "- `validated`：parquet action 在精确恢复场景中直接完成任务。",
        "- raw-joint 对照成功：优先检查 EEF 标签转换或 EEF controller/planner。",
        "- raw-joint 对照也失败：不能直接判为 parquet 标签错误，还需检查仿真动力学、"
        "抓取附着、紧 proximity sensor 以及 target 并非逐位 state replay 等因素。",
        "- `water_plants` 失败时可在记录的 `task_specific_diagnostics` 中查看 pour point 是否触发和水滴数量。",
        "",
        "## 文件",
        "",
        f"- 合并 JSON：`{root / 'summary.json'}`",
        f"- 合并 CSV：`{root / 'summary.csv'}`",
        f"- shard 日志：`{root / 'launcher_logs'}`",
        f"- 全部结果和失败视频：`{root / 'shards'}`",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report={root / 'REPORT.md'}")
    print(f"success={primary_success}/{len(records)} failure_videos={len(failure_videos)}")


if __name__ == "__main__":
    main()
