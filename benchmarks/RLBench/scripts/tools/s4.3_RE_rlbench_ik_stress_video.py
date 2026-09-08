#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a visual, reproducible Jacobian-IK stress-test video.

This is a diagnostic artifact, not a policy evaluation.  It keeps the real
RLBench scene and PyRep IK implementation, sends a reachable target, then a
large orientation change that triggers IKError, and finally a reachable
recovery target.  The video labels the injected target sequence explicitly.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


from _rlbench_tool_paths import LEROBOT_ROOT as REPO_ROOT
RL_BENCH_ROOT = REPO_ROOT / "benchmarks" / "RLBench"
sys.path.insert(0, str(RL_BENCH_ROOT))

from rlbench import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks.close_laptop_lid import CloseLaptopLid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "benchmarks/RLBench/outputs/eval/ik_stress_test_close_laptop_lid_20260812",
    )
    parser.add_argument("--display", default=":99")
    parser.add_argument(
        "--coppeliasim-root",
        default=str(REPO_ROOT / "benchmarks/CoppeliaSim"),
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--hold-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rgb_uint8(rgb):
    image = np.asarray(rgb)
    if image.dtype.kind == "f" and image.size and float(np.max(image)) <= 1.0 + 1e-6:
        image = image * 255.0
    return np.clip(np.nan_to_num(image), 0, 255).astype(np.uint8)


def pose7_to_target(pose7):
    pose7 = np.asarray(pose7, dtype=np.float64)
    quat = pose7[3:7] / max(np.linalg.norm(pose7[3:7]), 1e-12)
    return np.r_[pose7[:3], quat]


def pose_error(actual_pose7, target_pose7):
    actual = pose7_to_target(actual_pose7)
    target = pose7_to_target(target_pose7)
    position_error = float(np.linalg.norm(actual[:3] - target[:3]))
    relative = Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(target[3:])
    rotation_error = float(np.linalg.norm(relative.as_rotvec()))
    return position_error, rotation_error


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(frame, record, width, height, trace):
    frame = Image.fromarray(rgb_uint8(frame)).resize((width, height), Image.Resampling.BILINEAR)
    panel_width = 430
    canvas = Image.new("RGB", (width + panel_width, height), (20, 22, 27))
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = font(22)
    body_font = font(17)
    small_font = font(14)
    x = width + 18
    y = 16
    draw.text((x, y), "EEF Jacobian IK stress test", fill=(255, 255, 255), font=title_font)
    y += 38
    draw.text((x, y), "RLBench / PyRep / Panda", fill=(180, 190, 205), font=small_font)
    y += 30
    status = record["status"]
    status_color = {
        "REACHABLE": (60, 220, 100),
        "IK_ERROR": (255, 70, 70),
        "RECOVERED": (50, 190, 255),
    }.get(status, (255, 255, 255))
    draw.rectangle((x, y, x + panel_width - 36, y + 38), fill=(45, 45, 50), outline=status_color, width=2)
    draw.text((x + 10, y + 8), status, fill=status_color, font=body_font)
    y += 55
    lines = [
        "phase: " + record["phase"],
        "control frame: " + str(record["frame"]),
        "target position: " + str([round(v, 3) for v in record["target_pose7"][:3]]),
        "actual position: " + str([round(v, 3) for v in record["actual_pose7"][:3]]),
        "position error: " + ("n/a" if record["position_error_m"] is None else "%.3f m" % record["position_error_m"]),
        "rotation error: " + ("n/a" if record["rotation_error_rad"] is None else "%.3f rad" % record["rotation_error_rad"]),
        "ik exception: " + (record["error"] or "none"),
    ]
    for line in lines:
        draw.text((x, y), line, fill=(235, 238, 242), font=small_font)
        y += 25
    y += 8
    draw.text((x, y), "Interpretation", fill=(255, 220, 100), font=body_font)
    y += 29
    interpretation = [
        "same scene and real Jacobian IK",
        "target orientation changes by 90 deg",
        "IKError = local convergence failure",
        "not a fabricated RGB effect",
    ]
    for line in interpretation:
        draw.text((x, y), line, fill=(190, 198, 210), font=small_font)
        y += 23

    plot_top = height - 170
    plot_left = x
    plot_right = x + panel_width - 36
    plot_bottom = height - 22
    draw.text((plot_left, plot_top - 24), "EEF position: target(red) / actual(blue)", fill=(220, 225, 230), font=small_font)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 95, 105), width=1)
    if trace:
        values = np.asarray([[r["target_pose7"][0], r["target_pose7"][2]] for r in trace + [record]], dtype=float)
        actual_values = np.asarray([[r["actual_pose7"][0], r["actual_pose7"][2]] for r in trace + [record]], dtype=float)
        all_values = np.vstack([values, actual_values])
        lo = all_values.min(axis=0) - 0.01
        hi = all_values.max(axis=0) + 0.01
        span = np.maximum(hi - lo, 1e-6)

        def project(value):
            px = plot_left + 8 + (value[0] - lo[0]) / span[0] * (plot_right - plot_left - 16)
            py = plot_bottom - 8 - (value[1] - lo[1]) / span[1] * (plot_bottom - plot_top - 16)
            return int(px), int(py)

        target_points = [project(v) for v in values]
        actual_points = [project(v) for v in actual_values]
        if len(target_points) > 1:
            draw.line(target_points, fill=(255, 70, 70), width=3)
            draw.line(actual_points, fill=(60, 150, 255), width=3)
        draw.ellipse((*project(values[-1]), project(values[-1])[0] + 6, project(values[-1])[1] + 6), fill=(255, 70, 70))
        draw.ellipse((*project(actual_values[-1]), project(actual_values[-1])[0] + 6, project(actual_values[-1])[1] + 6), fill=(60, 150, 255))
    return np.asarray(canvas, dtype=np.uint8)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Output already exists: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    import os

    os.environ["DISPLAY"] = args.display
    os.environ["COPPELIASIM_ROOT"] = args.coppeliasim_root
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")

    env = Environment(
        MoveArmThenGripper(EndEffectorPoseViaIK(), Discrete()),
        headless=True,
        static_positions=False,
    )
    records = []
    frames = []
    trace = []
    frame_index = 0
    try:
        env.launch()
        task_env = env.get_task(CloseLaptopLid)
        task_env.set_variation(0)
        _, observation = task_env.reset()
        initial_pose = pose7_to_target(observation.gripper_pose)
        base_rotation = Rotation.from_quat(initial_pose[3:])
        targets = [
            ("baseline", "REACHABLE", initial_pose.copy()),
            (
                "large_orientation_change",
                "IK_ERROR",
                np.r_[
                    initial_pose[:3],
                    (Rotation.from_euler("y", 90, degrees=True) * base_rotation).as_quat(),
                ],
            ),
            (
                "recovery_orientation",
                "RECOVERED",
                np.r_[
                    initial_pose[:3],
                    (Rotation.from_euler("x", 90, degrees=True) * base_rotation).as_quat(),
                ],
            ),
            ("return_to_baseline", "REACHABLE", initial_pose.copy()),
        ]

        for phase, expected_status, target in targets:
            target = pose7_to_target(target)
            for _ in range(max(1, args.hold_frames)):
                actual = pose7_to_target(task_env.get_observation().gripper_pose)
                position_error, rotation_error = pose_error(actual, target)
                status = "REACHABLE"
                error = None
                try:
                    if _ == 0:
                        next_observation, _, _ = task_env.step(np.r_[target, 1.0])
                        observation = next_observation
                        actual = pose7_to_target(observation.gripper_pose)
                        position_error, rotation_error = pose_error(actual, target)
                        status = "RECOVERED" if phase == "recovery_orientation" else "REACHABLE"
                except Exception as exc:
                    status = "IK_ERROR"
                    error = type(exc).__name__ + ": " + str(exc)
                record = {
                    "frame": frame_index,
                    "phase": phase,
                    "status": status,
                    "target_pose7": target.tolist(),
                    "actual_pose7": actual.tolist(),
                    "position_error_m": None if error else position_error,
                    "rotation_error_rad": None if error else rotation_error,
                    "error": error,
                    "expected_status": expected_status,
                }
                records.append(record)
                trace.append(record)
                raw = task_env.get_observation().front_rgb
                frames.append(draw_panel(raw, record, args.video_width, args.video_height, trace[-20:]))
                frame_index += 1
    finally:
        env.shutdown()

    video_path = args.output_dir / "ik_stress_test.mp4"
    imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    failures = [r for r in records if r["status"] == "IK_ERROR"]
    summary = {
        "task": "close_laptop_lid",
        "artifact_type": "ik_stress_test",
        "video": str(video_path),
        "frames": len(frames),
        "ik_error_frames": len(failures),
        "ik_error_examples": [r for r in failures[:3]],
        "note": "This is a controlled target stress test, not a natural policy rollout.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
