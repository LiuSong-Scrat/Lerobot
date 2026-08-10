#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import REAL_DATA_ROOT
    from .camera_motion_utils import (
        camera_to_model_world_transforms,
        matrix_from_json_record,
        stationary_pose_report,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT
    from camera_motion_utils import (
        camera_to_model_world_transforms,
        matrix_from_json_record,
        stationary_pose_report,
    )


DEFAULT_BESTMAN_ROOT = Path("/home/liusong/ProgramFiles/BestMan")
cv2 = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record aligned RGB-D frames from BestMan RealSense cameras."
    )
    parser.add_argument("--bestman-root", default=str(DEFAULT_BESTMAN_ROOT))
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BESTMAN_ROOT / "Config/default_franka3.yaml"),
        help="BestMan YAML config containing Camera.L515/D435I.",
    )
    parser.add_argument(
        "--camera",
        default="overhead",
        choices=("overhead", "hand", "L515", "D435I"),
        help="overhead prefers L515 and falls back to D435I; hand prefers D435I.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "Override the selected BestMan camera RGB/depth stream FPS for this run. "
            "The value is applied to existing FPS fields under Camera.L515/D435I; "
            "unsupported stream profiles are rejected by RealSense when the camera opens."
        ),
    )
    parser.add_argument("--output", default=None, help="Output sequence directory.")
    parser.add_argument("--num-frames", type=int, default=0, help="0 means record until Ctrl-C.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means no duration limit.")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--preview-wait-ms",
        type=int,
        default=1,
        help=(
            "OpenCV keyboard polling delay. The legacy value 30 adds 30 ms to every "
            "recording iteration; 1 keeps the UI responsive without throttling capture."
        ),
    )
    parser.add_argument(
        "--debug-visualization",
        action="store_true",
        help=(
            "Show an RGB-D diagnostic dashboard with depth validity, RGB/depth edge alignment, "
            "point-cloud projections, timing, and optional IMU/odometry diagnostics. This never "
            "changes the frames written to disk."
        ),
    )
    parser.add_argument(
        "--debug-point-stride",
        type=int,
        default=4,
        help="Pixel stride used only by the diagnostic point-cloud view.",
    )
    parser.add_argument(
        "--debug-max-points",
        type=int,
        default=30000,
        help="Maximum number of points rendered by the diagnostic views.",
    )
    parser.add_argument(
        "--debug-depth-min-m",
        type=float,
        default=0.10,
        help="Near bound used only for depth coloring and diagnostic point filtering.",
    )
    parser.add_argument(
        "--debug-depth-max-m",
        type=float,
        default=2.0,
        help="Far bound used only for depth coloring and diagnostic point filtering.",
    )
    parser.add_argument(
        "--debug-save-every",
        type=int,
        default=0,
        help="Save one dashboard PNG every N saved frames; 0 disables dashboard snapshots.",
    )
    parser.add_argument(
        "--debug-update-every",
        type=int,
        default=1,
        help=(
            "Update the expensive diagnostic dashboard/ICP every N camera frames while "
            "reusing the latest dashboard between updates. This never drops saved RGB-D frames."
        ),
    )
    parser.add_argument(
        "--debug-rgbd-odometry",
        action="store_true",
        help=(
            "Estimate frame-to-frame camera motion with RGB-D point-to-plane ICP and visualize "
            "the cloud in model world (the first overview-camera frame). "
            "Use this only as a static-scene "
            "diagnostic, not as camera-pose ground truth for training."
        ),
    )
    parser.add_argument(
        "--debug-headless",
        action="store_true",
        help=(
            "Run RGB-D diagnostics and save reports without opening an OpenCV window. "
            "Useful on a headless acquisition server."
        ),
    )
    parser.add_argument(
        "--debug-odometry-voxel-m",
        type=float,
        default=0.02,
        help="Voxel size for diagnostic RGB-D ICP.",
    )
    parser.add_argument(
        "--debug-odometry-max-correspondence-m",
        type=float,
        default=0.06,
        help="Maximum ICP correspondence distance for diagnostic RGB-D odometry.",
    )
    parser.add_argument(
        "--debug-odometry-robust-kernel-m",
        type=float,
        default=0.01,
        help=(
            "Tukey robust-kernel scale for rejecting moving hands, manipulated objects, "
            "and depth outliers during diagnostic RGB-D odometry."
        ),
    )
    parser.add_argument(
        "--debug-odometry-anchor-min-fitness",
        type=float,
        default=0.30,
        help="Minimum fixed-world reference overlap required to apply drift correction.",
    )
    parser.add_argument(
        "--debug-odometry-anchor-max-correction-m",
        type=float,
        default=0.10,
        help="Reject a fixed-world correction that disagrees with local odometry by more than this distance.",
    )
    parser.add_argument(
        "--debug-odometry-anchor-max-correction-deg",
        type=float,
        default=10.0,
        help="Rotation counterpart of --debug-odometry-anchor-max-correction-m.",
    )
    parser.add_argument(
        "--debug-odometry-bootstrap-frames",
        type=int,
        default=5,
        help=(
            "Initial frames used to learn persistent background while holding the world origin. "
            "Keep the camera still during this short interval."
        ),
    )
    parser.add_argument(
        "--stationary-pose-check",
        action="store_true",
        help=(
            "Treat the camera as stationary, run diagnostic RGB-D odometry, save every estimated "
            "camera-to-world pose, and fail if drift exceeds the configured thresholds."
        ),
    )
    parser.add_argument(
        "--stationary-min-frames",
        type=int,
        default=90,
        help="Minimum diagnostic poses required by --stationary-pose-check.",
    )
    parser.add_argument(
        "--stationary-max-translation-drift-m",
        type=float,
        default=0.01,
        help="Maximum allowed camera translation drift from the first frame.",
    )
    parser.add_argument(
        "--stationary-max-rotation-drift-deg",
        type=float,
        default=1.0,
        help="Maximum allowed camera rotation drift from the first frame.",
    )
    parser.add_argument(
        "--stationary-min-accepted-ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of RGB-D ICP updates that must be accepted.",
    )
    parser.add_argument(
        "--record-imu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record asynchronous raw RealSense gyro/accelerometer samples to imu.jsonl and attach "
            "the latest samples to frames.jsonl. Raw L515/D435I IMU data is not a 6-DoF camera pose."
        ),
    )
    parser.add_argument(
        "--require-imu",
        action="store_true",
        help="Fail instead of warning when --record-imu cannot start.",
    )
    parser.add_argument(
        "--imu-gyro-fps",
        type=int,
        default=200,
        help="Preferred RealSense gyro rate; the closest advertised stream is selected.",
    )
    parser.add_argument(
        "--imu-accel-fps",
        type=int,
        default=100,
        help="Preferred RealSense accelerometer rate; the closest advertised stream is selected.",
    )
    parser.add_argument(
        "--space-toggle-recording",
        action="store_true",
        help=(
            "Start paused, press Space to start recording, press Space again to pause. "
            "Saved segments are written to segments.txt for build_humanhand_hdf5_dataset.py --segments."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=("legacy", "compressed", "video"),
        default="compressed",
        help=(
            "legacy: color PNG + float32 depth NPY; "
            "compressed: color JPEG + uint16 depth PNG in millimeters; "
            "video: color MP4 + uint16 depth PNG in millimeters."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality for --storage compressed.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=15.0,
        help="FPS metadata for --storage video.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="FourCC for --storage video, e.g. mp4v or avc1.",
    )
    parser.add_argument(
        "--save-depth-npy",
        action="store_true",
        help="Also save float32 meter depth NPY for maximum compatibility.",
    )
    parser.add_argument(
        "--save-color-png",
        action="store_true",
        help="Also save lossless color PNG beside compressed/video color.",
    )
    parser.add_argument(
        "--save-depth-png",
        action="store_true",
        help="Backward-compatible alias; legacy mode also writes uint16 millimeter depth PNG.",
    )
    parser.add_argument(
        "--camera-trajectory-mode",
        choices=("static", "rgbd_odometry", "external", "none"),
        default="static",
        help=(
            "Camera-pose handling for saved frames. static writes an identity pose for every "
            "frame (legacy fixed camera); rgbd_odometry estimates all poses after capture so "
            "ICP cannot reduce acquisition FPS; external attaches a complete SLAM/VIO JSONL; "
            "none preserves pose-free legacy files."
        ),
    )
    parser.add_argument(
        "--external-camera-pose-jsonl",
        default=None,
        help=(
            "Input SLAM/VIO trajectory used by --camera-trajectory-mode=external. "
            "Each line must contain record_index and a full camera_to_tracking pose."
        ),
    )
    parser.add_argument(
        "--camera-trajectory-output-jsonl",
        default="camera_pose.jsonl",
        help="Output path relative to the sequence directory, or an absolute path.",
    )
    parser.add_argument(
        "--camera-trajectory-max-sync-error-ms",
        type=float,
        default=20.0,
        help=(
            "Maximum RGB-D/pose timestamp mismatch for an external trajectory. "
            "record_index matching is always mandatory."
        ),
    )
    parser.add_argument(
        "--visualize-aligned-point-cloud",
        "--visualize-aligned-cloud",
        dest="visualize_aligned_point_cloud",
        action="store_true",
        help=(
            "After recording, replay every saved RGB-D point cloud after transforming it into "
            "the first saved frame. The Open3D view includes a fixed world-origin XYZ frame, "
            "an origin marker, the first cloud, the moving camera frame, and its trajectory."
        ),
    )
    parser.add_argument(
        "--playback-only",
        action="store_true",
        help=(
            "Read an existing sequence from --output and run aligned 3D playback without "
            "opening a camera or modifying frames.jsonl, metadata.json, or camera_pose.jsonl. "
            "By default <output>/camera_pose.jsonl is used."
        ),
    )
    parser.add_argument(
        "--aligned-point-cloud-pose-source",
        choices=("auto", "diagnostic", "external"),
        default="auto",
        help=(
            "Pose source for aligned playback. auto uses --aligned-point-cloud-pose-jsonl when "
            "provided, otherwise diagnostic RGB-D odometry. external is intended for SLAM/VIO."
        ),
    )
    parser.add_argument(
        "--aligned-point-cloud-pose-jsonl",
        default=None,
        help=(
            "SLAM/VIO JSONL containing record_index and camera_to_tracking poses. "
            "Every saved RGB-D frame must have one valid pose."
        ),
    )
    parser.add_argument(
        "--aligned-point-cloud-fps",
        type=float,
        default=15.0,
        help="Playback speed for the aligned Open3D visualization.",
    )
    parser.add_argument(
        "--aligned-point-cloud-point-stride",
        type=int,
        default=2,
        help="Pixel stride used when reconstructing each playback point cloud.",
    )
    parser.add_argument(
        "--aligned-point-cloud-max-points",
        type=int,
        default=100000,
        help="Maximum rendered points per aligned playback frame.",
    )
    parser.add_argument(
        "--aligned-point-cloud-axis-size-m",
        type=float,
        default=0.20,
        help="Length of the fixed world-origin XYZ axes in metres.",
    )
    parser.add_argument(
        "--aligned-point-cloud-loop",
        action="store_true",
        help="Loop the aligned playback instead of pausing on its final frame.",
    )
    parser.add_argument(
        "--aligned-point-cloud-hold-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the Open3D window open on the final frame. Disable for automated capture/validation runs."
        ),
    )
    args = parser.parse_args()
    if args.playback_only:
        if not args.output:
            raise ValueError("--playback-only requires an existing --output directory.")
        playback_output_dir = Path(args.output).expanduser().resolve()
        if not playback_output_dir.is_dir():
            raise FileNotFoundError(playback_output_dir)
        if not args.aligned_point_cloud_pose_jsonl:
            pose_path = (
                Path(args.external_camera_pose_jsonl).expanduser().resolve()
                if args.external_camera_pose_jsonl
                else _trajectory_output_path(
                    playback_output_dir,
                    str(args.camera_trajectory_output_jsonl),
                )
            )
            if not pose_path.is_file():
                raise FileNotFoundError(
                    f"Playback camera trajectory not found: {pose_path}. "
                    "Pass --aligned-point-cloud-pose-jsonl explicitly."
                )
            args.aligned_point_cloud_pose_jsonl = str(pose_path)
        args.aligned_point_cloud_pose_source = "external"
        args.visualize_aligned_point_cloud = True
    playback_uses_external_pose = args.aligned_point_cloud_pose_source == "external" or (
        args.aligned_point_cloud_pose_source == "auto"
        and (bool(args.aligned_point_cloud_pose_jsonl) or args.camera_trajectory_mode != "none")
    )
    if args.visualize_aligned_point_cloud and not playback_uses_external_pose:
        args.debug_rgbd_odometry = True
        args.debug_visualization = True
    if args.stationary_pose_check:
        args.debug_rgbd_odometry = True
        args.debug_visualization = True
        if args.num_frames <= 0 and args.duration_s <= 0.0:
            args.num_frames = int(args.stationary_min_frames)
    if args.space_toggle_recording:
        args.show = True
    if args.debug_visualization or args.debug_rgbd_odometry:
        args.debug_visualization = True
        if not args.debug_headless:
            args.show = True
    if args.require_imu:
        args.record_imu = True
    if args.fps is not None and int(args.fps) <= 0:
        raise ValueError("--fps must be a positive integer.")
    _validate_debug_args(args)

    global cv2
    import cv2 as cv2_module

    cv2 = cv2_module

    if args.playback_only:
        _play_aligned_point_cloud_sequence(
            output_dir=Path(args.output).expanduser().resolve(),
            args=args,
            diagnostic_saved_poses={},
        )
        return

    bestman_root = Path(args.bestman_root).resolve()
    if str(bestman_root) not in sys.path:
        sys.path.insert(0, str(bestman_root))

    from Sensor.Camera_Realsense import Camera_Realsense

    cfg = _load_yaml_namespace(Path(args.config))
    camera_fps_overrides: list[str] = []
    if args.fps is not None:
        camera_fps_overrides = _override_camera_stream_fps(
            cfg.Camera,
            args.camera,
            int(args.fps),
        )
        print(
            f"Requested RGB-D FPS override: {int(args.fps)}; "
            f"updated {', '.join(camera_fps_overrides)}"
        )
    output_dir = Path(args.output) if args.output else _default_output_dir(args.camera)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "color"
    color_jpg_dir = output_dir / "color_jpg"
    depth_dir = output_dir / "depth_m"
    depth_png_dir = output_dir / "depth_png"
    debug_dir = output_dir / "debug"
    if args.storage == "legacy" or args.save_color_png:
        color_dir.mkdir(exist_ok=True)
    if args.storage == "compressed":
        color_jpg_dir.mkdir(exist_ok=True)
    if args.storage == "legacy" or args.save_depth_npy:
        depth_dir.mkdir(exist_ok=True)
    if args.storage in ("compressed", "video") or args.save_depth_png:
        depth_png_dir.mkdir(exist_ok=True)
    if args.debug_save_every > 0:
        debug_dir.mkdir(exist_ok=True)

    camera = None
    motion_recorder = None
    debug_visualizer = None
    frames_file = None
    metadata = None
    color_video = None
    color_video_rel = Path("color.mp4")
    recording = not args.space_toggle_recording
    active_segment_start = 0 if recording and args.space_toggle_recording else None
    recorded_segments: list[dict] = []
    space_events: list[dict] = []
    diagnostic_saved_poses: dict[int, np.ndarray] = {}
    saved_wall_times: list[float] = []
    latest_debug_preview: np.ndarray | None = None
    raw_frame_count = 0
    stationary_report_data: dict | None = None
    stationary_check_failed = False
    try:
        camera_name, camera_cfg, camera = _open_camera(Camera_Realsense, cfg.Camera, args.camera)
        init_delay = float(getattr(cfg.Camera, "init_delay", 0.0))
        if init_delay > 0.0:
            time.sleep(init_delay)

        if args.record_imu:
            try:
                motion_recorder = RealSenseMotionRecorder(
                    output_dir / "imu.jsonl",
                    preferred_gyro_fps=args.imu_gyro_fps,
                    preferred_accel_fps=args.imu_accel_fps,
                )
                motion_recorder.start(camera)
                print(
                    "Recording raw RealSense IMU samples. "
                    "Important: accelerometer + gyroscope samples are not a 6-DoF camera pose."
                )
            except Exception as exc:
                if motion_recorder is not None:
                    motion_recorder.close()
                    motion_recorder = None
                message = f"Could not start RealSense IMU recording: {exc}"
                if args.require_imu:
                    raise RuntimeError(message) from exc
                print(f"[warn] {message}")

        for _ in range(max(0, int(args.warmup_frames))):
            camera.get_rgbd_image()

        probe_frame = camera.get_rgbd_image()
        if args.debug_visualization:
            debug_visualizer = RGBDDebugVisualizer(args, debug_dir)
        if args.storage == "video":
            color_video = _open_color_video_writer(
                output_dir / color_video_rel,
                probe_frame.color_bgr.shape,
                args.video_fps,
                args.video_codec,
            )

        metadata = {
            "format": "handpose_rgbd_sequence_v2",
            "camera_request": args.camera,
            "camera_config_name": camera_name,
            "camera_dev_name": getattr(camera_cfg, "dev_name", None),
            "requested_rgbd_fps": int(args.fps) if args.fps is not None else None,
            "camera_fps_config_fields": camera_fps_overrides,
            "created_unix_s": time.time(),
            "intrinsics": _intrinsics_to_dict(probe_frame.intrinsics),
            "depth_unit": "meter",
            "storage": args.storage,
            "color_format": _color_format(args),
            "depth_format": _depth_format(args),
            "depth_png_unit": "millimeter_uint16",
            "jpeg_quality": int(args.jpeg_quality),
            "video_fps": float(args.video_fps) if args.storage == "video" else None,
            "video_codec": args.video_codec if args.storage == "video" else None,
            "color_video_path": (color_video_rel.as_posix() if args.storage == "video" else None),
            "space_toggle_recording": bool(args.space_toggle_recording),
            "preview_wait_ms": int(args.preview_wait_ms),
            "debug_visualization": bool(args.debug_visualization),
            "debug_rgbd_odometry": bool(args.debug_rgbd_odometry),
            "debug_headless": bool(args.debug_headless),
            "debug_update_every": int(args.debug_update_every),
            "debug_odometry_config": {
                "voxel_m": float(args.debug_odometry_voxel_m),
                "max_correspondence_m": float(args.debug_odometry_max_correspondence_m),
                "robust_kernel_m": float(args.debug_odometry_robust_kernel_m),
                "anchor_min_fitness": float(args.debug_odometry_anchor_min_fitness),
                "anchor_max_correction_m": float(args.debug_odometry_anchor_max_correction_m),
                "anchor_max_correction_deg": float(args.debug_odometry_anchor_max_correction_deg),
                "bootstrap_frames": int(args.debug_odometry_bootstrap_frames),
            },
            "stationary_pose_check": bool(args.stationary_pose_check),
            "imu_recording_requested": bool(args.record_imu),
            "imu_recording_active": motion_recorder is not None,
            "imu_path": "imu.jsonl" if motion_recorder is not None else None,
            "imu_profiles": (motion_recorder.selected_profiles if motion_recorder is not None else None),
            "imu_device": (motion_recorder.device_metadata if motion_recorder is not None else None),
            "imu_semantics": (
                "asynchronous_raw_accelerometer_and_gyroscope_not_6dof_pose"
                if motion_recorder is not None
                else None
            ),
            "camera_pose_recorded": False,
            "camera_trajectory_mode": str(args.camera_trajectory_mode),
            "camera_trajectory_output": str(args.camera_trajectory_output_jsonl),
            "diagnostic_camera_pose_estimated": bool(args.debug_rgbd_odometry),
            "diagnostic_camera_pose_semantics": (
                "rgbd_icp_T_model_world<-current_camera_not_training_ground_truth"
                if args.debug_rgbd_odometry
                else None
            ),
            "bestman_root": str(bestman_root),
            "config": str(Path(args.config).resolve()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frames_file = open(  # noqa: SIM115
            output_dir / "frames.jsonl", "w", encoding="utf-8"
        )

        start_s = time.time()
        index = 0
        while True:
            if args.num_frames > 0 and index >= args.num_frames:
                break
            if args.duration_s > 0.0 and time.time() - start_s >= args.duration_s:
                break

            frame = camera.get_rgbd_image()
            raw_frame_count += 1
            imu_snapshot = (
                motion_recorder.snapshot(frame.timestamp_ms) if motion_recorder is not None else None
            )
            saved_this_frame = False
            saved_record_index = None
            if recording:
                saved_record_index = index
                _write_frame_record(
                    output_dir=output_dir,
                    args=args,
                    frame=frame,
                    index=index,
                    frames_file=frames_file,
                    color_video=color_video,
                    color_video_rel=color_video_rel,
                    imu_snapshot=imu_snapshot,
                )
                index += 1
                saved_wall_times.append(time.perf_counter())
                saved_this_frame = True

            preview = None
            if debug_visualizer is not None:
                should_update_debug = (
                    latest_debug_preview is None or (raw_frame_count - 1) % int(args.debug_update_every) == 0
                )
                if should_update_debug:
                    latest_debug_preview = debug_visualizer.render(
                        frame,
                        saved_index=index,
                        recording=recording,
                        imu_snapshot=imu_snapshot,
                    )
                preview = latest_debug_preview.copy()

            if (
                saved_record_index is not None
                and debug_visualizer is not None
                and debug_visualizer.odometry is not None
                and debug_visualizer.odometry.pose_history
            ):
                diagnostic_saved_poses[int(saved_record_index)] = (
                    debug_visualizer.odometry.camera_to_world.copy()
                )

            if args.show:
                status = "REC" if recording else "PAUSED"
                status_color = (0, 0, 255) if recording else (0, 220, 255)
                if preview is not None:
                    window_name = "record RGB-D diagnostics"
                else:
                    preview = frame.color_bgr.copy()
                    window_name = "record RGB-D"
                status_y = 54 if debug_visualizer is not None else 32
                help_y = 82 if debug_visualizer is not None else 64
                cv2.putText(
                    preview,
                    f"{status} saved={index} segments={len(recorded_segments)}",
                    (16, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                if args.space_toggle_recording:
                    cv2.putText(
                        preview,
                        "Space: start/resume or pause  Q/Esc: finish",
                        (16, help_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(int(args.preview_wait_ms)) & 0xFF
                if key in (27, ord("q")):
                    break
                if debug_visualizer is not None and key == ord("r"):
                    debug_visualizer.reset_odometry_reference()
                    print("Reset diagnostic RGB-D odometry reference to the next frame.")
                if debug_visualizer is not None and key == ord("s"):
                    snapshot_path = debug_visualizer.save_snapshot(preview, index)
                    print(f"Saved debug dashboard: {snapshot_path}")
                if args.space_toggle_recording and key == ord(" "):
                    if recording:
                        end_index = index - 1
                        if active_segment_start is not None and end_index >= active_segment_start:
                            recorded_segments.append(
                                {
                                    "start": int(active_segment_start),
                                    "end": int(end_index),
                                }
                            )
                        space_events.append(
                            _space_event(
                                "pause",
                                end_index,
                                frame.timestamp_ms,
                                frame.frame_number,
                            )
                        )
                        recording = False
                        active_segment_start = None
                        print(f"Paused recording at saved frame {end_index}")
                    else:
                        active_segment_start = index
                        space_events.append(
                            _space_event("start", index, frame.timestamp_ms, frame.frame_number)
                        )
                        recording = True
                        if not saved_this_frame:
                            saved_record_index = index
                            _write_frame_record(
                                output_dir=output_dir,
                                args=args,
                                frame=frame,
                                index=index,
                                frames_file=frames_file,
                                color_video=color_video,
                                color_video_rel=color_video_rel,
                                imu_snapshot=imu_snapshot,
                            )
                            index += 1
                            saved_wall_times.append(time.perf_counter())
                            if (
                                debug_visualizer is not None
                                and debug_visualizer.odometry is not None
                                and debug_visualizer.odometry.pose_history
                            ):
                                diagnostic_saved_poses[int(saved_record_index)] = (
                                    debug_visualizer.odometry.camera_to_world.copy()
                                )
                        print(f"Started recording at saved frame {active_segment_start}")

    except KeyboardInterrupt:
        pass
    finally:
        if (
            args.space_toggle_recording
            and recording
            and active_segment_start is not None
            and index > active_segment_start
        ):
            recorded_segments.append({"start": int(active_segment_start), "end": int(index - 1)})
            space_events.append(_space_event("stop", index - 1, None, None))
        if args.stationary_pose_check:
            if debug_visualizer is None or debug_visualizer.odometry is None:
                stationary_report_data = {
                    "passed": False,
                    "reason": "diagnostic_rgbd_odometry_unavailable",
                }
            else:
                stationary_report_data = debug_visualizer.save_odometry_artifacts(
                    min_frames=int(args.stationary_min_frames),
                    max_translation_drift_m=float(args.stationary_max_translation_drift_m),
                    max_rotation_drift_deg=float(args.stationary_max_rotation_drift_deg),
                    min_accepted_ratio=float(args.stationary_min_accepted_ratio),
                    motion_recorder=motion_recorder,
                )
            stationary_check_failed = not bool(stationary_report_data.get("passed", False))
            print(_format_stationary_report(stationary_report_data))
            if metadata is not None:
                metadata["stationary_pose_report"] = stationary_report_data
                (output_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        if color_video is not None:
            color_video.release()
        if motion_recorder is not None:
            motion_recorder.close()
        if frames_file is not None:
            frames_file.close()
        if camera is not None:
            camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    if metadata is not None:
        capture_stats = (
            _capture_rate_summary(
                output_dir,
                saved_wall_times,
                debug_pipeline_is_synchronous=bool(args.debug_visualization),
            )
            if index > 0
            else {
                "frame_count": 0,
                "host_saved_fps": 0.0,
                "camera_timestamp_fps": 0.0,
                "debug_pipeline_is_synchronous": bool(args.debug_visualization),
            }
        )
        metadata["capture_stats"] = capture_stats
        print(
            "Capture summary: "
            f"saved={capture_stats['frame_count']} "
            f"host={capture_stats['host_saved_fps']:.2f} FPS "
            f"camera={capture_stats['camera_timestamp_fps']:.2f} FPS"
        )
        if index > 0:
            trajectory_path, trajectory_metadata = _finalize_camera_trajectory(
                output_dir=output_dir,
                args=args,
            )
        else:
            trajectory_path = None
            trajectory_metadata = {
                "camera_pose_recorded": False,
                "camera_trajectory_mode": str(args.camera_trajectory_mode),
                "camera_pose_path": None,
                "camera_pose_frame_count": 0,
            }
        metadata.update(trajectory_metadata)
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if (
            args.visualize_aligned_point_cloud
            and args.aligned_point_cloud_pose_source in {"auto", "external"}
            and not args.aligned_point_cloud_pose_jsonl
            and trajectory_path is not None
        ):
            args.aligned_point_cloud_pose_jsonl = str(trajectory_path)

    if args.space_toggle_recording:
        _write_segments(output_dir, recorded_segments, space_events, metadata)
    print(f"Saved RGB-D sequence to {output_dir}")
    if args.visualize_aligned_point_cloud and index > 0:
        _play_aligned_point_cloud_sequence(
            output_dir=output_dir,
            args=args,
            diagnostic_saved_poses=diagnostic_saved_poses,
        )
    if stationary_check_failed:
        raise RuntimeError(
            "Stationary camera-pose check failed. See "
            f"{output_dir / 'debug' / 'stationary_pose_report.json'}."
        )


def _load_yaml_namespace(path: Path) -> SimpleNamespace:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load BestMan YAML config") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _to_namespace(data)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _validate_debug_args(args: argparse.Namespace) -> None:
    if args.preview_wait_ms < 1:
        raise ValueError("--preview-wait-ms must be at least 1.")
    if args.debug_update_every <= 0:
        raise ValueError("--debug-update-every must be positive.")
    if args.debug_point_stride <= 0:
        raise ValueError("--debug-point-stride must be positive.")
    if args.debug_max_points <= 0:
        raise ValueError("--debug-max-points must be positive.")
    if args.debug_depth_min_m < 0.0 or args.debug_depth_max_m <= args.debug_depth_min_m:
        raise ValueError("--debug-depth-max-m must be greater than --debug-depth-min-m >= 0.")
    if args.debug_save_every < 0:
        raise ValueError("--debug-save-every must be non-negative.")
    if args.debug_odometry_voxel_m <= 0.0:
        raise ValueError("--debug-odometry-voxel-m must be positive.")
    if args.debug_odometry_max_correspondence_m <= 0.0:
        raise ValueError("--debug-odometry-max-correspondence-m must be positive.")
    if args.debug_odometry_robust_kernel_m <= 0.0:
        raise ValueError("--debug-odometry-robust-kernel-m must be positive.")
    if not 0.0 <= args.debug_odometry_anchor_min_fitness <= 1.0:
        raise ValueError("--debug-odometry-anchor-min-fitness must be in [0, 1].")
    if args.debug_odometry_anchor_max_correction_m <= 0.0:
        raise ValueError("--debug-odometry-anchor-max-correction-m must be positive.")
    if args.debug_odometry_anchor_max_correction_deg <= 0.0:
        raise ValueError("--debug-odometry-anchor-max-correction-deg must be positive.")
    if args.debug_odometry_bootstrap_frames < 3:
        raise ValueError("--debug-odometry-bootstrap-frames must be at least 3.")
    if args.stationary_min_frames <= 0:
        raise ValueError("--stationary-min-frames must be positive.")
    if args.stationary_max_translation_drift_m < 0.0:
        raise ValueError("--stationary-max-translation-drift-m must be non-negative.")
    if args.stationary_max_rotation_drift_deg < 0.0:
        raise ValueError("--stationary-max-rotation-drift-deg must be non-negative.")
    if not 0.0 <= args.stationary_min_accepted_ratio <= 1.0:
        raise ValueError("--stationary-min-accepted-ratio must be in [0, 1].")
    if args.imu_gyro_fps <= 0 or args.imu_accel_fps <= 0:
        raise ValueError("IMU stream rates must be positive.")
    if args.camera_trajectory_max_sync_error_ms < 0.0:
        raise ValueError("--camera-trajectory-max-sync-error-ms must be non-negative.")
    if (
        args.camera_trajectory_mode == "external"
        and not args.external_camera_pose_jsonl
        and not args.playback_only
    ):
        raise ValueError("--camera-trajectory-mode=external requires --external-camera-pose-jsonl.")
    if (
        args.visualize_aligned_point_cloud
        and args.aligned_point_cloud_pose_source == "diagnostic"
        and args.debug_update_every != 1
    ):
        raise ValueError(
            "Diagnostic aligned playback requires --debug-update-every=1 so every "
            "saved RGB-D frame has a matching pose."
        )
    if args.stationary_pose_check and args.debug_update_every != 1:
        raise ValueError("--stationary-pose-check requires --debug-update-every=1.")
    if args.aligned_point_cloud_fps <= 0.0:
        raise ValueError("--aligned-point-cloud-fps must be positive.")
    if args.aligned_point_cloud_point_stride <= 0:
        raise ValueError("--aligned-point-cloud-point-stride must be positive.")
    if args.aligned_point_cloud_max_points <= 0:
        raise ValueError("--aligned-point-cloud-max-points must be positive.")
    if args.aligned_point_cloud_axis_size_m <= 0.0:
        raise ValueError("--aligned-point-cloud-axis-size-m must be positive.")
    if (
        args.visualize_aligned_point_cloud
        and args.aligned_point_cloud_pose_source == "external"
        and not args.aligned_point_cloud_pose_jsonl
        and args.camera_trajectory_mode == "none"
    ):
        raise ValueError(
            "--aligned-point-cloud-pose-source=external requires --aligned-point-cloud-pose-jsonl."
        )


def _realsense_device_metadata(device, rs) -> dict:
    metadata = {}
    fields = {
        "name": "name",
        "serial_number": "serial_number",
        "firmware_version": "firmware_version",
        "product_line": "product_line",
        "usb_type_descriptor": "usb_type_descriptor",
    }
    for name, field_name in fields.items():
        try:
            field = getattr(rs.camera_info, field_name)
            if device.supports(field):
                metadata[name] = str(device.get_info(field))
        except Exception:
            continue
    return metadata


def _realsense_extrinsics_metadata(source_profile, color_profile) -> dict | None:
    try:
        extrinsics = source_profile.get_extrinsics_to(color_profile)
    except Exception:
        return None
    return {
        # librealsense defines this as its native flat rs2_extrinsics rotation
        # array. Keeping the flat SDK order avoids silently transposing it.
        "rotation_flat_realsense_sdk": [float(value) for value in extrinsics.rotation],
        "translation_m": [float(value) for value in extrinsics.translation],
        "transform_direction": "motion_sensor_to_color_camera",
    }


def _realsense_motion_intrinsics_metadata(profile) -> dict | None:
    try:
        intrinsics = profile.as_motion_stream_profile().get_motion_intrinsics()
    except Exception:
        return None
    return {
        "data": np.asarray(intrinsics.data, dtype=np.float64).tolist(),
        "noise_variances": [float(value) for value in intrinsics.noise_variances],
        "bias_variances": [float(value) for value in intrinsics.bias_variances],
    }


class RealSenseMotionRecorder:
    """Record raw RealSense IMU streams without pretending they are a camera pose.

    Supported L515 and D435I configurations may expose acceleration and angular
    velocity. Those measurements are useful for synchronization and orientation
    diagnostics, but double-integrating them is not a reliable translation
    estimate. Consequently this class never writes a ``camera_to_tracking`` or
    ``camera_to_world`` transform.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        preferred_gyro_fps: int,
        preferred_accel_fps: int,
        history_size: int = 2000,
    ) -> None:
        self.output_path = Path(output_path)
        self.preferred_fps = {
            "gyro": int(preferred_gyro_fps),
            "accel": int(preferred_accel_fps),
        }
        self._history = {
            "gyro": deque(maxlen=history_size),
            "accel": deque(maxlen=history_size),
        }
        self._lock = threading.Lock()
        self._file = None
        self._rs = None
        self._opened_sensors: list[object] = []
        self._last_gyro_timestamp_ms: float | None = None
        self._gyro_orientation = np.eye(3, dtype=np.float64)
        self.selected_profiles: dict[str, dict] = {}
        self.device_metadata: dict = {}

    def start(self, camera) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for RealSense IMU recording.") from exc

        if not hasattr(camera, "pipeline"):
            raise RuntimeError("BestMan camera does not expose its RealSense pipeline.")
        active_profile = camera.pipeline.get_active_profile()
        device = active_profile.get_device()
        self.device_metadata = _realsense_device_metadata(device, rs)
        color_profile = active_profile.get_stream(rs.stream.color)
        selected_by_sensor: list[tuple[object, list[object]]] = []
        selected_names: set[str] = set()

        for sensor in device.query_sensors():
            candidates: dict[str, list[object]] = {"gyro": [], "accel": []}
            for profile in sensor.get_stream_profiles():
                stream_type = profile.stream_type()
                if stream_type == rs.stream.gyro:
                    candidates["gyro"].append(profile)
                elif stream_type == rs.stream.accel:
                    candidates["accel"].append(profile)

            selected_profiles: list[object] = []
            for name in ("gyro", "accel"):
                if not candidates[name]:
                    continue
                profile = min(
                    candidates[name],
                    key=lambda item: abs(int(item.fps()) - self.preferred_fps[name]),
                )
                selected_profiles.append(profile)
                selected_names.add(name)
                self.selected_profiles[name] = {
                    "fps": int(profile.fps()),
                    "format": str(profile.format()),
                    "stream_index": int(profile.stream_index()),
                    "extrinsics_to_color": _realsense_extrinsics_metadata(
                        profile,
                        color_profile,
                    ),
                    "motion_intrinsics": _realsense_motion_intrinsics_metadata(profile),
                }
            if selected_profiles:
                selected_by_sensor.append((sensor, selected_profiles))

        missing = {"gyro", "accel"} - selected_names
        if missing:
            raise RuntimeError(
                f"RealSense device does not advertise required motion stream(s): {sorted(missing)}"
            )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")  # noqa: SIM115
        self._rs = rs
        try:
            for sensor, profiles in selected_by_sensor:
                sensor.open(profiles)
                sensor.start(self._on_motion_frame)
                self._opened_sensors.append(sensor)
        except Exception:
            self.close()
            raise

    def _on_motion_frame(self, frame) -> None:
        try:
            motion_frame = frame.as_motion_frame()
            profile = frame.get_profile()
            stream_type = profile.stream_type()
            if stream_type == self._rs.stream.gyro:
                name = "gyro"
                unit = "rad_per_s"
            elif stream_type == self._rs.stream.accel:
                name = "accel"
                unit = "m_per_s2"
            else:
                return
            vector = motion_frame.get_motion_data()
            sample = {
                "stream": name,
                "timestamp_ms": float(frame.get_timestamp()),
                "frame_number": int(frame.get_frame_number()),
                "xyz": [float(vector.x), float(vector.y), float(vector.z)],
                "unit": unit,
                "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                "wall_time_s": time.time(),
            }
            with self._lock:
                self._history[name].append(sample)
                if name == "gyro":
                    self._integrate_gyro_locked(sample)
                if self._file is not None:
                    self._file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception as exc:
            # A callback exception must not terminate RGB-D acquisition.
            print(f"[warn] RealSense IMU callback error: {exc}")

    def _integrate_gyro_locked(self, sample: dict) -> None:
        timestamp_ms = float(sample["timestamp_ms"])
        if self._last_gyro_timestamp_ms is None:
            self._last_gyro_timestamp_ms = timestamp_ms
            return
        dt = (timestamp_ms - self._last_gyro_timestamp_ms) * 1e-3
        self._last_gyro_timestamp_ms = timestamp_ms
        if not 0.0 < dt <= 0.1:
            return
        rotation_delta = _rotation_matrix_from_rotvec(np.asarray(sample["xyz"], dtype=np.float64) * dt)
        self._gyro_orientation = self._gyro_orientation @ rotation_delta

    def snapshot(self, rgb_timestamp_ms: float | None) -> dict:
        with self._lock:
            snapshot: dict = {
                "raw_imu_is_not_camera_pose": True,
                "gyro_integrated_orientation_debug_only": self._gyro_orientation.tolist(),
            }
            for name in ("gyro", "accel"):
                if not self._history[name]:
                    snapshot[name] = None
                    continue
                if rgb_timestamp_ms is None:
                    sample = self._history[name][-1]
                else:
                    sample = min(
                        self._history[name],
                        key=lambda item: abs(float(item["timestamp_ms"]) - float(rgb_timestamp_ms)),
                    )
                copied = dict(sample)
                copied["rgb_timestamp_delta_ms"] = (
                    None
                    if rgb_timestamp_ms is None
                    else float(sample["timestamp_ms"]) - float(rgb_timestamp_ms)
                )
                snapshot[name] = copied
            return snapshot

    def history(self, name: str, max_samples: int = 300) -> list[dict]:
        with self._lock:
            return [dict(item) for item in list(self._history[name])[-max_samples:]]

    def close(self) -> None:
        for sensor in reversed(self._opened_sensors):
            with suppress(Exception):
                sensor.stop()
            with suppress(Exception):
                sensor.close()
        self._opened_sensors.clear()
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


class RGBDOdometryDebugger:
    """Diagnostic-only dynamic-robust RGB-D camera-motion estimate.

    ``camera_to_world`` is relative to the first RGB-D frame. It is useful for
    acquisition debugging, but is not automatically promoted to training-pose
    ground truth. Local robust ICP supplies a motion prediction; registration
    against the fixed first-frame reference removes random-walk drift. A Tukey
    kernel suppresses independently moving hands, objects, and depth outliers.
    """

    def __init__(
        self,
        voxel_size: float,
        max_correspondence: float,
        robust_kernel: float,
        anchor_min_fitness: float,
        anchor_max_correction_m: float,
        anchor_max_correction_deg: float,
        bootstrap_frames: int,
    ) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("--debug-rgbd-odometry requires open3d.") from exc
        self.o3d = o3d
        self.voxel_size = float(voxel_size)
        self.max_correspondence = float(max_correspondence)
        self.robust_kernel = float(robust_kernel)
        self.anchor_min_fitness = float(anchor_min_fitness)
        self.anchor_max_correction_m = float(anchor_max_correction_m)
        self.anchor_max_correction_deg = float(anchor_max_correction_deg)
        self.bootstrap_frames = int(bootstrap_frames)
        self.previous = None
        self.reference = None
        self.reference_xyz: np.ndarray | None = None
        self.reference_stability: np.ndarray | None = None
        self.anchor_observations = 0
        self.camera_to_world = np.eye(4, dtype=np.float64)
        self.pose_history: list[np.ndarray] = []
        self.metrics_history: list[dict] = []
        self.last_metrics = {
            "initialized": False,
            "accepted": False,
            "fitness": 0.0,
            "inlier_rmse_m": float("nan"),
            "translation_m": 0.0,
            "rotation_deg": 0.0,
        }

    def reset(self) -> None:
        self.previous = None
        self.reference = None
        self.reference_xyz = None
        self.reference_stability = None
        self.anchor_observations = 0
        self.camera_to_world = np.eye(4, dtype=np.float64)
        self.pose_history.clear()
        self.metrics_history.clear()
        self.last_metrics = {
            "initialized": False,
            "accepted": False,
            "fitness": 0.0,
            "inlier_rmse_m": float("nan"),
            "translation_m": 0.0,
            "rotation_deg": 0.0,
        }

    def _record(self, metrics: dict) -> None:
        self.last_metrics = dict(metrics)
        self.pose_history.append(self.camera_to_world.copy())
        self.metrics_history.append(dict(metrics))

    def _make_point_cloud(self, xyz: np.ndarray, colors_bgr: np.ndarray) -> object:
        point_cloud = self.o3d.geometry.PointCloud()
        point_cloud.points = self.o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        colors = np.asarray(colors_bgr, dtype=np.float64)
        if colors.shape != (len(xyz), 3):
            raise ValueError(f"RGB-D odometry colors must have shape ({len(xyz)},3), got {colors.shape}.")
        # Open3D expects RGB. Channel reversal does not affect color distance,
        # but using the documented convention keeps saved/debug clouds clear.
        point_cloud.colors = self.o3d.utility.Vector3dVector(np.clip(colors[:, ::-1] / 255.0, 0.0, 1.0))
        point_cloud = point_cloud.voxel_down_sample(self.voxel_size)
        if len(point_cloud.points) >= 50:
            point_cloud.estimate_normals(
                self.o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(2.5 * self.voxel_size, self.max_correspondence),
                    max_nn=30,
                )
            )
        return point_cloud

    def _robust_point_to_plane(self):
        loss = self.o3d.pipelines.registration.TukeyLoss(k=self.robust_kernel)
        return self.o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    def _temporally_stable_anchor_clouds(
        self,
        point_cloud,
        predicted_pose: np.ndarray,
    ) -> tuple[object, object, dict]:
        """Keep only geometry persistent under the camera-rigid SE(3)."""

        aligned_current = copy.deepcopy(point_cloud)
        aligned_current.transform(np.asarray(predicted_pose, dtype=np.float64))
        consistency_distance = max(
            1.5 * self.robust_kernel,
            1.25 * self.voxel_size,
        )
        reference_distances = np.asarray(
            self.reference.compute_point_cloud_distance(aligned_current),
            dtype=np.float64,
        )
        consistent_reference = reference_distances <= consistency_distance
        if self.reference_stability is None:
            self.reference_stability = consistent_reference.astype(np.float64)
        else:
            self.reference_stability = 0.8 * self.reference_stability + 0.2 * consistent_reference.astype(
                np.float64
            )
        self.anchor_observations += 1

        reference_mask = self.reference_stability >= 0.55
        reference_indices = np.flatnonzero(reference_mask).tolist()
        min_anchor_points = 100
        base_metrics = {
            "stable_reference_ratio": float(reference_mask.mean()),
            "consistency_distance_m": consistency_distance,
        }
        if self.anchor_observations < 3 or len(reference_indices) < min_anchor_points:
            return (
                point_cloud,
                self.reference,
                {
                    **base_metrics,
                    "temporal_static_filter_active": False,
                    "static_source_ratio": 1.0,
                },
            )

        stable_reference = self.reference.select_by_index(reference_indices)
        source_distances = np.asarray(
            aligned_current.compute_point_cloud_distance(stable_reference),
            dtype=np.float64,
        )
        source_mask = source_distances <= consistency_distance
        source_indices = np.flatnonzero(source_mask).tolist()
        if len(source_indices) < min_anchor_points:
            return (
                point_cloud,
                stable_reference,
                {
                    **base_metrics,
                    "temporal_static_filter_active": False,
                    "static_source_ratio": float(source_mask.mean()),
                },
            )
        return (
            point_cloud.select_by_index(source_indices),
            stable_reference,
            {
                **base_metrics,
                "temporal_static_filter_active": True,
                "static_source_ratio": float(source_mask.mean()),
            },
        )

    def update(self, xyz: np.ndarray, colors_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        point_cloud = self._make_point_cloud(xyz, colors_bgr)
        if len(point_cloud.points) < 50:
            metrics = dict(self.last_metrics)
            metrics["accepted"] = False
            metrics["reason"] = "too_few_points"
            self._record(metrics)
            return _transform_points(np.asarray(xyz), self.camera_to_world), dict(metrics)
        if self.previous is None:
            self.previous = point_cloud
            self.reference = point_cloud
            self.reference_xyz = np.asarray(point_cloud.points).copy()
            self.reference_stability = np.ones(len(point_cloud.points), dtype=np.float64)
            metrics = {
                "initialized": True,
                "accepted": True,
                "estimator": "reference_initialization",
                "anchor_accepted": True,
                "fitness": 1.0,
                "inlier_rmse_m": 0.0,
                "translation_m": 0.0,
                "rotation_deg": 0.0,
            }
            self._record(metrics)
            return np.asarray(xyz), dict(metrics)

        criteria = self.o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        local_result = self.o3d.pipelines.registration.registration_icp(
            point_cloud,
            self.previous,
            self.max_correspondence,
            np.eye(4, dtype=np.float64),
            self._robust_point_to_plane(),
            criteria,
        )
        local_accepted = bool(
            np.isfinite(local_result.inlier_rmse)
            and local_result.fitness >= 0.15
            and local_result.inlier_rmse <= self.max_correspondence
        )
        predicted_pose = (
            self.camera_to_world @ np.asarray(local_result.transformation)
            if local_accepted
            else self.camera_to_world.copy()
        )

        anchor_result = None
        anchor_error = None
        bootstrap_active = self.anchor_observations < self.bootstrap_frames - 1
        temporal_prediction = self.camera_to_world if bootstrap_active else predicted_pose
        anchor_source, anchor_target, temporal_metrics = self._temporally_stable_anchor_clouds(
            point_cloud, temporal_prediction
        )
        try:
            if bootstrap_active:
                raise RuntimeError("temporal_background_bootstrap")
            if temporal_metrics["temporal_static_filter_active"]:
                # Apply color only after temporal consistency has removed
                # independently moving RGB-D regions.
                colored_result = self.o3d.pipelines.registration.registration_colored_icp(
                    anchor_source,
                    anchor_target,
                    self.max_correspondence,
                    predicted_pose,
                    self.o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    criteria,
                )
                anchor_initial = np.asarray(colored_result.transformation)
            else:
                # During background initialization, dynamic color must not
                # pull the fixed world anchor.
                anchor_initial = predicted_pose
            anchor_result = self.o3d.pipelines.registration.registration_icp(
                anchor_source,
                anchor_target,
                self.max_correspondence,
                anchor_initial,
                self._robust_point_to_plane(),
                criteria,
            )
        except RuntimeError as exc:
            # Poor overlap can make colored ICP singular. Local odometry
            # remains available, and the failure is recorded for auditing.
            anchor_error = str(exc)

        anchor_accepted = False
        anchor_correction_m: float | None = None
        anchor_correction_deg: float | None = None
        if anchor_result is not None:
            predicted_to_anchor = _invert_transform(predicted_pose) @ np.asarray(anchor_result.transformation)
            anchor_correction_m = float(np.linalg.norm(predicted_to_anchor[:3, 3]))
            anchor_correction_deg = _rotation_angle_deg(predicted_to_anchor[:3, :3])
            anchor_accepted = bool(
                np.isfinite(anchor_result.inlier_rmse)
                and anchor_result.fitness >= self.anchor_min_fitness
                and anchor_result.inlier_rmse <= self.max_correspondence
                and anchor_correction_m is not None
                and anchor_correction_m <= self.anchor_max_correction_m
                and anchor_correction_deg is not None
                and anchor_correction_deg <= self.anchor_max_correction_deg
            )

        accepted = bool(bootstrap_active or anchor_accepted or local_accepted)
        if bootstrap_active:
            estimator = "world_anchor_bootstrap_pose_held"
            fitness = float(local_result.fitness)
            inlier_rmse = float(local_result.inlier_rmse)
        elif anchor_accepted:
            self.camera_to_world = np.asarray(anchor_result.transformation)
            estimator = "fixed_world_anchor_robust_rgbd"
            fitness = float(anchor_result.fitness)
            inlier_rmse = float(anchor_result.inlier_rmse)
        elif local_accepted:
            self.camera_to_world = predicted_pose
            estimator = "local_robust_icp_fallback"
            fitness = float(local_result.fitness)
            inlier_rmse = float(local_result.inlier_rmse)
        else:
            estimator = "pose_held_no_valid_registration"
            fitness = 0.0
            inlier_rmse = float("nan")

        if accepted:
            self.previous = point_cloud
        angle = _rotation_angle_deg(self.camera_to_world[:3, :3])
        metrics = {
            "initialized": True,
            "accepted": accepted,
            "estimator": estimator,
            "anchor_accepted": anchor_accepted,
            "anchor_correction_m": anchor_correction_m,
            "anchor_correction_deg": anchor_correction_deg,
            "anchor_error": anchor_error,
            "fitness": fitness,
            "inlier_rmse_m": inlier_rmse,
            "local_fitness": float(local_result.fitness),
            "local_inlier_rmse_m": float(local_result.inlier_rmse),
            **temporal_metrics,
            "translation_m": float(np.linalg.norm(self.camera_to_world[:3, 3])),
            "rotation_deg": float(angle),
        }
        self._record(metrics)
        aligned = _transform_points(np.asarray(xyz), self.camera_to_world)
        return aligned, dict(metrics)


class RGBDDebugVisualizer:
    """OpenCV dashboard for validating L515 or D435I RGB-D acquisition."""

    TILE_WIDTH = 640
    TILE_HEIGHT = 360

    def __init__(self, args: argparse.Namespace, debug_dir: Path) -> None:
        self.args = args
        self.debug_dir = Path(debug_dir)
        self.wall_times: deque[float] = deque(maxlen=120)
        self.frame_timestamps: deque[float] = deque(maxlen=120)
        self.last_snapshot_index: int | None = None
        self.odometry = (
            RGBDOdometryDebugger(
                args.debug_odometry_voxel_m,
                args.debug_odometry_max_correspondence_m,
                args.debug_odometry_robust_kernel_m,
                args.debug_odometry_anchor_min_fitness,
                args.debug_odometry_anchor_max_correction_m,
                args.debug_odometry_anchor_max_correction_deg,
                args.debug_odometry_bootstrap_frames,
            )
            if args.debug_rgbd_odometry
            else None
        )

    def reset_odometry_reference(self) -> None:
        if self.odometry is not None:
            self.odometry.reset()

    def save_snapshot(self, dashboard: np.ndarray, saved_index: int) -> Path:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.debug_dir / f"dashboard_manual_{saved_index:06d}_{timestamp}.png"
        if not cv2.imwrite(str(snapshot_path), dashboard):
            raise RuntimeError(f"Failed to save debug dashboard: {snapshot_path}")
        return snapshot_path

    def save_odometry_artifacts(
        self,
        *,
        min_frames: int,
        max_translation_drift_m: float,
        max_rotation_drift_deg: float,
        min_accepted_ratio: float,
        motion_recorder: RealSenseMotionRecorder | None,
    ) -> dict:
        if self.odometry is None or not self.odometry.pose_history:
            report = {
                "passed": False,
                "reason": "no_rgbd_odometry_poses",
                "frame_count": 0,
            }
        else:
            poses = np.asarray(self.odometry.pose_history, dtype=np.float64)
            accepted = np.asarray(
                [bool(metrics.get("anchor_accepted", False)) for metrics in self.odometry.metrics_history],
                dtype=bool,
            )
            report = stationary_pose_report(
                poses,
                min_frames=min_frames,
                max_translation_drift_m=max_translation_drift_m,
                max_rotation_drift_deg=max_rotation_drift_deg,
                accepted=accepted,
                min_accepted_ratio=min_accepted_ratio,
            )
            report["pose_semantics"] = "diagnostic_rgbd_icp_T_model_world<-current_camera"
            report["model_world_definition"] = "first_overview_camera_frame"
            report["accepted_semantics"] = "fixed_world_anchor_registration_accepted"
            report["odometry_config"] = {
                "voxel_m": self.odometry.voxel_size,
                "max_correspondence_m": self.odometry.max_correspondence,
                "robust_kernel_m": self.odometry.robust_kernel,
                "anchor_min_fitness": self.odometry.anchor_min_fitness,
                "anchor_max_correction_m": self.odometry.anchor_max_correction_m,
                "anchor_max_correction_deg": self.odometry.anchor_max_correction_deg,
                "bootstrap_frames": self.odometry.bootstrap_frames,
            }
            report["training_ground_truth"] = False
            report["imu"] = _imu_stationary_metrics(motion_recorder)

            self.debug_dir.mkdir(parents=True, exist_ok=True)
            pose_path = self.debug_dir / "rgbd_odometry.jsonl"
            with pose_path.open("w", encoding="utf-8") as pose_file:
                for index, (pose, metrics) in enumerate(
                    zip(poses, self.odometry.metrics_history, strict=True)
                ):
                    pose_file.write(
                        json.dumps(
                            {
                                "frame_index": index,
                                "camera_to_world": pose.tolist(),
                                "transform_direction": "current_camera_to_model_world",
                                "metrics": metrics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            report["pose_jsonl"] = str(pose_path)

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.debug_dir / "stationary_pose_report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def render(
        self,
        frame,
        *,
        saved_index: int,
        recording: bool,
        imu_snapshot: dict | None,
    ) -> np.ndarray:
        self.wall_times.append(time.perf_counter())
        if frame.timestamp_ms is not None:
            self.frame_timestamps.append(float(frame.timestamp_ms))

        rgb = np.asarray(frame.color_bgr)
        depth = np.asarray(frame.depth_m, dtype=np.float32)
        rgb_tile = _fit_debug_tile(rgb, self.TILE_WIDTH, self.TILE_HEIGHT)
        depth_colored, valid_mask, depth_stats = _colorize_depth(
            depth,
            min_depth=self.args.debug_depth_min_m,
            max_depth=self.args.debug_depth_max_m,
        )
        depth_tile = _fit_debug_tile(depth_colored, self.TILE_WIDTH, self.TILE_HEIGHT)
        alignment_tile = _rgb_depth_alignment_tile(
            rgb,
            depth,
            min_depth=self.args.debug_depth_min_m,
            max_depth=self.args.debug_depth_max_m,
            width=self.TILE_WIDTH,
            height=self.TILE_HEIGHT,
            depth_colored=depth_colored,
            valid_mask=valid_mask,
        )

        xyz, colors = _rgbd_debug_points(
            frame,
            stride=self.args.debug_point_stride,
            max_points=self.args.debug_max_points,
            min_depth=self.args.debug_depth_min_m,
            max_depth=self.args.debug_depth_max_m,
        )
        aligned_xyz = xyz
        odometry_metrics = None
        reference_xyz = None
        if self.odometry is not None and len(xyz) > 0:
            aligned_xyz, odometry_metrics = self.odometry.update(xyz, colors)
            reference_xyz = self.odometry.reference_xyz
        geometry_tile = _point_cloud_debug_tile(
            xyz,
            colors,
            aligned_xyz=aligned_xyz if self.odometry is not None else None,
            reference_xyz=reference_xyz,
            min_depth=self.args.debug_depth_min_m,
            max_depth=self.args.debug_depth_max_m,
            width=self.TILE_WIDTH,
            height=self.TILE_HEIGHT,
        )

        _draw_debug_title(rgb_tile, "RGB (depth is aligned to this image)")
        _draw_debug_title(depth_tile, "Depth / valid mask")
        _draw_debug_title(alignment_tile, "RGB + depth discontinuity overlay")
        _draw_debug_title(
            geometry_tile,
            (
                "Point cloud geometry"
                if self.odometry is None
                else "Static-scene ICP: reference (green), aligned current (magenta)"
            ),
        )

        host_fps = _rate_from_seconds(self.wall_times)
        camera_fps = _rate_from_milliseconds(self.frame_timestamps)
        lines = [
            f"host={host_fps:.1f} FPS  camera={camera_fps:.1f} FPS",
            (
                f"valid_depth={100.0 * depth_stats['valid_ratio']:.1f}%  "
                f"p05/p50/p95={depth_stats['p05_m']:.3f}/"
                f"{depth_stats['p50_m']:.3f}/{depth_stats['p95_m']:.3f} m"
            ),
            f"rendered_points={len(xyz)}  recording={recording}",
        ]
        if imu_snapshot is not None:
            gyro = imu_snapshot.get("gyro")
            accel = imu_snapshot.get("accel")
            if gyro is not None:
                gyro_norm = float(np.linalg.norm(gyro["xyz"]))
                lines.append(f"gyro={gyro_norm:.3f} rad/s  sync_dt={gyro['rgb_timestamp_delta_ms']:+.2f} ms")
            if accel is not None:
                accel_norm = float(np.linalg.norm(accel["xyz"]))
                lines.append(
                    f"accel={accel_norm:.3f} m/s^2  sync_dt={accel['rgb_timestamp_delta_ms']:+.2f} ms"
                )
            lines.append("IMU is accel+gyro only; it is NOT a 6-DoF camera pose")
        else:
            lines.append("IMU unavailable/disabled; no camera pose is being recorded")
        if odometry_metrics is not None:
            lines.extend(
                [
                    (
                        f"RGB-D accepted={odometry_metrics['accepted']} "
                        f"anchor={odometry_metrics.get('anchor_accepted', False)} "
                        f"fitness={odometry_metrics['fitness']:.3f} "
                        f"rmse={odometry_metrics['inlier_rmse_m'] * 1000.0:.1f} mm"
                    ),
                    (
                        f"camera motion |t|={odometry_metrics['translation_m'] * 1000.0:.1f} mm "
                        f"angle={odometry_metrics['rotation_deg']:.2f} deg "
                        f"{odometry_metrics.get('estimator', '')}"
                    ),
                ]
            )
            if "stable_reference_ratio" in odometry_metrics:
                lines.append(
                    "temporal background "
                    f"stable_ref={100.0 * odometry_metrics['stable_reference_ratio']:.1f}% "
                    f"static_src={100.0 * odometry_metrics.get('static_source_ratio', 1.0):.1f}% "
                    f"active={odometry_metrics.get('temporal_static_filter_active', False)}"
                )
        lines.append("Debug keys: S save dashboard, R reset ICP reference, Q/Esc finish")
        _draw_debug_lines(depth_tile, lines)

        dashboard = np.vstack(
            [
                np.hstack([rgb_tile, depth_tile]),
                np.hstack([alignment_tile, geometry_tile]),
            ]
        )
        if (
            self.args.debug_save_every > 0
            and saved_index > 0
            and saved_index % self.args.debug_save_every == 0
            and saved_index != self.last_snapshot_index
        ):
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self.debug_dir / f"dashboard_{saved_index:06d}.png"
            if not cv2.imwrite(str(snapshot_path), dashboard):
                print(f"[warn] Failed to save debug dashboard: {snapshot_path}")
            self.last_snapshot_index = saved_index
        return dashboard


def _fit_debug_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source = np.asarray(image)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    source_height, source_width = source.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        source,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return tile


def _colorize_depth(
    depth_m: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = (depth[valid] - min_depth) / max(max_depth - min_depth, 1e-6)
        normalized[valid] = np.clip((1.0 - scaled) * 255.0, 0.0, 255.0).astype(np.uint8)
        values = depth[valid]
        p05, p50, p95 = np.percentile(values, [5.0, 50.0, 95.0])
    else:
        p05 = p50 = p95 = float("nan")
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    stats = {
        "valid_ratio": float(valid.mean()),
        "p05_m": float(p05),
        "p50_m": float(p50),
        "p95_m": float(p95),
    }
    return colored, valid, stats


def _rgb_depth_alignment_tile(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    width: int,
    height: int,
    depth_colored: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if depth_colored is None or valid_mask is None:
        depth_colored, valid, _ = _colorize_depth(
            depth_m,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    else:
        depth_colored = np.asarray(depth_colored, dtype=np.uint8)
        valid = np.asarray(valid_mask, dtype=bool)
    rgb = np.asarray(color_bgr, dtype=np.uint8)
    overlay = cv2.addWeighted(rgb, 0.68, depth_colored, 0.32, 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (np.asarray(depth_m)[valid] - min_depth) / max(max_depth - min_depth, 1e-6) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    edges = cv2.Canny(normalized, 25, 70)
    edges[~valid] = 0
    overlay[edges > 0] = np.array([0, 255, 0], dtype=np.uint8)
    return _fit_debug_tile(overlay, width, height)


def _rgbd_debug_points(
    frame,
    *,
    stride: int,
    max_points: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth_m, dtype=np.float32)
    color = np.asarray(frame.color_bgr, dtype=np.uint8)
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth[vv, uu]
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth) & (sampled_depth <= max_depth)
    z = sampled_depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    intrinsics = frame.intrinsics
    x = (u - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y = (v - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    xyz = np.column_stack([x, y, z])
    colors = color[vv[valid], uu[valid]]
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[indices]
        colors = colors[indices]
    return xyz, colors


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ matrix[:3, 3])
    return inverse


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _point_cloud_debug_tile(
    xyz: np.ndarray,
    colors: np.ndarray,
    *,
    aligned_xyz: np.ndarray | None,
    reference_xyz: np.ndarray | None,
    min_depth: float,
    max_depth: float,
    width: int,
    height: int,
) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    half_height = height // 2
    x_extent = max(0.3, max_depth * 0.75)
    y_extent = max(0.2, max_depth * 0.5)

    _render_projection(
        tile[:half_height],
        xyz,
        colors,
        horizontal_axis=0,
        vertical_axis=2,
        horizontal_range=(-x_extent, x_extent),
        vertical_range=(min_depth, max_depth),
        vertical_flip=True,
    )
    _render_projection(
        tile[half_height:],
        xyz,
        colors,
        horizontal_axis=0,
        vertical_axis=1,
        horizontal_range=(-x_extent, x_extent),
        vertical_range=(-y_extent, y_extent),
        vertical_flip=False,
    )
    if reference_xyz is not None:
        _render_projection(
            tile[:half_height],
            reference_xyz,
            np.tile(np.array([[0, 180, 0]], dtype=np.uint8), (len(reference_xyz), 1)),
            horizontal_axis=0,
            vertical_axis=2,
            horizontal_range=(-x_extent, x_extent),
            vertical_range=(min_depth, max_depth),
            vertical_flip=True,
        )
    if aligned_xyz is not None:
        _render_projection(
            tile[:half_height],
            aligned_xyz,
            np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(aligned_xyz), 1)),
            horizontal_axis=0,
            vertical_axis=2,
            horizontal_range=(-x_extent, x_extent),
            vertical_range=(min_depth, max_depth),
            vertical_flip=True,
        )
    cv2.line(tile, (0, half_height), (width - 1, half_height), (80, 80, 80), 1)
    cv2.putText(tile, "XZ", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(
        tile,
        "XY",
        (8, half_height + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
    )
    return tile


def _render_projection(
    canvas: np.ndarray,
    xyz: np.ndarray,
    colors: np.ndarray,
    *,
    horizontal_axis: int,
    vertical_axis: int,
    horizontal_range: tuple[float, float],
    vertical_range: tuple[float, float],
    vertical_flip: bool,
) -> None:
    points = np.asarray(xyz)
    if len(points) == 0:
        return
    color_values = np.asarray(colors, dtype=np.uint8)
    finite = np.isfinite(points[:, horizontal_axis]) & np.isfinite(points[:, vertical_axis])
    h_value = points[:, horizontal_axis]
    v_value = points[:, vertical_axis]
    valid = (
        finite
        & (h_value >= horizontal_range[0])
        & (h_value <= horizontal_range[1])
        & (v_value >= vertical_range[0])
        & (v_value <= vertical_range[1])
    )
    if not np.any(valid):
        return
    h_value = h_value[valid]
    v_value = v_value[valid]
    color_values = color_values[valid]
    px = (
        (h_value - horizontal_range[0])
        / max(horizontal_range[1] - horizontal_range[0], 1e-9)
        * (canvas.shape[1] - 1)
    ).astype(np.int32)
    py = (
        (v_value - vertical_range[0])
        / max(vertical_range[1] - vertical_range[0], 1e-9)
        * (canvas.shape[0] - 1)
    ).astype(np.int32)
    if vertical_flip:
        py = canvas.shape[0] - 1 - py
    canvas[py, px] = color_values


def _draw_debug_title(tile: np.ndarray, title: str) -> None:
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_debug_lines(tile: np.ndarray, lines: list[str]) -> None:
    line_height = 21
    box_height = min(tile.shape[0] - 34, line_height * len(lines) + 10)
    overlay = tile.copy()
    cv2.rectangle(overlay, (4, 32), (tile.shape[1] - 4, 32 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, tile, 0.32, 0.0, dst=tile)
    for line_index, line in enumerate(lines):
        y = 51 + line_index * line_height
        if y >= 32 + box_height:
            break
        cv2.putText(
            tile,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _rate_from_seconds(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = float(timestamps[-1] - timestamps[0])
    return 0.0 if duration <= 0.0 else (len(timestamps) - 1) / duration


def _rate_from_milliseconds(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration_s = float(timestamps[-1] - timestamps[0]) * 1e-3
    return 0.0 if duration_s <= 0.0 else (len(timestamps) - 1) / duration_s


def _imu_stationary_metrics(motion_recorder: RealSenseMotionRecorder | None) -> dict:
    if motion_recorder is None:
        return {
            "available": False,
            "semantics": "raw_imu_not_6dof_camera_pose",
        }

    output: dict = {
        "available": True,
        "semantics": "raw_imu_diagnostic_only_not_6dof_camera_pose",
    }
    for name in ("gyro", "accel"):
        samples = motion_recorder.history(name, max_samples=100000)
        if not samples:
            output[name] = {"sample_count": 0}
            continue
        norms = np.asarray(
            [np.linalg.norm(np.asarray(sample["xyz"], dtype=np.float64)) for sample in samples],
            dtype=np.float64,
        )
        output[name] = {
            "sample_count": int(len(norms)),
            "norm_mean": float(np.mean(norms)),
            "norm_p95": float(np.percentile(norms, 95.0)),
            "norm_max": float(np.max(norms)),
            "unit": str(samples[-1].get("unit", "unknown")),
        }
    return output


def _format_stationary_report(report: dict) -> str:
    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    if "translation" not in report or "rotation" not in report:
        return (
            f"[stationary pose check] {status}: {report.get('reason', 'report unavailable')} "
            f"(frames={report.get('frame_count', 0)})"
        )
    translation = report["translation"]
    rotation = report["rotation"]
    return (
        f"[stationary pose check] {status}: frames={report['frame_count']} "
        f"accepted={100.0 * report['accepted_ratio']:.1f}% "
        f"max_translation={1000.0 * translation['max_drift_m']:.2f} mm "
        f"(limit={1000.0 * translation['threshold_m']:.2f} mm), "
        f"max_rotation={rotation['max_drift_deg']:.3f} deg "
        f"(limit={rotation['threshold_deg']:.3f} deg)"
    )



_CAMERA_FPS_FIELD_NAMES = {
    "fps",
    "frame_rate",
    "framerate",
    "color_fps",
    "colour_fps",
    "rgb_fps",
    "depth_fps",
    "rgbd_fps",
    "stream_fps",
}
_CAMERA_FPS_EXCLUDED_TOKENS = ("imu", "gyro", "accel", "motion")


def _override_camera_stream_fps(camera_cfg, camera: str, fps: int) -> list[str]:
    """Override existing RGB/depth FPS fields in the selected BestMan camera config."""

    if int(fps) <= 0:
        raise ValueError("--fps must be a positive integer.")

    changed: list[str] = []
    candidates = _camera_candidates(camera_cfg, camera)
    for camera_name, candidate_cfg in candidates:
        changed.extend(
            _override_fps_fields_recursive(
                candidate_cfg,
                int(fps),
                prefix=f"Camera.{camera_name}",
            )
        )

    if not changed:
        camera_names = ", ".join(name for name, _cfg in candidates)
        raise ValueError(
            f"--fps could not find an RGB/depth FPS field under {camera_names}. "
            "Expected an existing field such as fps, color_fps, depth_fps, "
            "frame_rate, or a nested color/depth .fps field. "
            "Update the BestMan camera YAML or Camera_Realsense configuration."
        )
    return changed


def _override_fps_fields_recursive(node, fps: int, prefix: str) -> list[str]:
    if not isinstance(node, SimpleNamespace):
        return []

    changed: list[str] = []
    for field_name, value in vars(node).items():
        normalized = str(field_name).strip().lower().replace("-", "_")
        field_path = f"{prefix}.{field_name}"
        is_excluded = any(token in normalized for token in _CAMERA_FPS_EXCLUDED_TOKENS)
        is_fps_field = (
            normalized in _CAMERA_FPS_FIELD_NAMES
            or (
                normalized.endswith("_fps")
                and not is_excluded
            )
        )

        numeric_value = isinstance(value, (int, float)) and not isinstance(value, bool)
        numeric_text = isinstance(value, str) and value.strip().isdigit()
        if is_fps_field and (numeric_value or numeric_text):
            setattr(node, field_name, int(fps))
            changed.append(field_path)
            continue

        if isinstance(value, SimpleNamespace):
            changed.extend(_override_fps_fields_recursive(value, fps, field_path))
    return changed


def _camera_candidates(camera_cfg, camera: str) -> list[tuple[str, object]]:
    if camera == "overhead":
        candidates = []
        for name in ("L515", "D435I"):
            if hasattr(camera_cfg, name):
                candidates.append((name, getattr(camera_cfg, name)))
        return candidates
    if camera == "hand":
        candidates = []
        for name in ("D435I", "L515"):
            if hasattr(camera_cfg, name):
                candidates.append((name, getattr(camera_cfg, name)))
        return candidates
    if hasattr(camera_cfg, camera):
        return [(camera, getattr(camera_cfg, camera))]
    raise KeyError(f"Camera.{camera} not found in config")


def _open_camera(camera_cls, camera_cfg, camera: str):
    last_error: BaseException | None = None
    for name, cfg in _camera_candidates(camera_cfg, camera):
        try:
            return name, cfg, camera_cls(cfg)
        except SystemExit as exc:
            last_error = exc
            print(f"Failed to open {name}, trying next candidate if available.")
        except Exception as exc:
            last_error = exc
            print(f"Failed to open {name}: {exc}")
    raise RuntimeError(f"Could not open RealSense camera for request {camera}") from last_error


def _default_output_dir(camera: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return REAL_DATA_ROOT / "raw_rgbd" / f"{camera}_{timestamp}"


def _intrinsics_to_dict(intrinsics) -> dict:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "coeffs": [float(v) for v in getattr(intrinsics, "coeffs", ())],
        "model": getattr(intrinsics, "model", None),
    }


def _write_frame_record(
    output_dir: Path,
    args: argparse.Namespace,
    frame,
    index: int,
    frames_file,
    color_video,
    color_video_rel: Path,
    imu_snapshot: dict | None = None,
) -> None:
    record = {
        "index": index,
        "timestamp_ms": frame.timestamp_ms,
        "frame_number": frame.frame_number,
        "intrinsics": _intrinsics_to_dict(frame.intrinsics),
    }
    if imu_snapshot is not None:
        record["imu_nearest"] = imu_snapshot

    if args.storage == "legacy" or args.save_color_png:
        color_rel = Path("color") / f"{index:06d}.png"
        _save_color_png(output_dir / color_rel, frame.color_bgr)
        if args.storage == "legacy":
            record["color_path"] = color_rel.as_posix()
        else:
            record["color_png_path"] = color_rel.as_posix()

    if args.storage == "compressed":
        color_jpg_rel = Path("color_jpg") / f"{index:06d}.jpg"
        _save_color_jpeg(output_dir / color_jpg_rel, frame.color_bgr, args.jpeg_quality)
        record["color_path"] = color_jpg_rel.as_posix()

    if args.storage == "video":
        if color_video is None:
            raise RuntimeError("Color video writer was not initialized.")
        color_video.write(frame.color_bgr)
        record["color_video_path"] = color_video_rel.as_posix()
        record["color_video_frame_index"] = index

    if args.storage == "legacy" or args.save_depth_npy:
        depth_rel = Path("depth_m") / f"{index:06d}.npy"
        np.save(output_dir / depth_rel, np.asarray(frame.depth_m, dtype=np.float32))
        record["depth_m_path"] = depth_rel.as_posix()

    if args.storage in ("compressed", "video") or args.save_depth_png:
        depth_png_rel = Path("depth_png") / f"{index:06d}.png"
        _save_depth_png_mm(output_dir / depth_png_rel, frame.depth_m)
        record["depth_png_path"] = depth_png_rel.as_posix()

    frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    frames_file.flush()


def _space_event(event: str, record_index: int, timestamp_ms, frame_number) -> dict:
    return {
        "event": event,
        "record_index": int(record_index),
        "timestamp_ms": None if timestamp_ms is None else float(timestamp_ms),
        "frame_number": None if frame_number is None else int(frame_number),
        "wall_time_s": time.time(),
    }


def _write_segments(
    output_dir: Path,
    segments: list[dict],
    space_events: list[dict],
    metadata: dict | None,
) -> None:
    segments_arg = ",".join(f"{segment['start']}:{segment['end']}" for segment in segments)
    (output_dir / "segments.txt").write_text(segments_arg + "\n", encoding="utf-8")
    payload = {
        "segments": segments,
        "segments_arg": segments_arg,
        "space_events": space_events,
        "build_command_hint": (
            (f"python build_humanhand_hdf5_dataset.py --input {output_dir} --segments {segments_arg}")
            if segments_arg
            else None
        ),
    }
    (output_dir / "segments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if metadata is not None:
        updated_metadata = dict(metadata)
        updated_metadata["space_toggle_segments"] = segments
        updated_metadata["space_toggle_segments_arg"] = segments_arg
        updated_metadata["space_toggle_events"] = space_events
        (output_dir / "metadata.json").write_text(
            json.dumps(updated_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if segments_arg:
        print(f"Saved segment ranges for --segments: {segments_arg}")
    else:
        print("No non-empty recording segments were saved.")


def _color_format(args) -> str:
    if args.storage == "legacy":
        return "bgr_png"
    if args.storage == "compressed":
        return "bgr_jpeg"
    if args.storage == "video":
        return "bgr_mp4"
    raise ValueError(args.storage)


def _depth_format(args) -> str:
    if args.storage == "legacy":
        return "float32_npy_meter"
    return "uint16_png_millimeter"


def _open_color_video_writer(path: Path, image_shape: tuple[int, ...], fps: float, codec: str):
    height, width = int(image_shape[0]), int(image_shape[1])
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open color video writer: {path}")
    return writer


def _save_color_png(path: Path, color_bgr: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), color_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


def _save_color_jpeg(path: Path, color_bgr: np.ndarray, quality: int) -> None:
    quality = int(np.clip(quality, 1, 100))
    ok = cv2.imwrite(str(path), color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


def _save_depth_png_mm(path: Path, depth_m: np.ndarray) -> None:
    depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    ok = cv2.imwrite(str(path), depth_mm)
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


def _load_recorded_frame_records(output_dir: Path) -> list[dict]:
    frames_path = output_dir / "frames.jsonl"
    if not frames_path.is_file():
        raise FileNotFoundError(frames_path)
    records = [
        json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No saved RGB-D frames were found in {frames_path}.")
    indices = [int(record.get("index", index)) for index, record in enumerate(records)]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate frame indices in {frames_path}.")
    return records


def _capture_rate_summary(
    output_dir: Path,
    saved_wall_times: list[float],
    *,
    debug_pipeline_is_synchronous: bool,
) -> dict[str, float | int | bool]:
    frame_records = _load_recorded_frame_records(output_dir)
    frame_count = len(frame_records)
    host_saved_fps = 0.0
    if len(saved_wall_times) >= 2:
        elapsed_s = float(saved_wall_times[-1] - saved_wall_times[0])
        if elapsed_s > 0.0:
            host_saved_fps = float((len(saved_wall_times) - 1) / elapsed_s)
    timestamps = [
        float(record["timestamp_ms"]) for record in frame_records if record.get("timestamp_ms") is not None
    ]
    camera_timestamp_fps = 0.0
    if len(timestamps) >= 2:
        elapsed_ms = float(timestamps[-1] - timestamps[0])
        if elapsed_ms > 0.0:
            camera_timestamp_fps = float((len(timestamps) - 1) * 1000.0 / elapsed_ms)
    return {
        "frame_count": int(frame_count),
        "host_saved_fps": host_saved_fps,
        "camera_timestamp_fps": camera_timestamp_fps,
        "debug_pipeline_is_synchronous": bool(debug_pipeline_is_synchronous),
    }


def _trajectory_output_path(output_dir: Path, requested: str) -> Path:
    path = Path(requested).expanduser()
    return path.resolve() if path.is_absolute() else (output_dir / path).resolve()


def _load_external_pose_records(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    pose_by_index: dict[int, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        raw_index = record.get("record_index", record.get("index"))
        if raw_index is None:
            raise KeyError(f"{path}:{line_number} has no record_index.")
        index = int(raw_index)
        if index in pose_by_index:
            raise ValueError(f"Duplicate pose for record_index={index} in {path}.")
        pose_by_index[index] = {
            **record,
            "record_index": index,
            "camera_to_tracking": matrix_from_json_record(record).tolist(),
        }
    return pose_by_index


def _static_camera_trajectory(frame_records: list[dict]) -> list[dict]:
    identity = np.eye(4, dtype=np.float64).tolist()
    return [
        {
            "record_index": int(frame_record.get("index", index)),
            "timestamp_ms": frame_record.get("timestamp_ms"),
            "camera_to_tracking": identity,
            "tracking_source": "static_identity_camera",
            "valid": True,
            "transform_direction": "camera_to_tracking",
        }
        for index, frame_record in enumerate(frame_records)
    ]


def _external_camera_trajectory(
    frame_records: list[dict],
    pose_path: Path,
    *,
    max_sync_error_ms: float,
) -> list[dict]:
    pose_by_index = _load_external_pose_records(pose_path)
    output = []
    missing = []
    for fallback_index, frame_record in enumerate(frame_records):
        index = int(frame_record.get("index", fallback_index))
        pose_record = pose_by_index.get(index)
        if pose_record is None:
            missing.append(index)
            continue
        if not bool(pose_record.get("valid", True)):
            raise ValueError(f"External camera pose record_index={index} is marked invalid.")
        output_record = dict(pose_record)
        rgb_timestamp_ms = frame_record.get("timestamp_ms")
        pose_timestamp_ms = pose_record.get("timestamp_ms")
        if rgb_timestamp_ms is not None and pose_timestamp_ms is not None:
            sync_error_ms = float(pose_timestamp_ms) - float(rgb_timestamp_ms)
            if abs(sync_error_ms) > float(max_sync_error_ms):
                raise ValueError(
                    f"External pose record_index={index} is out of sync by "
                    f"{sync_error_ms:+.3f} ms; limit={max_sync_error_ms:.3f} ms."
                )
            output_record["rgb_pose_sync_error_ms"] = sync_error_ms
        output_record.setdefault("tracking_source", "external_slam_vio")
        output_record["transform_direction"] = "camera_to_tracking"
        output.append(output_record)
    if missing:
        raise RuntimeError(
            f"External camera trajectory is missing {len(missing)} saved RGB-D frames; "
            f"first missing indices: {missing[:10]}."
        )
    return output


def _postprocess_rgbd_camera_trajectory(
    output_dir: Path,
    frame_records: list[dict],
    metadata: dict,
    args: argparse.Namespace,
) -> list[dict]:
    odometry = RGBDOdometryDebugger(
        args.debug_odometry_voxel_m,
        args.debug_odometry_max_correspondence_m,
        args.debug_odometry_robust_kernel_m,
        args.debug_odometry_anchor_min_fitness,
        args.debug_odometry_anchor_max_correction_m,
        args.debug_odometry_anchor_max_correction_deg,
        args.debug_odometry_bootstrap_frames,
    )
    video_captures: dict[str, object] = {}
    trajectory = []
    start_s = time.perf_counter()
    try:
        for frame_offset, frame_record in enumerate(frame_records):
            color_bgr, depth_m, intrinsics = _read_recorded_rgbd(
                output_dir,
                frame_record,
                metadata,
                video_captures,
            )
            frame = SimpleNamespace(
                color_bgr=color_bgr,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
            xyz, colors = _rgbd_debug_points(
                frame,
                stride=int(args.debug_point_stride),
                max_points=int(args.debug_max_points),
                min_depth=float(args.debug_depth_min_m),
                max_depth=float(args.debug_depth_max_m),
            )
            _aligned_xyz, metrics = odometry.update(xyz, colors)
            trajectory.append(
                {
                    "record_index": int(frame_record.get("index", frame_offset)),
                    "timestamp_ms": frame_record.get("timestamp_ms"),
                    "camera_to_tracking": odometry.camera_to_world.tolist(),
                    "tracking_source": "postprocess_dynamic_robust_rgbd",
                    "valid": bool(metrics.get("accepted", False)),
                    "transform_direction": "camera_to_tracking",
                    "training_ground_truth": False,
                    "metrics": metrics,
                }
            )
            if (frame_offset + 1) % 30 == 0 or frame_offset + 1 == len(frame_records):
                elapsed_s = max(time.perf_counter() - start_s, 1e-9)
                print(
                    "Post-processing camera trajectory: "
                    f"{frame_offset + 1}/{len(frame_records)} "
                    f"({(frame_offset + 1) / elapsed_s:.2f} FPS)"
                )
    finally:
        for capture in video_captures.values():
            capture.release()
    invalid = [int(record["record_index"]) for record in trajectory if not bool(record["valid"])]
    if invalid:
        raise RuntimeError(
            "Post-process RGB-D odometry failed for "
            f"{len(invalid)} frames; first invalid indices: {invalid[:10]}. "
            "Use a metric external SLAM/VIO trajectory for training data."
        )
    return trajectory


def _write_camera_trajectory(
    output_dir: Path,
    output_path: Path,
    frame_records: list[dict],
    trajectory: list[dict],
) -> None:
    if len(frame_records) != len(trajectory):
        raise ValueError(f"Frame/trajectory count mismatch: {len(frame_records)} vs {len(trajectory)}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in trajectory),
        encoding="utf-8",
    )
    enriched_frames = []
    for frame_record, pose_record in zip(frame_records, trajectory, strict=True):
        enriched = dict(frame_record)
        enriched.update(
            {
                "camera_to_tracking": pose_record["camera_to_tracking"],
                "camera_pose_source": pose_record["tracking_source"],
                "camera_pose_timestamp_ms": pose_record.get("timestamp_ms"),
                "camera_pose_valid": bool(pose_record.get("valid", True)),
            }
        )
        if "rgb_pose_sync_error_ms" in pose_record:
            enriched["camera_pose_sync_error_ms"] = pose_record["rgb_pose_sync_error_ms"]
        enriched_frames.append(enriched)
    (output_dir / "frames.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in enriched_frames),
        encoding="utf-8",
    )


def _finalize_camera_trajectory(
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path | None, dict]:
    mode = str(args.camera_trajectory_mode)
    if mode == "none":
        return None, {
            "camera_pose_recorded": False,
            "camera_trajectory_mode": mode,
            "camera_pose_path": None,
        }
    frame_records = _load_recorded_frame_records(output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if mode == "static":
        trajectory = _static_camera_trajectory(frame_records)
        training_ground_truth = True
        semantics = "constant_identity_pose_fixed_camera"
    elif mode == "external":
        if not args.external_camera_pose_jsonl:
            raise ValueError("--camera-trajectory-mode=external requires --external-camera-pose-jsonl.")
        trajectory = _external_camera_trajectory(
            frame_records,
            Path(args.external_camera_pose_jsonl).expanduser().resolve(),
            max_sync_error_ms=float(args.camera_trajectory_max_sync_error_ms),
        )
        training_ground_truth = True
        semantics = "external_metric_slam_vio_camera_to_tracking"
    elif mode == "rgbd_odometry":
        trajectory = _postprocess_rgbd_camera_trajectory(output_dir, frame_records, metadata, args)
        training_ground_truth = False
        semantics = "postprocess_rgbd_diagnostic_camera_to_first_frame"
    else:
        raise ValueError(f"Unsupported camera trajectory mode: {mode!r}.")

    output_path = _trajectory_output_path(output_dir, str(args.camera_trajectory_output_jsonl))
    _write_camera_trajectory(output_dir, output_path, frame_records, trajectory)
    print(f"Saved {mode} camera trajectory: {output_path}")
    return output_path, {
        "camera_pose_recorded": True,
        "camera_trajectory_mode": mode,
        "camera_pose_path": str(output_path),
        "camera_pose_transform_direction": "camera_to_tracking",
        "camera_pose_semantics": semantics,
        "camera_pose_training_ground_truth": training_ground_truth,
        "camera_pose_frame_count": len(trajectory),
    }


def _resolve_aligned_playback_poses(
    frame_records: list[dict],
    args: argparse.Namespace,
    diagnostic_saved_poses: dict[int, np.ndarray],
) -> tuple[np.ndarray, str]:
    use_external = args.aligned_point_cloud_pose_source == "external" or (
        args.aligned_point_cloud_pose_source == "auto" and bool(args.aligned_point_cloud_pose_jsonl)
    )
    if use_external:
        pose_path = Path(args.aligned_point_cloud_pose_jsonl).expanduser().resolve()
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        pose_by_index: dict[int, np.ndarray] = {}
        for line_number, line in enumerate(pose_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_index = record.get("record_index", record.get("index"))
            if raw_index is None:
                raise KeyError(f"{pose_path}:{line_number} has no record_index.")
            index = int(raw_index)
            if index in pose_by_index:
                raise ValueError(f"Duplicate pose for record_index={index} in {pose_path}.")
            if not bool(record.get("valid", True)):
                raise ValueError(f"Invalid SLAM/VIO pose for record_index={index} in {pose_path}.")
            pose_by_index[index] = matrix_from_json_record(record)
        missing = [
            int(record.get("index", index))
            for index, record in enumerate(frame_records)
            if int(record.get("index", index)) not in pose_by_index
        ]
        if missing:
            raise RuntimeError(
                f"SLAM/VIO pose file is missing {len(missing)} saved RGB-D frames; "
                f"first missing indices: {missing[:10]}."
            )
        camera_to_tracking = np.stack(
            [pose_by_index[int(record.get("index", index))] for index, record in enumerate(frame_records)]
        )
        source = f"external:{pose_path}"
    else:
        missing = [
            int(record.get("index", index))
            for index, record in enumerate(frame_records)
            if int(record.get("index", index)) not in diagnostic_saved_poses
        ]
        if missing:
            raise RuntimeError(
                "Diagnostic RGB-D odometry did not produce a pose for every saved frame; "
                f"first missing indices: {missing[:10]}."
            )
        camera_to_tracking = np.stack(
            [
                diagnostic_saved_poses[int(record.get("index", index))]
                for index, record in enumerate(frame_records)
            ]
        )
        source = "diagnostic_dynamic_robust_rgbd_not_training_ground_truth"

    # The source trajectory may use any global tracking frame. Rebase it so
    # the first saved RGB-D camera frame is exactly the visualization world.
    camera_to_first_frame = camera_to_model_world_transforms(camera_to_tracking)
    return camera_to_first_frame, source


def _read_recorded_rgbd(
    output_dir: Path,
    frame_record: dict,
    metadata: dict,
    video_captures: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    color_path = frame_record.get("color_path") or frame_record.get("color_png_path")
    if color_path is not None:
        color_bgr = cv2.imread(str(output_dir / color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(output_dir / color_path)
    else:
        video_path_value = frame_record.get("color_video_path") or metadata.get("color_video_path")
        if video_path_value is None:
            raise KeyError(f"Frame {frame_record.get('index')} has no readable color source.")
        video_path = output_dir / video_path_value
        capture_key = str(video_path)
        capture = video_captures.get(capture_key)
        if capture is None:
            capture = cv2.VideoCapture(capture_key)
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open color video: {video_path}")
            video_captures[capture_key] = capture
        frame_index = int(frame_record.get("color_video_frame_index", frame_record.get("index", 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, color_bgr = capture.read()
        if not ok or color_bgr is None:
            raise RuntimeError(f"Failed to read color frame {frame_index} from {video_path}.")

    depth_m_path = frame_record.get("depth_m_path")
    depth_png_path = frame_record.get("depth_png_path")
    if depth_m_path is not None:
        depth_m = np.load(output_dir / depth_m_path).astype(np.float32)
    elif depth_png_path is not None:
        depth_mm = cv2.imread(str(output_dir / depth_png_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise FileNotFoundError(output_dir / depth_png_path)
        depth_m = np.asarray(depth_mm, dtype=np.float32) / 1000.0
    else:
        raise KeyError(f"Frame {frame_record.get('index')} has no readable depth source.")

    intrinsics_data = frame_record.get("intrinsics") or metadata.get("intrinsics")
    if intrinsics_data is None:
        raise KeyError(f"Frame {frame_record.get('index')} has no intrinsics.")
    intrinsics = SimpleNamespace(
        width=int(intrinsics_data["width"]),
        height=int(intrinsics_data["height"]),
        fx=float(intrinsics_data["fx"]),
        fy=float(intrinsics_data["fy"]),
        ppx=float(intrinsics_data["ppx"]),
        ppy=float(intrinsics_data["ppy"]),
    )
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError(
            f"Aligned RGB/depth shape mismatch for frame {frame_record.get('index')}: "
            f"color={color_bgr.shape[:2]}, depth={depth_m.shape}."
        )
    return color_bgr, depth_m, intrinsics


def _camera_axes_lines(
    camera_to_world: np.ndarray, axis_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    points = np.vstack(
        [
            origin,
            origin + rotation[:, 0] * axis_size,
            origin + rotation[:, 1] * axis_size,
            origin + rotation[:, 2] * axis_size,
        ]
    )
    lines = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.asarray(
        [[1.0, 0.15, 0.15], [0.15, 1.0, 0.15], [0.15, 0.35, 1.0]],
        dtype=np.float64,
    )
    return points, lines, colors


def _play_aligned_point_cloud_sequence(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    diagnostic_saved_poses: dict[int, np.ndarray],
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("--visualize-aligned-point-cloud requires open3d.") from exc

    frame_records = _load_recorded_frame_records(output_dir)
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    poses, pose_source = _resolve_aligned_playback_poses(frame_records, args, diagnostic_saved_poses)
    pose_output_path = None
    if not args.playback_only:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        pose_output_path = debug_dir / "aligned_point_cloud_poses.jsonl"
        with pose_output_path.open("w", encoding="utf-8") as pose_file:
            for frame_record, pose in zip(frame_records, poses, strict=True):
                pose_file.write(
                    json.dumps(
                        {
                            "record_index": int(frame_record["index"]),
                            "timestamp_ms": frame_record.get("timestamp_ms"),
                            "camera_to_first_frame": pose.tolist(),
                            "transform_direction": "current_camera_to_first_saved_camera",
                            "pose_source": pose_source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        manifest_path = debug_dir / "aligned_point_cloud_playback.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "frame_count": len(frame_records),
                    "pose_source": pose_source,
                    "world_definition": "first_saved_rgbd_camera_frame",
                    "pose_jsonl": str(pose_output_path),
                    "axis_size_m": float(args.aligned_point_cloud_axis_size_m),
                    "point_stride": int(args.aligned_point_cloud_point_stride),
                    "max_points": int(args.aligned_point_cloud_max_points),
                    "depth_range_m": [
                        float(args.debug_depth_min_m),
                        float(args.debug_depth_max_m),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    video_captures: dict[str, object] = {}

    def load_aligned_cloud(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        color_bgr, depth_m, intrinsics = _read_recorded_rgbd(
            output_dir,
            frame_records[frame_index],
            metadata,
            video_captures,
        )
        frame = SimpleNamespace(
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=intrinsics,
        )
        xyz, colors_bgr = _rgbd_debug_points(
            frame,
            stride=int(args.aligned_point_cloud_point_stride),
            max_points=int(args.aligned_point_cloud_max_points),
            min_depth=float(args.debug_depth_min_m),
            max_depth=float(args.debug_depth_max_m),
        )
        return _transform_points(xyz, poses[frame_index]), colors_bgr

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    if not visualizer.create_window(
        window_name="RGB-D clouds aligned to first saved frame",
        width=1280,
        height=800,
    ):
        raise RuntimeError(
            "Open3D could not create a visualization window. Run with a valid DISPLAY or use X forwarding."
        )

    axis_size = float(args.aligned_point_cloud_axis_size_m)
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0])
    origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.006, 0.045 * axis_size))
    origin_marker.paint_uniform_color([1.0, 0.85, 0.0])

    first_xyz, first_colors_bgr = load_aligned_cloud(0)
    reference_cloud = o3d.geometry.PointCloud()
    reference_cloud.points = o3d.utility.Vector3dVector(first_xyz)
    reference_cloud.colors = o3d.utility.Vector3dVector(np.full((len(first_xyz), 3), 0.22, dtype=np.float64))
    current_cloud = o3d.geometry.PointCloud()
    current_cloud.points = o3d.utility.Vector3dVector(first_xyz)
    current_cloud.colors = o3d.utility.Vector3dVector(
        np.clip(first_colors_bgr[:, ::-1].astype(np.float64) / 255.0, 0.0, 1.0)
    )
    camera_axes = o3d.geometry.LineSet()
    initial_axis_points, initial_axis_lines, initial_axis_colors = _camera_axes_lines(
        poses[0], 0.45 * axis_size
    )
    camera_axes.points = o3d.utility.Vector3dVector(initial_axis_points)
    camera_axes.lines = o3d.utility.Vector2iVector(initial_axis_lines)
    camera_axes.colors = o3d.utility.Vector3dVector(initial_axis_colors)
    trajectory = o3d.geometry.LineSet()
    camera_centers = poses[:, :3, 3]
    trajectory.points = o3d.utility.Vector3dVector(np.repeat(camera_centers[:1], 2, axis=0))
    trajectory.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    trajectory.colors = o3d.utility.Vector3dVector(np.asarray([[1.0, 0.65, 0.05]], dtype=np.float64))

    visualizer.add_geometry(reference_cloud)
    visualizer.add_geometry(current_cloud)
    visualizer.add_geometry(world_axes)
    visualizer.add_geometry(origin_marker)
    visualizer.add_geometry(camera_axes)
    visualizer.add_geometry(trajectory)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([0.015, 0.015, 0.02])
    render_option.point_size = 2.0

    state = {"paused": False, "quit": False, "restart": False}

    def toggle_pause(_visualizer) -> bool:
        state["paused"] = not state["paused"]
        return False

    def restart(_visualizer) -> bool:
        state["restart"] = True
        state["paused"] = False
        return False

    def quit_visualization(_visualizer) -> bool:
        state["quit"] = True
        return False

    visualizer.register_key_callback(ord(" "), toggle_pause)
    visualizer.register_key_callback(ord("R"), restart)
    visualizer.register_key_callback(ord("Q"), quit_visualization)

    def update_geometry(frame_index: int) -> None:
        aligned_xyz, colors_bgr = load_aligned_cloud(frame_index)
        current_cloud.points = o3d.utility.Vector3dVector(aligned_xyz)
        current_cloud.colors = o3d.utility.Vector3dVector(
            np.clip(colors_bgr[:, ::-1].astype(np.float64) / 255.0, 0.0, 1.0)
        )
        axis_points, axis_lines, axis_colors = _camera_axes_lines(poses[frame_index], 0.45 * axis_size)
        camera_axes.points = o3d.utility.Vector3dVector(axis_points)
        camera_axes.lines = o3d.utility.Vector2iVector(axis_lines)
        camera_axes.colors = o3d.utility.Vector3dVector(axis_colors)
        visible_centers = camera_centers[: frame_index + 1]
        if len(visible_centers) < 2:
            trajectory_points = np.repeat(visible_centers[:1], 2, axis=0)
            trajectory_lines = np.asarray([[0, 1]], dtype=np.int32)
        else:
            trajectory_points = visible_centers
            trajectory_lines = np.column_stack(
                [
                    np.arange(len(visible_centers) - 1, dtype=np.int32),
                    np.arange(1, len(visible_centers), dtype=np.int32),
                ]
            )
        trajectory.points = o3d.utility.Vector3dVector(trajectory_points)
        trajectory.lines = o3d.utility.Vector2iVector(trajectory_lines)
        trajectory.colors = o3d.utility.Vector3dVector(
            np.tile(
                np.asarray([[1.0, 0.65, 0.05]], dtype=np.float64),
                (len(trajectory_lines), 1),
            )
        )
        visualizer.update_geometry(current_cloud)
        visualizer.update_geometry(camera_axes)
        visualizer.update_geometry(trajectory)

    print(
        f"Aligned 3D playback: {len(frame_records)} frames, source={pose_source}\n"
        "Open3D controls: Space pause/resume, R restart, Q close; "
        "mouse rotates/zooms the fixed first-frame world."
    )
    frame_index = 0
    update_geometry(frame_index)
    frame_index += 1
    next_frame_time = time.perf_counter() + 1.0 / float(args.aligned_point_cloud_fps)
    try:
        while not state["quit"]:
            if not visualizer.poll_events():
                break
            if state["restart"]:
                frame_index = 0
                state["restart"] = False
                next_frame_time = time.perf_counter()
            now = time.perf_counter()
            if not state["paused"] and now >= next_frame_time:
                if frame_index >= len(frame_records):
                    if args.aligned_point_cloud_loop:
                        frame_index = 0
                    elif not args.aligned_point_cloud_hold_final:
                        state["quit"] = True
                        continue
                    else:
                        state["paused"] = True
                        print("Aligned playback reached the final frame; press R to replay or Q to close.")
                        visualizer.update_renderer()
                        continue
                update_geometry(frame_index)
                frame_index += 1
                next_frame_time = now + 1.0 / float(args.aligned_point_cloud_fps)
            visualizer.update_renderer()
            time.sleep(0.002)
    finally:
        for capture in video_captures.values():
            capture.release()
        visualizer.destroy_window()
    if pose_output_path is not None:
        print(f"Aligned playback pose audit: {pose_output_path}")


if __name__ == "__main__":
    main()
