#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R, Slerp

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import REAL_DATA_ROOT
    from .camera_motion_utils import (
        matrix_from_json_record,
        validate_transform_sequence,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT
    from real_setting.camera_motion_utils import (
        matrix_from_json_record,
        validate_transform_sequence,
    )


HANDPOSE_ROOT = Path("/home/liusong/ProgramFiles/HandPoseExtraction")
RAW_RIGID_HANDPOSE_PIPELINE_VERSION = "wilor_mano_mesh_rgbd_rigid_icp_v6"
REQUIRED_HANDPOSE_PIPELINE_VERSION = "wilor_mano_mesh_rgbd_rigid_icp_temporal_v7"
DEFAULT_POINTS_NUM = 640 * 480
_VIDEO_CAPTURE_CACHE = {}
_SEGMENT_WORKER_CONTEXT = None

HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

HUMANHAND_L515_CAMERA_TO_WORLD = np.asarray(
    [
        [-0.05659152, 0.80283684, -0.59350569, 0.84141181],
        [0.99720040, 0.01635126, -0.07297025, -0.00164086],
        [-0.04887985, -0.59597368, -0.80151482, 0.66857453],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple[float, ...] = ()
    model: str | None = None


@dataclass(frozen=True)
class Segment:
    start: int
    end: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build BestMan HumanHand HDF5 episodes from recorded RGB-D frames "
            "and offline WiLoR gripper predictions."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="RGB-D directory produced by record_bestman_rgbd.py.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="WiLoR JSONL. Defaults to <input>/handpose_wilor.jsonl.",
    )
    parser.add_argument(
        "--camera-pose-jsonl",
        default=None,
        help=(
            "Optional full-6DoF VIO/SLAM camera poses. Accepts either one JSONL file or an ORB-SLAM3 "
            "directory containing segment_*/dataset/camera_pose_orbslam3.jsonl. All records are merged "
            "once by record_index before interactive slicing. Each exported episode is then independently "
            "aligned to its own first camera frame when --align-to-episode-first is enabled."
        ),
    )
    parser.add_argument(
        "--require-camera-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject every exported segment that lacks a valid full-SE(3) camera pose for any frame.",
    )
    parser.add_argument(
        "--camera-pose-max-sync-error-ms",
        type=float,
        default=20.0,
        help=(
            "Maximum absolute camera-pose/RGB-D timestamp difference when both timestamps are present. "
            "Set 0 to disable this validation; record_index matching remains mandatory."
        ),
    )
    parser.add_argument(
        "--camera-reference-mode",
        choices=("episode_first", "canonical"),
        default="episode_first",
        help=(
            "How later conversion defines model world. episode_first uses each segment's "
            "first camera frame; canonical uses one fixed camera pose in a persistent "
            "tracking/base frame."
        ),
    )
    parser.add_argument(
        "--canonical-camera-to-tracking-matrix",
        nargs=16,
        type=float,
        default=None,
        metavar="M",
        help=(
            "Row-major T_tracking<-canonicalCamera in meters, stored in HDF5 for "
            "cross-episode canonical fixed-view alignment."
        ),
    )
    parser.add_argument(
        "--align-to-episode-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "During HDF5 export, transform cloud_rgb, gripper pose/orientation, 3D hand keypoints, "
            "camera_tracking_pose, and (by default) RGB images into the first camera frame of each "
            "exported episode. The first camera frame becomes the episode world frame. Requires a "
            "complete --camera-pose-jsonl trajectory, --camera-reference-mode episode_first, and "
            "--pose-frame camera."
        ),
    )
    parser.add_argument(
        "--reproject-rgb-to-episode-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --align-to-episode-first, replace each stored RGB frame by a z-buffered rendering "
            "of that frame's complete valid RGB-D point cloud from the episode-first camera view. "
            "All valid depth pixels are used; --max-points affects cloud_rgb only. Disable with "
            "--no-reproject-rgb-to-episode-first to retain the legacy unwarped RGB images."
        ),
    )
    parser.add_argument(
        "--rgb-reproject-workers",
        type=int,
        default=0,
        help=(
            "Parallel frame workers used by episode-first RGB reprojection. "
            "0 selects a conservative automatic value, 1 disables frame-level parallelism. "
            "The implementation uses a bounded NumPy thread pool so full-resolution RGB-D "
            "frames are not copied between processes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REAL_DATA_ROOT / "hdf5_raw"),
        help="Directory where episode_*.hdf5 files will be written.",
    )
    parser.add_argument("--task", default="humanhand_offline")
    parser.add_argument("--camera-names", default="overhead,hand")
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run HandPoseExtraction offline WiLoR first.",
    )
    parser.add_argument("--handpose-root", default=str(HANDPOSE_ROOT))
    parser.add_argument("--wilor-repo", default=str(HANDPOSE_ROOT / "external/WiLoR"))
    parser.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
    parser.add_argument("--model-cfg", default="pretrained_models/model_config.yaml")
    parser.add_argument("--detector", default="pretrained_models/detector.pt")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force-handedness", choices=("left", "right"), default=None)
    parser.add_argument(
        "--fusion-mode",
        choices=("model-depth", "keypoint-depth"),
        default="model-depth",
    )
    parser.add_argument("--gripper-x-offset-cm", type=float, default=1.5)
    parser.add_argument("--gripper-z-offset-cm", type=float, default=3.5)
    parser.add_argument("--hand-depth-window", type=int, default=5)
    parser.add_argument("--hand-min-depth-m", type=float, default=0.30)
    parser.add_argument("--hand-max-depth-m", type=float, default=3.0)
    parser.add_argument("--hand-min-valid-depth-points", type=int, default=6)
    parser.add_argument(
        "--hand-depth-knn-backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Nearest-neighbor backend for MANO/RGB-D alignment.",
    )
    parser.add_argument(
        "--hand-depth-rigid-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine the complete WiLoR MANO hand with one conservative RGB-D SE(3) "
            "correction before the unchanged virtual-gripper fitting stage."
        ),
    )
    parser.add_argument(
        "--hand-rigid-temporal-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Interpolate short rejected RGB-D rigid-pose gaps and smooth only the "
            "global MANO correction before the unchanged gripper fitting output."
        ),
    )
    parser.add_argument("--hand-rigid-temporal-filter-window", type=int, default=21)
    parser.add_argument("--hand-rigid-temporal-filter-order", type=int, default=2)
    parser.add_argument("--hand-rigid-temporal-max-gap", type=int, default=10)
    parser.add_argument(
        "--ego-trajectory-filter",
        choices=("none", "se3_lowpass"),
        default="none",
        help=(
            "Optional per-episode offline filter applied after camera-motion compensation. "
            "se3_lowpass uses a zero-phase Butterworth filter for translation and quaternion "
            "orientation, so it suppresses high-frequency hand-pose noise without causal lag."
        ),
    )
    parser.add_argument(
        "--ego-trajectory-filter-cutoff-hz",
        type=float,
        default=4.0,
        help="Low-pass cutoff for filtered gripper translation/orientation (default: 4 Hz).",
    )
    parser.add_argument(
        "--ego-trajectory-filter-order",
        type=int,
        default=3,
        help="Butterworth order used by --ego-trajectory-filter=se3_lowpass (default: 3).",
    )
    parser.add_argument(
        "--ego-trajectory-max-angular-speed-deg-s",
        type=float,
        default=900.0,
        help=(
            "Treat a larger frame-to-frame angular speed as an orientation-frame reset and "
            "carry a constant correction into following frames before low-pass filtering. "
            "900 deg/s is above normal demonstration motion but catches WiLoR anatomical-axis "
            "flips; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--ego-trajectory-parallel-jaw-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before smoothing, choose the temporally continuous representative of the parallel-jaw "
            "gripper's 180-degree rotation symmetry around its local approach axis."
        ),
    )
    parser.add_argument(
        "--ego-trajectory-filter-preserve-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a smooth correction so the filtered episode retains its original start/end poses.",
    )
    parser.add_argument(
        "--allow-legacy-handpose-jsonl",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch. By default HDF5 export rejects hand-pose JSONL "
            "that predates calibrated RGB-D ray fusion."
        ),
    )
    parser.add_argument(
        "--reuse-jsonl",
        action="store_true",
        help="Do not run inference even if --run-inference is set.",
    )
    parser.add_argument(
        "--show-inference",
        action="store_true",
        help="Show OpenCV preview while running offline WiLoR.",
    )
    parser.add_argument(
        "--no-inference-progress",
        action="store_true",
        help="Do not show progress while running offline WiLoR.",
    )
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument(
        "--segments",
        default="",
        help="Comma-separated inclusive frame ranges in record_index space, e.g. 0:120,150:260.",
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for non-interactive --segments export. "
            "Use 1 for serial output, 0 for conservative auto. Keep small when --max-points is large."
        ),
    )
    parser.add_argument(
        "--pose-frame",
        choices=("camera", "base", "world"),
        default="camera",
        help=(
            "Coordinate frame for pose_eular/cloud_rgb/keypoints_3d_m. "
            "'base' is kept for backward compatibility; 'world' uses the same transform path."
        ),
    )
    parser.add_argument(
        "--transform-to-world",
        action="store_true",
        help="Shortcut for --pose-frame world --camera-to-world-preset humanhand_l515.",
    )
    parser.add_argument(
        "--camera-to-base-preset",
        choices=("identity", "humanhand_l515"),
        default="identity",
        help="Backward-compatible preset name. Prefer --camera-to-world-preset for new data.",
    )
    parser.add_argument(
        "--camera-to-world-preset",
        choices=("identity", "humanhand_l515"),
        default=None,
        help="Camera-to-world extrinsic preset. humanhand_l515 is the matrix from sample_runthrough_HumanHand.py.",
    )
    parser.add_argument(
        "--camera-to-world-matrix",
        nargs=16,
        type=float,
        default=None,
        metavar="M",
        help="Custom row-major 4x4 camera-to-world extrinsic matrix.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_POINTS_NUM,
        help="Point-cloud samples per frame. Use 307200 to keep the old 640x480 full cloud.",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--allow-missing-gripper", action="store_true")
    parser.add_argument("--window-name", default="HumanHand offline slicer")
    args = parser.parse_args()
    if args.camera_pose_max_sync_error_ms < 0.0:
        raise ValueError("--camera-pose-max-sync-error-ms must be non-negative.")
    if args.rgb_reproject_workers < 0:
        raise ValueError("--rgb-reproject-workers must be >= 0.")
    if args.hand_depth_window < 1:
        raise ValueError("--hand-depth-window must be >= 1.")
    if args.hand_min_depth_m < 0.0 or args.hand_max_depth_m <= args.hand_min_depth_m:
        raise ValueError("Require 0 <= --hand-min-depth-m < --hand-max-depth-m.")
    if args.hand_min_valid_depth_points < 1:
        raise ValueError("--hand-min-valid-depth-points must be >= 1.")
    if (
        args.hand_rigid_temporal_filter_window < 3
        or args.hand_rigid_temporal_filter_window % 2 == 0
    ):
        raise ValueError("--hand-rigid-temporal-filter-window must be an odd integer >= 3.")
    if args.hand_rigid_temporal_filter_order < 1:
        raise ValueError("--hand-rigid-temporal-filter-order must be >= 1.")
    if args.hand_rigid_temporal_max_gap < 0:
        raise ValueError("--hand-rigid-temporal-max-gap must be >= 0.")
    if args.ego_trajectory_filter_cutoff_hz <= 0.0:
        raise ValueError("--ego-trajectory-filter-cutoff-hz must be positive.")
    if args.ego_trajectory_filter_order < 1:
        raise ValueError("--ego-trajectory-filter-order must be >= 1.")
    if args.ego_trajectory_max_angular_speed_deg_s < 0.0:
        raise ValueError("--ego-trajectory-max-angular-speed-deg-s must be non-negative.")
    if args.camera_reference_mode == "canonical" and args.canonical_camera_to_tracking_matrix is None:
        raise ValueError("--camera-reference-mode=canonical requires --canonical-camera-to-tracking-matrix.")
    if args.transform_to_world:
        args.pose_frame = "world"
        if args.camera_to_world_preset is None and args.camera_to_world_matrix is None:
            args.camera_to_world_preset = "humanhand_l515"
    if args.align_to_episode_first:
        if args.camera_pose_jsonl is None:
            raise ValueError("--align-to-episode-first requires --camera-pose-jsonl.")
        if args.camera_reference_mode != "episode_first":
            raise ValueError(
                "--align-to-episode-first requires --camera-reference-mode episode_first."
            )
        if args.pose_frame != "camera":
            raise ValueError("--align-to-episode-first requires --pose-frame camera.")
        args.require_camera_pose = True

    input_dir = Path(args.input).resolve()
    jsonl_path = Path(args.jsonl).resolve() if args.jsonl else input_dir / "handpose_wilor.jsonl"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_inference and not args.reuse_jsonl:
        run_offline_wilor(args, input_dir, jsonl_path)

    frame_records = load_frame_records(input_dir)
    if args.camera_pose_jsonl:
        frame_records = attach_external_camera_poses(
            frame_records,
            Path(args.camera_pose_jsonl).expanduser().resolve(),
            max_sync_error_ms=args.camera_pose_max_sync_error_ms,
        )
    metadata = load_metadata(input_dir)
    payloads = load_payloads(jsonl_path)
    validate_handpose_pipeline_versions(
        payloads,
        allow_legacy=bool(args.allow_legacy_handpose_jsonl),
        required_version=(
            REQUIRED_HANDPOSE_PIPELINE_VERSION
            if args.hand_rigid_temporal_filter
            else RAW_RIGID_HANDPOSE_PIPELINE_VERSION
        ),
    )
    samples = align_samples(frame_records, payloads)
    if not samples:
        raise RuntimeError("No matching RGB-D frames and WiLoR payloads were found.")

    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    camera_to_world = get_camera_to_world(args)

    static_segments = parse_segments(args.segments)
    saved_paths: list[Path] = []
    if static_segments:
        saved_paths.extend(
            save_static_segments_hdf5(
                static_segments,
                samples,
                input_dir,
                metadata,
                output_dir,
                camera_names,
                camera_to_world,
                args,
            )
        )

    if not args.no_interactive:
        saved_paths.extend(
            run_interactive_slicer(
                samples=samples,
                input_dir=input_dir,
                metadata=metadata,
                output_dir=output_dir,
                camera_names=camera_names,
                camera_to_world=camera_to_world,
                args=args,
            )
        )

    print(f"Done. Saved {len(saved_paths)} HDF5 episode(s) to {output_dir}")
    for path in saved_paths:
        print(path)


def run_offline_wilor(args: argparse.Namespace, input_dir: Path, jsonl_path: Path) -> None:
    script = Path(args.handpose_root) / "scripts/run_rgbd_sequence_wilor.py"
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--wilor-repo",
        str(args.wilor_repo),
        "--input",
        str(input_dir),
        "--jsonl",
        str(jsonl_path),
        "--checkpoint",
        args.checkpoint,
        "--model-cfg",
        args.model_cfg,
        "--detector",
        args.detector,
        "--fusion-mode",
        args.fusion_mode,
        "--gripper-x-offset-cm",
        str(args.gripper_x_offset_cm),
        "--gripper-z-offset-cm",
        str(args.gripper_z_offset_cm),
        "--depth-window",
        str(args.hand_depth_window),
        "--min-depth-m",
        str(args.hand_min_depth_m),
        "--max-depth-m",
        str(args.hand_max_depth_m),
        "--min-valid-depth-points",
        str(args.hand_min_valid_depth_points),
        "--depth-knn-backend",
        str(args.hand_depth_knn_backend),
        (
            "--depth-rigid-refinement"
            if args.hand_depth_rigid_refinement
            else "--no-depth-rigid-refinement"
        ),
        (
            "--rigid-temporal-filter"
            if args.hand_rigid_temporal_filter
            else "--no-rigid-temporal-filter"
        ),
        "--rigid-temporal-filter-window",
        str(args.hand_rigid_temporal_filter_window),
        "--rigid-temporal-filter-order",
        str(args.hand_rigid_temporal_filter_order),
        "--rigid-temporal-max-gap",
        str(args.hand_rigid_temporal_max_gap),
    ]
    if args.show_inference:
        cmd.append("--show")
    if args.fast:
        cmd.append("--fast")
    if args.force_handedness is not None:
        cmd.extend(["--force-handedness", args.force_handedness])
    print("Running offline WiLoR inference...")
    if args.no_inference_progress:
        subprocess.run(cmd, check=True)
    else:
        run_subprocess_with_jsonl_progress(
            cmd,
            total=len(load_frame_records(input_dir)),
            label="WiLoR inference",
        )


def run_subprocess_with_jsonl_progress(cmd: list[str], total: int, label: str) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess stdout.")

    start_s = time.time()
    completed = 0
    recent_non_json = deque(maxlen=20)
    try:
        for line in process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{") and stripped.endswith("}"):
                completed += 1
                print_progress(label, completed, total, start_s)
            else:
                recent_non_json.append(stripped)
                sys.stderr.write("\n" + stripped + "\n")
                sys.stderr.flush()
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        raise

    sys.stderr.write("\n")
    sys.stderr.flush()
    if return_code != 0:
        if recent_non_json:
            sys.stderr.write("Recent subprocess output:\n")
            for line in recent_non_json:
                sys.stderr.write(f"  {line}\n")
        raise subprocess.CalledProcessError(return_code, cmd)


def print_progress(label: str, completed: int, total: int, start_s: float) -> None:
    elapsed_s = max(time.time() - start_s, 1e-9)
    fps = completed / elapsed_s
    if total > 0:
        percent = min(100.0, completed / total * 100.0)
        message = f"\r{label}: {completed}/{total} ({percent:5.1f}%) {fps:5.2f} fps"
    else:
        message = f"\r{label}: {completed} frames {fps:5.2f} fps"
    sys.stderr.write(message)
    sys.stderr.flush()


def load_metadata(input_dir: Path) -> dict:
    path = input_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame_records(input_dir: Path) -> list[dict]:
    path = input_dir / "frames.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def resolve_camera_pose_jsonl_paths(path: Path) -> list[Path]:
    """Resolve one pose JSONL file or all per-segment ORB-SLAM3 pose JSONLs."""

    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    preferred = sorted(path.glob("segment_*/dataset/camera_pose_orbslam3.jsonl"))
    if preferred:
        return preferred

    direct = path / "camera_pose_orbslam3.jsonl"
    if direct.is_file():
        return [direct]

    recursive = sorted(path.glob("**/camera_pose_orbslam3.jsonl"))
    if recursive:
        return recursive

    raise FileNotFoundError(
        f"No camera_pose_orbslam3.jsonl files found under {path}. Expected "
        "segment_*/dataset/camera_pose_orbslam3.jsonl."
    )


def camera_pose_sequence_name(jsonl_path: Path) -> str:
    for parent in (jsonl_path.parent, *jsonl_path.parents):
        if parent.name.startswith("segment_"):
            return parent.name
    return jsonl_path.stem


def attach_external_camera_poses(
    frame_records: list[dict],
    jsonl_path: Path,
    *,
    max_sync_error_ms: float | None = None,
) -> list[dict]:
    """Merge full-dataset per-segment camera trajectories, then attach them by record_index."""

    jsonl_paths = resolve_camera_pose_jsonl_paths(jsonl_path)
    pose_by_index: dict[int, dict] = {}
    for pose_path in jsonl_paths:
        sequence_name = camera_pose_sequence_name(pose_path)
        for line_number, line in enumerate(pose_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_index = record.get("record_index", record.get("index"))
            if raw_index is None:
                raise KeyError(f"{pose_path}:{line_number} has no record_index.")
            index = int(raw_index)
            matrix = matrix_from_json_record(record)
            pose_record = {
                "camera_to_tracking": matrix.tolist(),
                "camera_pose_source": str(
                    record.get("tracking_source", record.get("source", "external_vio"))
                ),
                "camera_pose_timestamp_ms": record.get("timestamp_ms"),
                "camera_pose_valid": bool(record.get("valid", True)),
                "camera_pose_sequence": sequence_name,
                "camera_pose_jsonl": str(pose_path),
            }
            if index in pose_by_index:
                previous = pose_by_index[index]
                previous_matrix = np.asarray(previous["camera_to_tracking"], dtype=np.float64)
                if not np.allclose(previous_matrix, matrix, atol=1e-9):
                    raise ValueError(
                        f"Conflicting camera poses for record_index={index}: "
                        f"{previous['camera_pose_jsonl']} and {pose_path}."
                    )
                continue
            pose_by_index[index] = pose_record

    output = []
    matched = 0
    for frame in frame_records:
        enriched = dict(frame)
        index = int(frame.get("index", -1))
        if index in pose_by_index:
            pose_record = dict(pose_by_index[index])
            rgb_timestamp_ms = frame.get("timestamp_ms")
            pose_timestamp_ms = pose_record.get("camera_pose_timestamp_ms")
            if rgb_timestamp_ms is not None and pose_timestamp_ms is not None:
                sync_error_ms = float(pose_timestamp_ms) - float(rgb_timestamp_ms)
                if (
                    max_sync_error_ms is not None
                    and float(max_sync_error_ms) > 0.0
                    and abs(sync_error_ms) > float(max_sync_error_ms)
                ):
                    raise ValueError(
                        f"Camera pose record_index={index} is out of sync with RGB-D by "
                        f"{sync_error_ms:+.3f} ms (limit={float(max_sync_error_ms):.3f} ms)."
                    )
                pose_record["camera_pose_sync_error_ms"] = sync_error_ms
            enriched.update(pose_record)
            matched += 1
        output.append(enriched)
    if matched == 0:
        raise RuntimeError(
            f"No record_index values from {jsonl_path} matched the RGB-D frames. "
            "Timestamp-only matching is intentionally not guessed."
        )
    print(
        f"Attached {matched}/{len(frame_records)} external full-SE(3) camera poses "
        f"from {len(jsonl_paths)} segment trajectory file(s)."
    )
    return output


def load_payloads(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    payloads = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payloads.append(json.loads(line))
    return payloads


def validate_handpose_pipeline_versions(
    payloads: list[dict],
    *,
    allow_legacy: bool,
    required_version: str = REQUIRED_HANDPOSE_PIPELINE_VERSION,
) -> None:
    versions = sorted(
        {
            str(payload.get("handpose_pipeline_version", "missing"))
            for payload in payloads
        }
    )
    if versions == [str(required_version)]:
        return
    message = (
        "Hand-pose JSONL was generated by an incompatible fusion pipeline: "
        f"found={versions}, required={str(required_version)!r}. "
        "Regenerate it with --run-inference so hand keypoints and scene points use the same "
        "calibrated camera rays."
    )
    if not allow_legacy:
        raise RuntimeError(message)
    print(f"[warn] {message}", file=sys.stderr)


def align_samples(frame_records: list[dict], payloads: list[dict]) -> list[tuple[dict, dict]]:
    frames_by_index = {int(record.get("index", idx)): record for idx, record in enumerate(frame_records)}
    samples = []
    for payload_idx, payload in enumerate(payloads):
        record_index = int(payload.get("record_index", payload_idx))
        frame = frames_by_index.get(record_index)
        if frame is not None:
            samples.append((frame, payload))
    return samples


def parse_segments(text: str) -> list[Segment]:
    segments = []
    if not text.strip():
        return segments
    for item in text.split(","):
        start_text, end_text = item.strip().split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
        if end < start:
            start, end = end, start
        segments.append(Segment(start=start, end=end))
    return segments


def get_camera_to_world(args: argparse.Namespace) -> np.ndarray:
    if args.camera_to_world_matrix is not None:
        transform = np.asarray(args.camera_to_world_matrix, dtype=np.float64).reshape(4, 4)
        validate_extrinsic(transform)
        return transform

    preset = args.camera_to_world_preset
    if preset is None:
        preset = args.camera_to_base_preset
    if preset == "identity":
        return np.eye(4, dtype=np.float64)
    if preset == "humanhand_l515":
        return HUMANHAND_L515_CAMERA_TO_WORLD.copy()
    raise ValueError(preset)


def validate_extrinsic(transform: np.ndarray) -> None:
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Camera-to-world extrinsic must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("Camera-to-world extrinsic last row must be [0, 0, 0, 1].")


def run_interactive_slicer(
    samples: list[tuple[dict, dict]],
    input_dir: Path,
    metadata: dict,
    output_dir: Path,
    camera_names: list[str],
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> list[Path]:
    import cv2

    index = 0
    start_record_index: int | None = None
    start_camera_pose_sequence: str | None = None
    saved_paths: list[Path] = []
    print(
        "Controls: Right/D next, Left/A previous, Up/W set start, "
        "Down/S save end, R clear start, U delete last saved, Q/Esc quit"
    )
    while True:
        frame_record, payload = samples[index]
        color_bgr, _depth_m, intrinsics = load_rgbd_frame(input_dir, frame_record, metadata)
        preview = color_bgr.copy()
        draw_payload_preview(cv2, preview, payload, intrinsics)
        draw_overlay(
            cv2,
            preview,
            index,
            len(samples),
            int(frame_record.get("index", index)),
            str(frame_record.get("camera_pose_sequence", "no_pose")),
            start_record_index,
            start_camera_pose_sequence,
            len(saved_paths),
        )
        cv2.imshow(args.window_name, preview)
        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (83, 2555904, 65363, ord("d"), ord("D")):
            index = min(index + 1, len(samples) - 1)
        elif key in (81, 2424832, 65361, ord("a"), ord("A")):
            index = max(index - 1, 0)
        elif key in (82, 2490368, 65362, ord("w"), ord("W")):
            start_record_index = int(frame_record.get("index", index))
            start_camera_pose_sequence = str(frame_record.get("camera_pose_sequence", "no_pose"))
            print(
                f"Episode start = frame {start_record_index} "
                f"({start_camera_pose_sequence})"
            )
        elif key in (84, 2621440, 65364, ord("s"), ord("S")):
            end_record_index = int(frame_record.get("index", index))
            if start_record_index is None:
                print("Set a start frame first with Up/W.")
                continue
            end_camera_pose_sequence = str(frame_record.get("camera_pose_sequence", "no_pose"))
            if end_camera_pose_sequence != start_camera_pose_sequence:
                print(
                    "Episode was not saved: start/end cross independent ORB-SLAM3 trajectories "
                    f"({start_camera_pose_sequence} -> {end_camera_pose_sequence})."
                )
                continue
            segment = Segment(
                start=min(start_record_index, end_record_index),
                end=max(start_record_index, end_record_index),
            )
            try:
                saved_path = save_segment_hdf5(
                    segment,
                    samples,
                    input_dir,
                    metadata,
                    output_dir,
                    camera_names,
                    camera_to_world,
                    args,
                )
            except RuntimeError as exc:
                print(f"Episode was not saved: {exc}")
                continue
            saved_paths.append(saved_path)
            start_record_index = None
            start_camera_pose_sequence = None
        elif key in (ord("r"), ord("R")):
            start_record_index = None
            start_camera_pose_sequence = None
            print("Cleared current episode start.")
        elif key in (ord("u"), ord("U")) and saved_paths:
            last = saved_paths.pop()
            last.unlink(missing_ok=True)
            print(f"Deleted {last}")
    cv2.destroyAllWindows()
    return saved_paths


def save_static_segments_hdf5(
    segments: list[Segment],
    samples: list[tuple[dict, dict]],
    input_dir: Path,
    metadata: dict,
    output_dir: Path,
    camera_names: list[str],
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> list[Path]:
    output_paths = reserve_episode_paths(output_dir, len(segments))
    worker_count = resolve_segment_workers(args.segment_workers, len(segments))
    if worker_count <= 1:
        return [
            save_segment_hdf5(
                segment,
                samples,
                input_dir,
                metadata,
                output_dir,
                camera_names,
                camera_to_world,
                args,
                output_path=path,
                progress_enabled=True,
            )
            for segment, path in zip(segments, output_paths)
        ]

    print(
        f"Saving {len(segments)} static segments with {worker_count} worker processes.",
        flush=True,
    )
    result_paths: list[Path | None] = [None] * len(segments)
    start_s = time.time()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_segment_worker,
        initargs=(
            samples,
            str(input_dir),
            metadata,
            str(output_dir),
            camera_names,
            camera_to_world,
            args,
        ),
    ) as executor:
        future_to_index = {
            executor.submit(save_segment_worker, (index, segment, str(output_paths[index]))): index
            for index, segment in enumerate(segments)
        }
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            result_paths[index] = future.result()
            completed += 1
            print_progress("Saving static segments", completed, len(segments), start_s)
    sys.stderr.write("\n")
    sys.stderr.flush()
    return [path for path in result_paths if path is not None]


def resolve_segment_workers(requested: int, segment_count: int) -> int:
    if segment_count <= 1:
        return 1
    if requested < 0:
        raise ValueError("--segment-workers must be >= 0")
    if requested == 0:
        return min(segment_count, max(1, os.cpu_count() or 1), 4)
    return min(segment_count, max(1, requested))


def resolve_rgb_reproject_workers(
    requested: int,
    frame_count: int,
    segment_workers: int,
) -> int:
    """Choose bounded frame-level parallelism for full-resolution RGB reprojection."""

    if frame_count <= 1:
        return 1
    if requested < 0:
        raise ValueError("--rgb-reproject-workers must be >= 0")
    if requested > 0:
        return min(frame_count, requested)

    cpu_count = max(1, os.cpu_count() or 1)
    # When episodes are already exported by several processes, use fewer threads in
    # each process to avoid CPU oversubscription and excessive RGB-D memory pressure.
    automatic_cap = 2 if segment_workers != 1 else 4
    return min(frame_count, cpu_count, automatic_cap)


def init_segment_worker(
    samples: list[tuple[dict, dict]],
    input_dir: str,
    metadata: dict,
    output_dir: str,
    camera_names: list[str],
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> None:
    global _SEGMENT_WORKER_CONTEXT, _VIDEO_CAPTURE_CACHE
    _VIDEO_CAPTURE_CACHE = {}
    _SEGMENT_WORKER_CONTEXT = {
        "samples": samples,
        "input_dir": Path(input_dir),
        "metadata": metadata,
        "output_dir": Path(output_dir),
        "camera_names": camera_names,
        "camera_to_world": np.asarray(camera_to_world),
        "args": args,
    }


def save_segment_worker(task: tuple[int, Segment, str]) -> Path:
    _index, segment, output_path = task
    if _SEGMENT_WORKER_CONTEXT is None:
        raise RuntimeError("Segment worker was not initialized.")
    return save_segment_hdf5(
        segment,
        _SEGMENT_WORKER_CONTEXT["samples"],
        _SEGMENT_WORKER_CONTEXT["input_dir"],
        _SEGMENT_WORKER_CONTEXT["metadata"],
        _SEGMENT_WORKER_CONTEXT["output_dir"],
        _SEGMENT_WORKER_CONTEXT["camera_names"],
        _SEGMENT_WORKER_CONTEXT["camera_to_world"],
        _SEGMENT_WORKER_CONTEXT["args"],
        output_path=Path(output_path),
        progress_enabled=False,
    )


def save_segment_hdf5(
    segment: Segment,
    samples: list[tuple[dict, dict]],
    input_dir: Path,
    metadata: dict,
    output_dir: Path,
    camera_names: list[str],
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
    output_path: Path | None = None,
    progress_enabled: bool = True,
) -> Path:
    selected = [
        (frame, payload)
        for frame, payload in samples
        if segment.start <= int(frame.get("index", -1)) <= segment.end
    ]
    if not selected:
        raise RuntimeError(f"Segment {segment.start}:{segment.end} contains no frames.")

    if not camera_names:
        raise ValueError("At least one camera name is required.")

    source_indices = [
        int(frame.get("index", index)) for index, (frame, _payload) in enumerate(selected)
    ]
    camera_to_tracking_poses: list[np.ndarray | None] = []
    camera_pose_sources: list[str] = []
    camera_pose_sequences: list[str] = []
    camera_pose_sync_errors_ms: list[float] = []
    for frame_record, _payload in selected:
        raw_camera_pose = frame_record.get("camera_to_tracking", frame_record.get("camera_to_world"))
        camera_pose_valid = bool(frame_record.get("camera_pose_valid", raw_camera_pose is not None))
        camera_pose_sync_errors_ms.append(float(frame_record.get("camera_pose_sync_error_ms", np.nan)))
        if raw_camera_pose is None or not camera_pose_valid:
            camera_to_tracking_poses.append(None)
            continue
        camera_to_tracking_poses.append(
            validate_transform_sequence(np.asarray(raw_camera_pose, dtype=np.float64).reshape(4, 4))[0]
        )
        camera_pose_sources.append(str(frame_record.get("camera_pose_source", "recorded_pose")))
        camera_pose_sequences.append(str(frame_record.get("camera_pose_sequence", "unknown")))

    available_camera_pose_count = sum(pose is not None for pose in camera_to_tracking_poses)
    if 0 < available_camera_pose_count < len(camera_to_tracking_poses):
        missing_indices = [
            source_indices[index] for index, pose in enumerate(camera_to_tracking_poses) if pose is None
        ]
        raise RuntimeError(
            "Camera poses are only partially available inside the selected segment. "
            f"Missing/invalid record_index values: {missing_indices[:20]}"
            f"{'...' if len(missing_indices) > 20 else ''}. "
            "A partial trajectory must not be silently treated as a fixed camera."
        )
    has_camera_poses = available_camera_pose_count == len(camera_to_tracking_poses)
    if args.require_camera_pose and not has_camera_poses:
        raise RuntimeError(
            f"Segment {segment.start}:{segment.end} has no full-SE(3) camera trajectory, "
            "but --require-camera-pose was requested."
        )
    camera_pose_source = (
        camera_pose_sources[0]
        if camera_pose_sources and len(set(camera_pose_sources)) == 1
        else ("mixed" if camera_pose_sources else "none")
    )
    unique_camera_pose_sequences = sorted(set(camera_pose_sequences))
    camera_pose_sequence = (
        unique_camera_pose_sequences[0]
        if len(unique_camera_pose_sequences) == 1
        else ("mixed" if unique_camera_pose_sequences else "none")
    )

    episode_first_camera_to_tracking: np.ndarray | None = None
    tracking_to_episode_first_camera: np.ndarray | None = None
    output_camera_poses: list[np.ndarray | None] = list(camera_to_tracking_poses)
    if args.align_to_episode_first:
        if not has_camera_poses:
            raise RuntimeError(
                "--align-to-episode-first requires a valid camera pose for every frame in the episode."
            )
        if len(unique_camera_pose_sequences) != 1:
            raise RuntimeError(
                "The selected episode crosses independent ORB-SLAM3 segment trajectories: "
                f"{unique_camera_pose_sequences}. Choose start/end frames inside one segment."
            )
        episode_first_camera_to_tracking = np.asarray(camera_to_tracking_poses[0], dtype=np.float64)
        tracking_to_episode_first_camera = np.linalg.inv(episode_first_camera_to_tracking)
        output_camera_poses = [
            tracking_to_episode_first_camera @ np.asarray(pose, dtype=np.float64)
            for pose in camera_to_tracking_poses
        ]
        if not np.allclose(output_camera_poses[0], np.eye(4), atol=1e-8):
            raise RuntimeError("Episode-first camera pose did not normalize to identity.")

    clouds = []
    qpos = []
    pose_eular = []
    action = []
    eff_angular = []
    gripper_quat_xyzw = []
    keypoints_3d_m = []
    timestamps_ms = []

    start_s = time.time()
    total_frames = len(selected)
    images: list[np.ndarray | None] = [None] * total_frames
    episode_first_intrinsics: CameraIntrinsics | None = None
    episode_first_image_shape: tuple[int, int] | None = None

    rgb_reprojection_active = bool(
        args.align_to_episode_first and args.reproject_rgb_to_episode_first
    )
    rgb_reproject_worker_count = (
        resolve_rgb_reproject_workers(
            int(args.rgb_reproject_workers),
            total_frames,
            int(args.segment_workers),
        )
        if rgb_reprojection_active
        else 0
    )
    rgb_executor = (
        ThreadPoolExecutor(
            max_workers=rgb_reproject_worker_count,
            thread_name_prefix="rgb-reproject",
        )
        if rgb_reproject_worker_count > 1
        else None
    )
    pending_rgb = deque()
    max_pending_rgb = max(1, 2 * rgb_reproject_worker_count)

    def collect_one_rgb() -> None:
        image_index, future = pending_rgb.popleft()
        images[image_index] = future.result()

    try:
        for frame_index, (frame_record, payload) in enumerate(selected):
            frame_offset = frame_index + 1
            color_bgr, depth_m, intrinsics = load_rgbd_frame(input_dir, frame_record, metadata)
            if color_bgr.shape[:2] != depth_m.shape:
                raise ValueError(
                    f"Aligned RGB/depth shape mismatch at record_index={source_indices[frame_index]}: "
                    f"color={color_bgr.shape[:2]}, depth={depth_m.shape}."
                )
            if frame_index == 0:
                episode_first_intrinsics = intrinsics
                episode_first_image_shape = depth_m.shape

            cloud = make_cloud_rgb(color_bgr, depth_m, intrinsics, args.max_points)
            hand = choose_hand(payload, allow_missing=args.allow_missing_gripper)
            pose, quat, opening = gripper_state_from_hand(
                hand,
                args.pose_frame,
                camera_to_world,
                requested_x_offset_m=args.gripper_x_offset_cm / 100.0,
                requested_z_offset_m=args.gripper_z_offset_cm / 100.0,
            )
            joints = keypoints_from_hand(hand, args.pose_frame, camera_to_world)

            if args.align_to_episode_first:
                current_camera_to_episode_world = np.asarray(
                    output_camera_poses[frame_index],
                    dtype=np.float64,
                )
                cloud = transform_cloud(cloud, current_camera_to_episode_world)
                if hand is not None:
                    pose, quat = transform_gripper_state(
                        pose,
                        quat,
                        current_camera_to_episode_world,
                    )
                joints = transform_keypoints(joints, current_camera_to_episode_world)

                if args.reproject_rgb_to_episode_first:
                    if episode_first_intrinsics is None or episode_first_image_shape is None:
                        raise RuntimeError("Episode-first RGB projection metadata is unavailable.")
                    projection_kwargs = dict(
                        color_bgr=color_bgr,
                        depth_m=depth_m,
                        source_intrinsics=intrinsics,
                        source_camera_to_reference=current_camera_to_episode_world,
                        reference_intrinsics=episode_first_intrinsics,
                        reference_height=int(episode_first_image_shape[0]),
                        reference_width=int(episode_first_image_shape[1]),
                        output_width=int(args.image_width),
                        output_height=int(args.image_height),
                    )
                    if rgb_executor is None:
                        images[frame_index] = reproject_rgbd_to_reference_rgb(**projection_kwargs)
                    else:
                        pending_rgb.append(
                            (
                                frame_index,
                                rgb_executor.submit(
                                    reproject_rgbd_to_reference_rgb,
                                    **projection_kwargs,
                                ),
                            )
                        )
                        if len(pending_rgb) >= max_pending_rgb:
                            collect_one_rgb()
                else:
                    images[frame_index] = resize_image(
                        color_bgr,
                        args.image_width,
                        args.image_height,
                    )[:, :, ::-1].copy()
            else:
                images[frame_index] = resize_image(
                    color_bgr,
                    args.image_width,
                    args.image_height,
                )[:, :, ::-1].copy()
                if should_transform_to_output_frame(args.pose_frame):
                    cloud = transform_cloud(cloud, camera_to_world)

            clouds.append(cloud)
            qpos.append(np.zeros(7, dtype=np.float64))
            pose_eular.append(pose)
            action.append(np.zeros(7, dtype=np.float64))
            eff_angular.append(np.asarray([opening], dtype=np.float64))
            gripper_quat_xyzw.append(quat)
            keypoints_3d_m.append(joints)
            timestamps_ms.append(
                float(payload.get("timestamp_ms") or frame_record.get("timestamp_ms") or np.nan)
            )
            if progress_enabled and (
                frame_offset == 1
                or frame_offset == total_frames
                or frame_offset % 25 == 0
            ):
                print_progress(
                    f"Preparing segment {segment.start}:{segment.end}",
                    frame_offset,
                    total_frames,
                    start_s,
                )

        while pending_rgb:
            collect_one_rgb()
    finally:
        if rgb_executor is not None:
            rgb_executor.shutdown(wait=True, cancel_futures=True)

    missing_rendered_images = [
        index for index, image in enumerate(images) if image is None
    ]
    if missing_rendered_images:
        raise RuntimeError(
            "RGB reprojection did not produce every frame: "
            f"missing local indices {missing_rendered_images[:20]}"
        )
    stored_images = [image for image in images if image is not None]
    trajectory_filter_metrics: dict[str, float | int | bool | str] = {
        "method": "none",
    }
    if args.ego_trajectory_filter == "se3_lowpass":
        pose_eular, gripper_quat_xyzw, trajectory_filter_metrics = filter_gripper_pose_sequence(
            pose_eular,
            gripper_quat_xyzw,
            timestamps_ms,
            cutoff_hz=float(args.ego_trajectory_filter_cutoff_hz),
            order=int(args.ego_trajectory_filter_order),
            max_angular_speed_deg_s=float(
                args.ego_trajectory_max_angular_speed_deg_s
            ),
            parallel_jaw_symmetry=bool(args.ego_trajectory_parallel_jaw_symmetry),
            preserve_endpoints=bool(args.ego_trajectory_filter_preserve_endpoints),
        )
        print(
            "[ego-trajectory-filter] "
            f"segment={segment.start}:{segment.end} "
            f"cutoff={trajectory_filter_metrics['cutoff_hz']:.2f}Hz "
            f"translation_residual_rms="
            f"{trajectory_filter_metrics['translation_residual_rms_mm']:.2f}mm "
            f"rotation_residual_rms="
            f"{trajectory_filter_metrics['rotation_residual_rms_deg']:.2f}deg "
            f"symmetry_switches="
            f"{trajectory_filter_metrics['parallel_jaw_state_switch_count']} "
            f"frame_resets="
            f"{trajectory_filter_metrics['orientation_frame_reset_count']} "
            f"timestamp_repairs="
            f"{trajectory_filter_metrics['timestamp_repair_count']}",
            flush=True,
        )
    else:
        pose_eular, gripper_quat_xyzw = canonicalize_gripper_pose_sequence(
            pose_eular,
            gripper_quat_xyzw,
        )
    if progress_enabled:
        sys.stderr.write("\n")
        sys.stderr.flush()

    path = output_path if output_path is not None else next_episode_path(output_dir)
    with h5py.File(path, "x", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs["sim"] = False
        root.attrs["task"] = args.task
        root.attrs["source_rgbd_dir"] = str(input_dir)
        root.attrs["source_jsonl"] = str(
            Path(args.jsonl).resolve() if args.jsonl else input_dir / "handpose_wilor.jsonl"
        )
        root.attrs["segment_start_record_index"] = segment.start
        root.attrs["segment_end_record_index"] = segment.end
        root.attrs["source_pose_frame"] = args.pose_frame
        root.attrs["episode_first_alignment_applied"] = bool(args.align_to_episode_first)
        root.attrs["rgb_reprojected_to_episode_first"] = bool(
            args.align_to_episode_first and args.reproject_rgb_to_episode_first
        )
        root.attrs["rgb_reprojection_point_selection"] = (
            "all_valid_depth_pixels"
            if args.align_to_episode_first and args.reproject_rgb_to_episode_first
            else "not_applied"
        )
        root.attrs["rgb_reprojection_occlusion_method"] = (
            "nearest_depth_z_buffer"
            if args.align_to_episode_first and args.reproject_rgb_to_episode_first
            else "not_applied"
        )
        root.attrs["rgb_reprojection_hole_fill"] = "black_zero"
        root.attrs["rgb_reprojection_compute"] = (
            "vectorized_numpy_matrix" if rgb_reprojection_active else "not_applied"
        )
        root.attrs["rgb_reprojection_parallel_backend"] = (
            "bounded_thread_pool"
            if rgb_reprojection_active and rgb_reproject_worker_count > 1
            else ("serial" if rgb_reprojection_active else "not_applied")
        )
        root.attrs["rgb_reprojection_workers"] = int(rgb_reproject_worker_count)
        if args.align_to_episode_first:
            if episode_first_camera_to_tracking is None or tracking_to_episode_first_camera is None:
                raise RuntimeError("Internal error: episode-first alignment metadata is unavailable.")
            root.attrs["pose_frame"] = "world"
            root.attrs["reference_frame"] = "episode_first_camera"
            root.attrs["world_frame_definition"] = "camera frame at first source_record_index"
            root.attrs["episode_first_record_index"] = int(source_indices[0])
            root.attrs["episode_first_camera_to_tracking"] = episode_first_camera_to_tracking
            root.attrs["tracking_to_episode_first_camera"] = tracking_to_episode_first_camera
            root.attrs["overview_camera_storage_name"] = camera_names[0]
            root.attrs["uses_camera_extrinsic"] = False
            root.attrs["uses_camera_tracking_pose"] = True
        else:
            root.attrs["pose_frame"] = args.pose_frame
            if args.pose_frame == "camera":
                # This is still raw per-frame camera data. The converter defines
                # model world from the first overview-camera frame after applying
                # the optional camera-to-tracking trajectory.
                root.attrs["reference_frame"] = "current_camera"
                root.attrs["overview_camera_storage_name"] = camera_names[0]
                root.attrs["uses_camera_extrinsic"] = False
            else:
                # Legacy export mode only. The current training/conversion pipeline
                # expects pose_frame="camera" and does not consume this extrinsic.
                root.attrs["camera_to_world"] = camera_to_world
                root.attrs["uses_camera_extrinsic"] = True
        root.attrs["gripper_x_offset_cm"] = float(args.gripper_x_offset_cm)
        root.attrs["gripper_z_offset_cm"] = float(args.gripper_z_offset_cm)
        root.attrs["image_color_format"] = "rgb"
        root.attrs["cloud_color_format"] = "rgb_0_255"
        root.attrs["max_points"] = int(args.max_points)
        if episode_first_intrinsics is not None:
            root.attrs["camera_distortion_model"] = str(
                episode_first_intrinsics.model or "none"
            )
            root.attrs["camera_distortion_coeffs"] = np.asarray(
                episode_first_intrinsics.coeffs,
                dtype=np.float64,
            )
        root.attrs["handpose_pipeline_version"] = str(
            samples[0][1].get("handpose_pipeline_version", "missing")
        )
        root.attrs["gripper_rotation_representation"] = "quaternion_xyzw+continuous_euler_zyx"
        root.attrs["ego_trajectory_filter"] = str(args.ego_trajectory_filter)
        root.attrs["ego_trajectory_filter_metrics_json"] = json.dumps(
            trajectory_filter_metrics,
            ensure_ascii=False,
            sort_keys=True,
        )
        root.attrs["ego_trajectory_filter_clouds_modified"] = False
        root.attrs["ego_trajectory_filter_keypoints_modified"] = False
        root.attrs["camera_names"] = json.dumps(camera_names, ensure_ascii=False)
        root.attrs["camera_datasets_hardlinked"] = len(camera_names) > 1
        root.attrs["camera_tracking_pose_available"] = bool(has_camera_poses)
        root.attrs["camera_pose_source"] = camera_pose_source
        root.attrs["camera_pose_sequence"] = camera_pose_sequence
        root.attrs["camera_reference_mode"] = str(args.camera_reference_mode)
        if args.canonical_camera_to_tracking_matrix is not None:
            canonical_camera_to_tracking = validate_transform_sequence(
                np.asarray(args.canonical_camera_to_tracking_matrix, dtype=np.float64).reshape(4, 4)
            )[0]
            root.attrs["canonical_camera_to_tracking"] = canonical_camera_to_tracking

        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        cloud_grp = obs.create_group("cloud_rgb")
        first_camera = camera_names[0]
        first_image = image_grp.create_dataset(
            first_camera,
            data=np.asarray(stored_images, dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )
        first_cloud = cloud_grp.create_dataset(
            first_camera,
            data=np.asarray(clouds, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        first_cloud.attrs["coordinate_frame"] = str(root.attrs["reference_frame"])
        first_cloud.attrs["alignment_applied"] = bool(args.align_to_episode_first)
        for name in camera_names[1:]:
            image_grp[name] = first_image
            cloud_grp[name] = first_cloud
        if has_camera_poses:
            camera_pose_grp = obs.create_group("camera_tracking_pose")
            first_camera_pose = camera_pose_grp.create_dataset(
                first_camera,
                data=np.asarray(output_camera_poses, dtype=np.float64),
            )
            if args.align_to_episode_first:
                first_camera_pose.attrs["transform_direction"] = "camera_to_episode_first_camera"
                first_camera_pose.attrs["notation"] = "T_episodeFirstCamera<-camera"
                first_camera_pose.attrs["reference_record_index"] = int(source_indices[0])
                first_camera_pose.attrs["first_pose_is_identity"] = True
            else:
                first_camera_pose.attrs["transform_direction"] = "camera_to_tracking"
                first_camera_pose.attrs["notation"] = "T_tracking<-camera"
            first_camera_pose.attrs["translation_unit"] = "meter"
            first_camera_pose.attrs["pose_format"] = "matrix"
            first_camera_pose.attrs["tracking_source"] = camera_pose_source
            first_camera_pose.attrs["camera_reference_mode"] = str(args.camera_reference_mode)
            if args.canonical_camera_to_tracking_matrix is not None:
                first_camera_pose.attrs["canonical_camera_to_tracking"] = np.asarray(
                    args.canonical_camera_to_tracking_matrix,
                    dtype=np.float64,
                ).reshape(4, 4)
            finite_sync_errors = np.asarray(camera_pose_sync_errors_ms, dtype=np.float64)
            finite_sync_errors = finite_sync_errors[np.isfinite(finite_sync_errors)]
            first_camera_pose.attrs["max_abs_sync_error_ms"] = (
                float(np.max(np.abs(finite_sync_errors))) if len(finite_sync_errors) else np.nan
            )
            for name in camera_names[1:]:
                camera_pose_grp[name] = first_camera_pose
            obs.create_dataset(
                "camera_tracking_pose_sync_error_ms",
                data=np.asarray(camera_pose_sync_errors_ms, dtype=np.float64),
            )

        obs.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float64))
        pose_dataset = obs.create_dataset("pose_eular", data=np.asarray(pose_eular, dtype=np.float64))
        pose_dataset.attrs["coordinate_frame"] = str(root.attrs["reference_frame"])
        pose_dataset.attrs["trajectory_filter"] = str(args.ego_trajectory_filter)
        obs.create_dataset("eff_angular", data=np.asarray(eff_angular, dtype=np.float64))
        quat_dataset = obs.create_dataset(
            "gripper_quat_xyzw", data=np.asarray(gripper_quat_xyzw, dtype=np.float64)
        )
        quat_dataset.attrs["coordinate_frame"] = str(root.attrs["reference_frame"])
        quat_dataset.attrs["trajectory_filter"] = str(args.ego_trajectory_filter)
        keypoint_dataset = obs.create_dataset(
            "keypoints_3d_m", data=np.asarray(keypoints_3d_m, dtype=np.float64)
        )
        keypoint_dataset.attrs["coordinate_frame"] = str(root.attrs["reference_frame"])
        root.create_dataset("action", data=np.asarray(action, dtype=np.float64))
        root.create_dataset("source_record_index", data=np.asarray(source_indices, dtype=np.int64))
        root.create_dataset("timestamp_ms", data=np.asarray(timestamps_ms, dtype=np.float64))

    if progress_enabled:
        print(f"Saved {path} ({len(selected)} frames, record_index {segment.start}:{segment.end})")
    return path


def reserve_episode_paths(output_dir: Path, count: int) -> list[Path]:
    start_index = next_episode_index(output_dir)
    return [output_dir / f"episode_{start_index + offset}.hdf5" for offset in range(count)]


def next_episode_path(output_dir: Path) -> Path:
    return output_dir / f"episode_{next_episode_index(output_dir)}.hdf5"


def next_episode_index(output_dir: Path) -> int:
    existing = []
    for path in output_dir.glob("episode_*.hdf5"):
        try:
            existing.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(existing, default=-1) + 1


def load_rgbd_frame(
    input_dir: Path,
    frame_record: dict,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    color_bgr = read_frame_color_bgr(input_dir, frame_record, metadata)
    depth_m = read_frame_depth_m(input_dir, frame_record)
    intrinsics_data = frame_record.get("intrinsics") or metadata.get("intrinsics")
    if intrinsics_data is None:
        raise KeyError("No intrinsics in frame record or metadata")
    intrinsics = CameraIntrinsics(
        width=int(intrinsics_data["width"]),
        height=int(intrinsics_data["height"]),
        fx=float(intrinsics_data["fx"]),
        fy=float(intrinsics_data["fy"]),
        ppx=float(intrinsics_data["ppx"]),
        ppy=float(intrinsics_data["ppy"]),
        coeffs=tuple(float(value) for value in intrinsics_data.get("coeffs", ())),
        model=(
            None
            if intrinsics_data.get("model") is None
            else str(intrinsics_data.get("model"))
        ),
    )
    return color_bgr, depth_m, intrinsics


def read_frame_color_bgr(input_dir: Path, frame_record: dict, metadata: dict) -> np.ndarray:
    if "color_path" in frame_record:
        return read_color_bgr(input_dir / frame_record["color_path"])
    color_video_path = frame_record.get("color_video_path") or metadata.get("color_video_path")
    if color_video_path is not None:
        frame_index = int(frame_record.get("color_video_frame_index", frame_record.get("index", 0)))
        return read_color_video_frame(input_dir / color_video_path, frame_index)
    if "color_png_path" in frame_record:
        return read_color_bgr(input_dir / frame_record["color_png_path"])
    raise KeyError("Frame record has neither color_path nor color_video_path.")


def read_frame_depth_m(input_dir: Path, frame_record: dict) -> np.ndarray:
    if "depth_m_path" in frame_record:
        return np.load(input_dir / frame_record["depth_m_path"]).astype(np.float32)
    if "depth_png_path" in frame_record:
        return read_depth_png_m(input_dir / frame_record["depth_png_path"])
    raise KeyError("Frame record has neither depth_m_path nor depth_png_path.")


def read_color_bgr(path: Path) -> np.ndarray:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image
    except ModuleNotFoundError:
        from PIL import Image

        if not path.exists():
            raise FileNotFoundError(path)
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()


def read_color_video_frame(path: Path, frame_index: int) -> np.ndarray:
    import cv2

    key = str(path)
    state = _VIDEO_CAPTURE_CACHE.get(key)
    if state is None:
        capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open color video: {path}")
        state = {"capture": capture, "next_frame_index": 0}
        _VIDEO_CAPTURE_CACHE[key] = state
    capture = state["capture"]
    if int(state["next_frame_index"]) != int(frame_index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
    state["next_frame_index"] = int(frame_index) + 1
    return frame_bgr


def read_depth_png_m(path: Path) -> np.ndarray:
    try:
        import cv2

        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise FileNotFoundError(path)
    except ModuleNotFoundError:
        from PIL import Image

        if not path.exists():
            raise FileNotFoundError(path)
        depth_mm = np.asarray(Image.open(path), dtype=np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


def choose_hand(payload: dict, allow_missing: bool) -> dict | None:
    hands = payload.get("hands", [])
    hands_with_gripper = [hand for hand in hands if hand.get("gripper") is not None]
    if hands_with_gripper:
        return max(hands_with_gripper, key=lambda hand: float(hand.get("score", 0.0)))
    if allow_missing:
        return None
    raise RuntimeError(f"No gripper prediction at record_index={payload.get('record_index')}")


def gripper_state_from_hand(
    hand: dict | None,
    pose_frame: str,
    camera_to_world: np.ndarray,
    requested_x_offset_m: float = 0.0,
    requested_z_offset_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if hand is None:
        return np.zeros(6, dtype=np.float64), np.asarray([0.0, 0.0, 0.0, 1.0]), 0.0
    gripper = hand["gripper"]
    position = np.asarray(gripper["position_m"], dtype=np.float64)
    rotation = np.asarray(gripper["rotation_camera_gripper"], dtype=np.float64)
    position = apply_gripper_local_offset_delta(
        position,
        rotation,
        x_offset_delta_m=float(requested_x_offset_m) - float(gripper.get("tcp_offset_x_m", 0.0)),
        z_offset_delta_m=(
            None
            if requested_z_offset_m is None
            else float(requested_z_offset_m) - float(gripper.get("tcp_offset_z_m", requested_z_offset_m))
        ),
    )
    opening = float(gripper.get("opening_width_m", 0.0))
    if should_transform_to_output_frame(pose_frame):
        position, rotation = transform_pose(position, rotation, camera_to_world)
    euler_zyx = R.from_matrix(rotation).as_euler("zyx")
    quat = R.from_matrix(rotation).as_quat()
    return np.concatenate([position, euler_zyx]), quat, opening


def apply_gripper_local_offset_delta(
    position: np.ndarray,
    rotation_camera_gripper: np.ndarray,
    x_offset_delta_m: float = 0.0,
    z_offset_delta_m: float | None = None,
) -> np.ndarray:
    """Apply TCP offset deltas along gripper local axes, before world transform."""

    adjusted = np.asarray(position, dtype=np.float64).copy()
    rotation = np.asarray(rotation_camera_gripper, dtype=np.float64)
    if x_offset_delta_m:
        adjusted = adjusted - rotation[:, 0] * float(x_offset_delta_m)
    if z_offset_delta_m is not None:
        if z_offset_delta_m:
            adjusted = adjusted - rotation[:, 2] * float(z_offset_delta_m)
    return adjusted


def keypoints_from_hand(hand: dict | None, pose_frame: str, camera_to_world: np.ndarray) -> np.ndarray:
    if hand is None:
        return np.full((21, 3), np.nan, dtype=np.float64)
    keypoints = np.asarray(hand.get("keypoints_3d_m", []), dtype=np.float64)
    if keypoints.shape != (21, 3):
        return np.full((21, 3), np.nan, dtype=np.float64)
    if should_transform_to_output_frame(pose_frame):
        valid = np.all(np.isfinite(keypoints), axis=1)
        output = keypoints.copy()
        output[valid] = transform_points(output[valid], camera_to_world)
        return output
    return keypoints


def should_transform_to_output_frame(pose_frame: str) -> bool:
    return pose_frame in {"base", "world"}


def transform_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out_position = transform[:3, :3] @ position + transform[:3, 3]
    out_rotation = transform[:3, :3] @ rotation
    return out_position, out_rotation


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_cloud(cloud_rgb: np.ndarray, transform: np.ndarray) -> np.ndarray:
    output = cloud_rgb.copy()
    output[:, :3] = transform_points(output[:, :3], transform)
    return output


def transform_gripper_state(
    pose_euler: np.ndarray,
    quaternion_xyzw: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one gripper pose from the current camera frame into the output frame."""

    pose = np.asarray(pose_euler, dtype=np.float64)
    quat = np.asarray(quaternion_xyzw, dtype=np.float64)
    if pose.shape != (6,) or quat.shape != (4,):
        raise ValueError("Invalid gripper pose shape.")
    position, rotation = transform_pose(
        pose[:3],
        R.from_quat(quat).as_matrix(),
        np.asarray(transform, dtype=np.float64),
    )
    output_rotation = R.from_matrix(rotation)
    return (
        np.concatenate([position, output_rotation.as_euler("zyx")]),
        output_rotation.as_quat(),
    )


def canonicalize_gripper_pose_sequence(
    pose_euler: list[np.ndarray] | np.ndarray,
    quaternion_xyzw: list[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make stored quaternion signs and Euler branches continuous.

    Quaternion ``q`` and ``-q`` encode the same physical rotation, while Euler
    angles wrap at +/-pi.  Canonicalization removes those representation-only
    jumps without filtering or changing any physical pose.
    """

    poses = np.asarray(pose_euler, dtype=np.float64).copy()
    quaternions = np.asarray(quaternion_xyzw, dtype=np.float64).copy()
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"Expected pose_euler [T, 6], got {poses.shape}.")
    if quaternions.shape != (poses.shape[0], 4):
        raise ValueError(
            f"Expected quaternion_xyzw [{poses.shape[0]}, 4], got {quaternions.shape}."
        )
    if poses.shape[0] == 0:
        return poses, quaternions

    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("Gripper quaternion sequence contains invalid rotations.")
    quaternions /= norms[:, None]
    for index in range(1, quaternions.shape[0]):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0

    continuous_euler = R.from_quat(quaternions).as_euler("zyx")
    poses[:, 3:] = np.unwrap(continuous_euler, axis=0)
    return poses, quaternions


def _continuous_parallel_jaw_rotations(
    rotations: R,
) -> tuple[R, dict[str, int]]:
    """Choose the shortest sequence in the parallel-jaw orientation quotient.

    Rotating an ideal parallel-jaw gripper by 180 degrees around its local
    approach axis swaps the two identical fingers and represents the same
    grasp. Hand anatomy occasionally chooses opposite representatives of this
    symmetry on adjacent frames. Dynamic programming removes those persistent
    basis flips while anchoring the first frame to its original representative.
    """

    matrices = rotations.as_matrix()
    frame_count = matrices.shape[0]
    if frame_count == 0:
        return rotations, {"corrected_frame_count": 0, "state_switch_count": 0}

    symmetry = np.diag([-1.0, -1.0, 1.0])
    candidates = np.stack([matrices, matrices @ symmetry], axis=1)
    costs = np.full((frame_count, 2), np.inf, dtype=np.float64)
    previous_state = np.zeros((frame_count, 2), dtype=np.int8)
    costs[0, 0] = 0.0

    for frame_index in range(1, frame_count):
        for current_state in range(2):
            transition_costs = []
            for prior_state in range(2):
                relative = (
                    candidates[frame_index - 1, prior_state].T
                    @ candidates[frame_index, current_state]
                )
                angle = float(R.from_matrix(relative).magnitude())
                transition_costs.append(costs[frame_index - 1, prior_state] + angle * angle)
            best_prior = int(np.argmin(transition_costs))
            costs[frame_index, current_state] = transition_costs[best_prior]
            previous_state[frame_index, current_state] = best_prior

    states = np.zeros(frame_count, dtype=np.int8)
    states[-1] = int(np.argmin(costs[-1]))
    for frame_index in range(frame_count - 1, 0, -1):
        states[frame_index - 1] = previous_state[frame_index, states[frame_index]]
    selected = candidates[np.arange(frame_count), states]
    return R.from_matrix(selected), {
        "corrected_frame_count": int(np.count_nonzero(states)),
        "state_switch_count": int(np.count_nonzero(np.diff(states))),
    }


def _stabilize_rotation_frame_resets(
    rotations: R,
    timestamps_s: np.ndarray,
    *,
    max_angular_speed_deg_s: float,
) -> tuple[R, dict[str, float | int]]:
    """Carry a constant SO(3) correction across implausible frame resets.

    This differs from clipping every large angular velocity. Once WiLoR changes
    anatomical basis, subsequent relative motion is usually useful but remains
    expressed in the new basis. A persistent left correction reconnects that
    complete run to the preceding orientation while preserving all following
    relative rotations.
    """

    matrices = rotations.as_matrix()
    if len(matrices) <= 1 or max_angular_speed_deg_s <= 0.0:
        return rotations, {"reset_count": 0, "largest_rejected_speed_deg_s": 0.0}

    positive_steps = np.diff(timestamps_s)
    positive_steps = positive_steps[positive_steps > 0.0]
    if positive_steps.size == 0:
        raise ValueError("Rotation reset stabilization requires a positive timestamp interval.")
    nominal_dt = float(np.median(positive_steps))
    correction = np.eye(3, dtype=np.float64)
    corrected = np.empty_like(matrices)
    corrected[0] = matrices[0]
    reset_count = 0
    largest_rejected_speed = 0.0
    for frame_index in range(1, len(matrices)):
        candidate = correction @ matrices[frame_index]
        dt = float(timestamps_s[frame_index] - timestamps_s[frame_index - 1])
        if dt <= 0.0:
            dt = nominal_dt
        angle = float(R.from_matrix(corrected[frame_index - 1].T @ candidate).magnitude())
        speed_deg_s = float(np.rad2deg(angle) / dt)
        if speed_deg_s > float(max_angular_speed_deg_s):
            # Make this frame continuous, then retain one constant coordinate
            # correction so future raw relative rotations remain untouched.
            correction = corrected[frame_index - 1] @ matrices[frame_index].T
            candidate = correction @ matrices[frame_index]
            reset_count += 1
            largest_rejected_speed = max(largest_rejected_speed, speed_deg_s)
        corrected[frame_index] = candidate
    return R.from_matrix(corrected), {
        "reset_count": reset_count,
        "largest_rejected_speed_deg_s": largest_rejected_speed,
    }


def _pose_filter_metrics(
    original_positions: np.ndarray,
    original_rotations: R,
    filtered_positions: np.ndarray,
    filtered_rotations: R,
) -> dict[str, float]:
    position_residual = np.linalg.norm(filtered_positions - original_positions, axis=1)
    rotation_residual = (filtered_rotations.inv() * original_rotations).magnitude()
    original_position_steps = np.linalg.norm(np.diff(original_positions, axis=0), axis=1)
    filtered_position_steps = np.linalg.norm(np.diff(filtered_positions, axis=0), axis=1)
    original_rotation_steps = (original_rotations[:-1].inv() * original_rotations[1:]).magnitude()
    filtered_rotation_steps = (filtered_rotations[:-1].inv() * filtered_rotations[1:]).magnitude()

    def percentile(values: np.ndarray, quantile: float) -> float:
        return float(np.percentile(values, quantile)) if values.size else 0.0

    return {
        "translation_residual_rms_mm": float(np.sqrt(np.mean(position_residual**2)) * 1000.0),
        "translation_residual_p95_mm": percentile(position_residual, 95.0) * 1000.0,
        "translation_residual_max_mm": float(position_residual.max(initial=0.0) * 1000.0),
        "rotation_residual_rms_deg": float(np.rad2deg(np.sqrt(np.mean(rotation_residual**2)))),
        "rotation_residual_p95_deg": float(np.rad2deg(percentile(rotation_residual, 95.0))),
        "rotation_residual_max_deg": float(np.rad2deg(rotation_residual.max(initial=0.0))),
        "translation_path_before_m": float(original_position_steps.sum()),
        "translation_path_after_m": float(filtered_position_steps.sum()),
        "rotation_path_before_deg": float(np.rad2deg(original_rotation_steps.sum())),
        "rotation_path_after_deg": float(np.rad2deg(filtered_rotation_steps.sum())),
        "translation_step_p95_before_mm": percentile(original_position_steps, 95.0) * 1000.0,
        "translation_step_p95_after_mm": percentile(filtered_position_steps, 95.0) * 1000.0,
        "rotation_step_p95_before_deg": float(
            np.rad2deg(percentile(original_rotation_steps, 95.0))
        ),
        "rotation_step_p95_after_deg": float(
            np.rad2deg(percentile(filtered_rotation_steps, 95.0))
        ),
    }


def filter_gripper_pose_sequence(
    pose_euler: list[np.ndarray] | np.ndarray,
    quaternion_xyzw: list[np.ndarray] | np.ndarray,
    timestamps_ms: list[float] | np.ndarray,
    *,
    cutoff_hz: float = 4.0,
    order: int = 3,
    max_angular_speed_deg_s: float = 900.0,
    parallel_jaw_symmetry: bool = True,
    preserve_endpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    """Offline zero-phase smoothing of one camera-compensated gripper trajectory.

    Translation is filtered in the episode reference frame. Quaternion
    components are sign-continuous, filtered forward/backward, normalized back
    onto S3, and converted to SO(3). RGB-D clouds, keypoints, and the source
    WiLoR JSONL are deliberately left untouched.
    """

    poses, quaternions = canonicalize_gripper_pose_sequence(pose_euler, quaternion_xyzw)
    frame_count = poses.shape[0]
    if frame_count < 8:
        raise ValueError(
            "SE(3) trajectory filtering requires at least 8 frames, "
            f"got {frame_count}."
        )
    if cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be positive.")
    if order < 1:
        raise ValueError("order must be >= 1.")
    if max_angular_speed_deg_s < 0.0:
        raise ValueError("max_angular_speed_deg_s must be non-negative.")

    timestamps_s = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1) / 1000.0
    if timestamps_s.shape != (frame_count,) or not np.isfinite(timestamps_s).all():
        raise ValueError(
            f"Expected {frame_count} finite trajectory timestamps, got {timestamps_s.shape}."
        )
    timestamp_steps = np.diff(timestamps_s)
    positive_steps = timestamp_steps[timestamp_steps > 0.0]
    if positive_steps.size == 0:
        raise ValueError("Trajectory timestamps contain no positive interval.")
    nominal_dt = float(np.median(positive_steps))
    timestamp_repair_count = int(np.count_nonzero(timestamp_steps <= 0.0))
    if timestamp_repair_count:
        filter_times = timestamps_s[0] + np.arange(frame_count, dtype=np.float64) * nominal_dt
    else:
        filter_times = timestamps_s
    duration_s = float(filter_times[-1] - filter_times[0])
    if duration_s <= 0.0:
        raise ValueError("Trajectory duration must be positive.")
    uniform_times = np.linspace(filter_times[0], filter_times[-1], frame_count)
    sampling_hz = float((frame_count - 1) / duration_s)
    nyquist_hz = 0.5 * sampling_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be below Nyquist frequency {nyquist_hz:.3f} Hz."
        )

    original_positions = poses[:, :3].copy()
    raw_rotations = R.from_quat(quaternions)
    symmetry_metrics = {"corrected_frame_count": 0, "state_switch_count": 0}
    canonical_rotations = raw_rotations
    if parallel_jaw_symmetry:
        canonical_rotations, symmetry_metrics = _continuous_parallel_jaw_rotations(raw_rotations)
    canonical_rotations, reset_metrics = _stabilize_rotation_frame_resets(
        canonical_rotations,
        filter_times,
        max_angular_speed_deg_s=float(max_angular_speed_deg_s),
    )

    canonical_quaternions = canonical_rotations.as_quat()
    for frame_index in range(1, frame_count):
        if float(np.dot(canonical_quaternions[frame_index - 1], canonical_quaternions[frame_index])) < 0.0:
            canonical_quaternions[frame_index] *= -1.0

    uniform_positions = np.column_stack(
        [
            np.interp(uniform_times, filter_times, original_positions[:, axis])
            for axis in range(3)
        ]
    )
    uniform_rotations = Slerp(filter_times, R.from_quat(canonical_quaternions))(uniform_times)
    uniform_quaternions = uniform_rotations.as_quat()
    for frame_index in range(1, frame_count):
        if float(np.dot(uniform_quaternions[frame_index - 1], uniform_quaternions[frame_index])) < 0.0:
            uniform_quaternions[frame_index] *= -1.0

    filter_sos = butter(
        int(order),
        float(cutoff_hz),
        btype="lowpass",
        fs=sampling_hz,
        output="sos",
    )
    try:
        filtered_uniform_positions = sosfiltfilt(filter_sos, uniform_positions, axis=0)
        filtered_uniform_quaternions = sosfiltfilt(filter_sos, uniform_quaternions, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Trajectory is too short for the requested zero-phase filter: "
            f"frames={frame_count}, order={order}."
        ) from exc
    quaternion_norms = np.linalg.norm(filtered_uniform_quaternions, axis=1)
    if np.any(~np.isfinite(quaternion_norms)) or np.any(quaternion_norms < 1e-8):
        raise ValueError("Quaternion low-pass filtering produced an invalid orientation.")
    filtered_uniform_quaternions /= quaternion_norms[:, None]
    filtered_uniform_rotations = R.from_quat(filtered_uniform_quaternions)

    if preserve_endpoints:
        progress = np.linspace(0.0, 1.0, frame_count)
        filtered_uniform_positions += (
            (1.0 - progress)[:, None]
            * (uniform_positions[0] - filtered_uniform_positions[0])[None]
            + progress[:, None]
            * (uniform_positions[-1] - filtered_uniform_positions[-1])[None]
        )
        start_correction = filtered_uniform_rotations[0].inv() * uniform_rotations[0]
        end_correction = filtered_uniform_rotations[-1].inv() * uniform_rotations[-1]
        endpoint_correction = Slerp(
            [0.0, 1.0],
            R.from_quat(np.stack([start_correction.as_quat(), end_correction.as_quat()])),
        )(progress)
        filtered_uniform_rotations = filtered_uniform_rotations * endpoint_correction

    filtered_positions = np.column_stack(
        [
            np.interp(filter_times, uniform_times, filtered_uniform_positions[:, axis])
            for axis in range(3)
        ]
    )
    filtered_rotations = Slerp(uniform_times, filtered_uniform_rotations)(filter_times)
    filtered_quaternions = filtered_rotations.as_quat()
    for frame_index in range(1, frame_count):
        if float(np.dot(filtered_quaternions[frame_index - 1], filtered_quaternions[frame_index])) < 0.0:
            filtered_quaternions[frame_index] *= -1.0

    filtered_poses = poses.copy()
    filtered_poses[:, :3] = filtered_positions
    filtered_poses[:, 3:] = np.unwrap(filtered_rotations.as_euler("zyx"), axis=0)
    metrics: dict[str, float | int | bool | str] = {
        "method": "zero_phase_butterworth_translation_quaternion",
        "cutoff_hz": float(cutoff_hz),
        "order": int(order),
        "sampling_hz": sampling_hz,
        "parallel_jaw_symmetry": bool(parallel_jaw_symmetry),
        "parallel_jaw_corrected_frame_count": int(symmetry_metrics["corrected_frame_count"]),
        "parallel_jaw_state_switch_count": int(symmetry_metrics["state_switch_count"]),
        "max_angular_speed_deg_s": float(max_angular_speed_deg_s),
        "orientation_frame_reset_count": int(reset_metrics["reset_count"]),
        "largest_rejected_angular_speed_deg_s": float(
            reset_metrics["largest_rejected_speed_deg_s"]
        ),
        "preserve_endpoints": bool(preserve_endpoints),
        "timestamp_repair_count": timestamp_repair_count,
        **_pose_filter_metrics(
            original_positions,
            canonical_rotations,
            filtered_positions,
            filtered_rotations,
        ),
    }
    raw_rotation_steps = (raw_rotations[:-1].inv() * raw_rotations[1:]).magnitude()
    metrics["raw_rotation_step_max_deg"] = float(
        np.rad2deg(raw_rotation_steps.max(initial=0.0))
    )
    return filtered_poses, filtered_quaternions, metrics


def transform_keypoints(keypoints: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform finite 3D keypoints while preserving NaN rows for missing detections."""

    output = np.asarray(keypoints, dtype=np.float64).copy()
    if output.shape != (21, 3):
        return output
    valid = np.all(np.isfinite(output), axis=1)
    if np.any(valid):
        output[valid] = transform_points(output[valid], np.asarray(transform, dtype=np.float64))
    return output


def normalized_pixel_rays(
    pixels_uv: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    iterations: int = 8,
) -> np.ndarray:
    """Convert stored pixels to undistorted normalized camera rays."""

    pixels = np.asarray(pixels_uv, dtype=np.float64)
    distorted_x = (pixels[..., 0] - float(intrinsics.ppx)) / float(intrinsics.fx)
    distorted_y = (pixels[..., 1] - float(intrinsics.ppy)) / float(intrinsics.fy)
    coefficients = brown_conrady_coefficients(intrinsics)
    if coefficients is None:
        return np.stack([distorted_x, distorted_y], axis=-1)

    k1, k2, p1, p2, k3 = coefficients
    x = distorted_x.copy()
    y = distorted_y.copy()
    for _ in range(max(1, int(iterations))):
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        radial = np.where(np.abs(radial) > 1e-12, radial, np.nan)
        delta_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        delta_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - delta_x) / radial
        y = (distorted_y - delta_y) / radial
    return np.stack([x, y], axis=-1)


def distort_normalized_points(
    normalized_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    normalized = np.asarray(normalized_xy, dtype=np.float64)
    x = normalized[..., 0]
    y = normalized[..., 1]
    coefficients = brown_conrady_coefficients(intrinsics)
    if coefficients is not None:
        k1, k2, p1, p2, k3 = coefficients
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        distorted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = distorted_x, distorted_y
    return np.stack([x, y], axis=-1)


def brown_conrady_coefficients(
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float, float, float] | None:
    model = "" if intrinsics.model is None else str(intrinsics.model).lower()
    if "." in model:
        model = model.rsplit(".", 1)[-1]
    if model not in {"brown_conrady", "modified_brown_conrady"}:
        return None
    if len(intrinsics.coeffs) < 5:
        return None
    values = tuple(float(value) for value in intrinsics.coeffs[:5])
    if not np.all(np.isfinite(values)):
        return None
    return values  # type: ignore[return-value]


def reproject_rgbd_to_reference_camera(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    source_camera_to_reference: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    reference_height: int,
    reference_width: int,
) -> np.ndarray:
    """Render one RGB-D frame using vectorized matrix operations over every valid depth pixel.

    The input transform uses the convention T_reference<-sourceCamera. Points behind
    the reference camera or outside its image are discarded. When multiple source
    points hit the same reference pixel, the nearest positive reference-frame depth
    wins. Pixels with no projected point remain black.
    """

    color = np.asarray(color_bgr, dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    if color.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth shape mismatch for reprojection: color={color.shape[:2]}, depth={depth.shape}."
        )
    if reference_height <= 0 or reference_width <= 0:
        raise ValueError("Reference image dimensions must be positive.")

    valid = np.isfinite(depth) & (depth > 0.0)
    output = np.zeros((reference_height, reference_width, 3), dtype=np.uint8)
    if not np.any(valid):
        return output

    source_v, source_u = np.nonzero(valid)
    source_z = depth[source_v, source_u].astype(np.float64, copy=False)
    source_rays = normalized_pixel_rays(
        np.column_stack([source_u, source_v]),
        source_intrinsics,
    )
    source_x = source_rays[:, 0] * source_z
    source_y = source_rays[:, 1] * source_z
    source_points = np.column_stack((source_x, source_y, source_z))

    transform = validate_transform_sequence(
        np.asarray(source_camera_to_reference, dtype=np.float64).reshape(4, 4)
    )[0]
    reference_points = transform_points(source_points, transform)
    reference_z = reference_points[:, 2]
    finite_front = np.all(np.isfinite(reference_points), axis=1) & (reference_z > 0.0)
    if not np.any(finite_front):
        return output

    reference_points = reference_points[finite_front]
    reference_z = reference_z[finite_front]
    colors = color[source_v[finite_front], source_u[finite_front]]

    projected_normalized = distort_normalized_points(
        reference_points[:, :2] / reference_z[:, None],
        reference_intrinsics,
    )
    projected_u = np.rint(
        float(reference_intrinsics.fx) * projected_normalized[:, 0]
        + float(reference_intrinsics.ppx)
    ).astype(np.int64)
    projected_v = np.rint(
        float(reference_intrinsics.fy) * projected_normalized[:, 1]
        + float(reference_intrinsics.ppy)
    ).astype(np.int64)

    inside = (
        (projected_u >= 0)
        & (projected_u < reference_width)
        & (projected_v >= 0)
        & (projected_v < reference_height)
    )
    if not np.any(inside):
        return output

    projected_u = projected_u[inside]
    projected_v = projected_v[inside]
    reference_z = reference_z[inside]
    colors = colors[inside]
    flat_indices = projected_v * reference_width + projected_u

    z_buffer = np.full(reference_height * reference_width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_indices, reference_z)
    nearest = reference_z <= z_buffer[flat_indices] + 1e-9

    output.reshape(-1, 3)[flat_indices[nearest]] = colors[nearest]
    return output


def reproject_rgbd_to_reference_rgb(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    source_camera_to_reference: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    reference_height: int,
    reference_width: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Vectorized full-depth reprojection followed by resize and BGR-to-RGB conversion."""

    projected_bgr = reproject_rgbd_to_reference_camera(
        color_bgr=color_bgr,
        depth_m=depth_m,
        source_intrinsics=source_intrinsics,
        source_camera_to_reference=source_camera_to_reference,
        reference_intrinsics=reference_intrinsics,
        reference_height=reference_height,
        reference_width=reference_width,
    )
    resized_bgr = resize_image(projected_bgr, output_width, output_height)
    return resized_bgr[:, :, ::-1].copy()


def make_cloud_rgb(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_points: int,
) -> np.ndarray:
    valid_flat = np.flatnonzero(np.isfinite(depth_m) & (depth_m > 0.0))
    if valid_flat.size == 0:
        output_count = max(0, int(max_points))
        return np.zeros((output_count, 6), dtype=np.float32)

    if max_points > 0:
        if valid_flat.size >= max_points:
            sample = np.linspace(0, valid_flat.size - 1, int(max_points), dtype=np.int64)
            selected_flat = valid_flat[sample]
        else:
            repeat = np.resize(
                np.arange(valid_flat.size, dtype=np.int64),
                int(max_points) - valid_flat.size,
            )
            selected_flat = np.concatenate([valid_flat, valid_flat[repeat]])
    else:
        selected_flat = valid_flat

    width = depth_m.shape[1]
    ys = (selected_flat // width).astype(np.int64)
    xs = (selected_flat % width).astype(np.int64)
    z = depth_m[ys, xs].astype(np.float32)
    rays = normalized_pixel_rays(np.column_stack([xs, ys]), intrinsics).astype(np.float32)
    x = rays[:, 0] * z
    y = rays[:, 1] * z
    colors_rgb = color_bgr[ys, xs, ::-1].astype(np.float32)
    return np.column_stack([x, y, z, colors_rgb]).astype(np.float32, copy=False)


def uniform_sample(xyzrgb: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return xyzrgb
    size = xyzrgb.shape[0]
    if size == 0:
        return np.zeros((count, 6), dtype=np.float64)
    if size >= count:
        idx = np.linspace(0, size - 1, count).astype(np.int64)
        return xyzrgb[idx]
    extra = np.resize(np.arange(size, dtype=np.int64), count - size)
    return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)


def resize_image(color_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return color_bgr
    if color_bgr.shape[1] == width and color_bgr.shape[0] == height:
        return color_bgr
    try:
        import cv2

        return cv2.resize(color_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    except ModuleNotFoundError:
        from PIL import Image

        rgb = color_bgr[:, :, ::-1]
        resized = Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)[:, :, ::-1].copy()


def draw_payload_preview(cv2, image: np.ndarray, payload: dict, intrinsics: CameraIntrinsics) -> None:
    for hand in payload.get("hands", []):
        pts = np.asarray(hand.get("keypoints_2d_px", []), dtype=np.float64)
        if pts.shape == (21, 2):
            for a, b in HAND_EDGES:
                if np.all(np.isfinite(pts[[a, b]])):
                    cv2.line(
                        image,
                        tuple(np.round(pts[a]).astype(int)),
                        tuple(np.round(pts[b]).astype(int)),
                        (0, 220, 180),
                        2,
                    )
            for idx, point in enumerate(pts):
                if np.all(np.isfinite(point)):
                    color = (40, 220, 40) if idx in (4, 8) else (0, 180, 255)
                    cv2.circle(image, tuple(np.round(point).astype(int)), 3, color, -1)

        gripper = hand.get("gripper")
        if gripper is not None:
            draw_gripper(cv2, image, gripper, intrinsics)


def draw_gripper(cv2, image: np.ndarray, gripper: dict, intrinsics: CameraIntrinsics) -> None:
    position = np.asarray(gripper.get("position_m"), dtype=np.float64)
    rotation = np.asarray(gripper.get("rotation_camera_gripper"), dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        return
    origin = project_one(position, intrinsics)
    if origin is None:
        return
    cv2.circle(image, origin, 6, (255, 255, 255), -1)
    length = float(np.clip(float(gripper.get("opening_width_m", 0.04)) * 1.8, 0.055, 0.12))
    axes = ((0, (0, 0, 255), "X"), (1, (0, 255, 0), "Y"), (2, (255, 0, 0), "Z"))
    for axis_idx, color, label in axes:
        end = position + rotation[:, axis_idx] * length
        end_px = project_one(end, intrinsics)
        if end_px is None:
            continue
        cv2.arrowedLine(image, origin, end_px, color, 3, tipLength=0.05)
        cv2.putText(
            image,
            label,
            (end_px[0] + 4, end_px[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def project_one(point_xyz: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[int, int] | None:
    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    u = point[0] / point[2] * intrinsics.fx + intrinsics.ppx
    v = point[1] / point[2] * intrinsics.fy + intrinsics.ppy
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def draw_overlay(
    cv2,
    image: np.ndarray,
    index: int,
    total: int,
    record_index: int,
    camera_pose_sequence: str,
    start_record_index: int | None,
    start_camera_pose_sequence: str | None,
    saved_count: int,
) -> None:
    lines = [
        f"{index + 1}/{total} record_index={record_index} pose={camera_pose_sequence}",
        (
            f"start={start_record_index if start_record_index is not None else '-'} "
            f"start_pose={start_camera_pose_sequence or '-'} saved={saved_count}"
        ),
        "Right/D Left/A | Up/W start | Down/S save | U undo | Q quit",
    ]
    x, y = 12, 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


if __name__ == "__main__":
    main()
