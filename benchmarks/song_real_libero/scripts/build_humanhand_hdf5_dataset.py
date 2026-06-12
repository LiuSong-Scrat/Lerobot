#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


BESTMAN_ROOT = Path(__file__).resolve().parent
HANDPOSE_ROOT = Path("/home/liusong/ProgramFiles/HandPoseExtraction")
DEFAULT_POINTS_NUM = 640*480
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
    parser.add_argument("--input", required=True, help="RGB-D directory produced by record_bestman_rgbd.py.")
    parser.add_argument("--jsonl", default=None, help="WiLoR JSONL. Defaults to <input>/handpose_wilor.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=str(BESTMAN_ROOT / "Dataset/dataset/humanhand_offline"),
        help="Directory where episode_*.hdf5 files will be written.",
    )
    parser.add_argument("--task", default="humanhand_offline")
    parser.add_argument("--camera-names", default="overhead,hand")
    parser.add_argument("--run-inference", action="store_true", help="Run HandPoseExtraction offline WiLoR first.")
    parser.add_argument("--handpose-root", default=str(HANDPOSE_ROOT))
    parser.add_argument("--wilor-repo", default=str(HANDPOSE_ROOT / "external/WiLoR"))
    parser.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
    parser.add_argument("--model-cfg", default="pretrained_models/model_config.yaml")
    parser.add_argument("--detector", default="pretrained_models/detector.pt")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force-handedness", choices=("left", "right"), default=None)
    parser.add_argument("--fusion-mode", choices=("model-depth", "keypoint-depth"), default="model-depth")
    parser.add_argument("--gripper-x-offset-cm", type=float, default=1.5)
    parser.add_argument("--gripper-z-offset-cm", type=float, default=3.5)
    parser.add_argument("--reuse-jsonl", action="store_true", help="Do not run inference even if --run-inference is set.")
    parser.add_argument("--show-inference", action="store_true", help="Show OpenCV preview while running offline WiLoR.")
    parser.add_argument("--no-inference-progress", action="store_true", help="Do not show progress while running offline WiLoR.")
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
    if args.transform_to_world:
        args.pose_frame = "world"
        if args.camera_to_world_preset is None and args.camera_to_world_matrix is None:
            args.camera_to_world_preset = "humanhand_l515"

    input_dir = Path(args.input).resolve()
    jsonl_path = Path(args.jsonl).resolve() if args.jsonl else input_dir / "handpose_wilor.jsonl"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_inference and not args.reuse_jsonl:
        run_offline_wilor(args, input_dir, jsonl_path)

    frame_records = load_frame_records(input_dir)
    metadata = load_metadata(input_dir)
    payloads = load_payloads(jsonl_path)
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


def load_payloads(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    payloads = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payloads.append(json.loads(line))
    return payloads


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
            start_record_index,
            len(saved_paths),
        )
        cv2.imshow(args.window_name, preview)
        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (83, 2555904, 65363, ord("d"), ord("D")):
            index = min(index + 1, len(samples) - 1)
        elif key in (81, 2424832,65361, ord("a"), ord("A")):
            index = max(index - 1, 0)
        elif key in (82, 2490368,65362, ord("w"), ord("W")):
            start_record_index = int(frame_record.get("index", index))
            print(f"Segment start = frame {start_record_index}")
        elif key in (84, 2621440,65364, ord("s"), ord("S")):
            end_record_index = int(frame_record.get("index", index))
            if start_record_index is None:
                print("Set a start frame first with Up/W.")
                continue
            segment = Segment(
                start=min(start_record_index, end_record_index),
                end=max(start_record_index, end_record_index),
            )
            saved_paths.append(
                save_segment_hdf5(
                    segment,
                    samples,
                    input_dir,
                    metadata,
                    output_dir,
                    camera_names,
                    camera_to_world,
                    args,
                )
            )
            start_record_index = None
        elif key in (ord("r"), ord("R")):
            start_record_index = None
            print("Cleared current segment start.")
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

    print(f"Saving {len(segments)} static segments with {worker_count} worker processes.", flush=True)
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

    images = []
    clouds = []
    qpos = []
    pose_eular = []
    action = []
    eff_angular = []
    gripper_quat_xyzw = []
    keypoints_3d_m = []
    source_indices = []
    timestamps_ms = []

    start_s = time.time()
    total_frames = len(selected)
    for frame_offset, (frame_record, payload) in enumerate(selected, start=1):
        color_bgr, depth_m, intrinsics = load_rgbd_frame(input_dir, frame_record, metadata)
        resized_rgb = resize_image(color_bgr, args.image_width, args.image_height)[:, :, ::-1].copy()
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
        if should_transform_to_output_frame(args.pose_frame):
            cloud = transform_cloud(cloud, camera_to_world)

        images.append(resized_rgb)
        clouds.append(cloud)
        qpos.append(np.zeros(7, dtype=np.float64))
        pose_eular.append(pose)
        action.append(np.zeros(7, dtype=np.float64))
        eff_angular.append(np.asarray([opening], dtype=np.float64))
        gripper_quat_xyzw.append(quat)
        keypoints_3d_m.append(joints)
        source_indices.append(int(frame_record.get("index", len(source_indices))))
        timestamps_ms.append(float(payload.get("timestamp_ms") or frame_record.get("timestamp_ms") or np.nan))
        if progress_enabled and (frame_offset == 1 or frame_offset == total_frames or frame_offset % 25 == 0):
            print_progress(
                f"Preparing segment {segment.start}:{segment.end}",
                frame_offset,
                total_frames,
                start_s,
            )
    if progress_enabled:
        sys.stderr.write("\n")
        sys.stderr.flush()

    path = output_path if output_path is not None else next_episode_path(output_dir)
    with h5py.File(path, "x", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs["sim"] = False
        root.attrs["task"] = args.task
        root.attrs["source_rgbd_dir"] = str(input_dir)
        root.attrs["source_jsonl"] = str(Path(args.jsonl).resolve() if args.jsonl else input_dir / "handpose_wilor.jsonl")
        root.attrs["segment_start_record_index"] = segment.start
        root.attrs["segment_end_record_index"] = segment.end
        root.attrs["pose_frame"] = args.pose_frame
        root.attrs["camera_to_world"] = camera_to_world
        root.attrs["gripper_x_offset_cm"] = float(args.gripper_x_offset_cm)
        root.attrs["gripper_z_offset_cm"] = float(args.gripper_z_offset_cm)
        root.attrs["image_color_format"] = "rgb"
        root.attrs["cloud_color_format"] = "rgb_0_255"
        root.attrs["max_points"] = int(args.max_points)
        root.attrs["camera_names"] = json.dumps(camera_names, ensure_ascii=False)
        root.attrs["camera_datasets_hardlinked"] = len(camera_names) > 1

        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        cloud_grp = obs.create_group("cloud_rgb")
        first_camera = camera_names[0]
        first_image = image_grp.create_dataset(
            first_camera,
            data=np.asarray(images, dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )
        first_cloud = cloud_grp.create_dataset(
            first_camera,
            data=np.asarray(clouds, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        for name in camera_names[1:]:
            image_grp[name] = first_image
            cloud_grp[name] = first_cloud

        obs.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float64))
        obs.create_dataset("pose_eular", data=np.asarray(pose_eular, dtype=np.float64))
        obs.create_dataset("eff_angular", data=np.asarray(eff_angular, dtype=np.float64))
        obs.create_dataset("gripper_quat_xyzw", data=np.asarray(gripper_quat_xyzw, dtype=np.float64))
        obs.create_dataset("keypoints_3d_m", data=np.asarray(keypoints_3d_m, dtype=np.float64))
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
        z_offset_delta_m=None
        if requested_z_offset_m is None
        else float(requested_z_offset_m) - float(gripper.get("tcp_offset_z_m", requested_z_offset_m)),
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
            repeat = np.resize(np.arange(valid_flat.size, dtype=np.int64), int(max_points) - valid_flat.size)
            selected_flat = np.concatenate([valid_flat, valid_flat[repeat]])
    else:
        selected_flat = valid_flat

    width = depth_m.shape[1]
    ys = (selected_flat // width).astype(np.int64)
    xs = (selected_flat % width).astype(np.int64)
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - np.float32(intrinsics.ppx)) * z / np.float32(intrinsics.fx)
    y = (ys.astype(np.float32) - np.float32(intrinsics.ppy)) * z / np.float32(intrinsics.fy)
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
                    cv2.line(image, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), (0, 220, 180), 2)
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
        cv2.putText(image, label, (end_px[0] + 4, end_px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


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
    start_record_index: int | None,
    saved_count: int,
) -> None:
    lines = [
        f"{index + 1}/{total} record_index={record_index}",
        f"start={start_record_index if start_record_index is not None else '-'} saved={saved_count}",
        "Right/D Left/A | Up/W start | Down/S save | U undo | Q quit",
    ]
    x, y = 12, 24
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


if __name__ == "__main__":
    main()
