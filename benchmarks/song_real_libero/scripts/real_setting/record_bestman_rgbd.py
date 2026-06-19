#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import REAL_DATA_ROOT
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import REAL_DATA_ROOT


DEFAULT_BESTMAN_ROOT = Path("/home/liusong/ProgramFiles/BestMan")
cv2 = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record aligned RGB-D frames from BestMan RealSense cameras.")
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
    parser.add_argument("--output", default=None, help="Output sequence directory.")
    parser.add_argument("--num-frames", type=int, default=0, help="0 means record until Ctrl-C.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means no duration limit.")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--show", action="store_true")
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
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for --storage compressed.")
    parser.add_argument("--video-fps", type=float, default=15.0, help="FPS metadata for --storage video.")
    parser.add_argument("--video-codec", default="mp4v", help="FourCC for --storage video, e.g. mp4v or avc1.")
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
    args = parser.parse_args()
    if args.space_toggle_recording:
        args.show = True

    global cv2
    import cv2 as cv2_module

    cv2 = cv2_module

    bestman_root = Path(args.bestman_root).resolve()
    if str(bestman_root) not in sys.path:
        sys.path.insert(0, str(bestman_root))

    from Sensor.Camera_Realsense import Camera_Realsense

    cfg = _load_yaml_namespace(Path(args.config))
    output_dir = Path(args.output) if args.output else _default_output_dir(args.camera)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "color"
    color_jpg_dir = output_dir / "color_jpg"
    depth_dir = output_dir / "depth_m"
    depth_png_dir = output_dir / "depth_png"
    if args.storage == "legacy" or args.save_color_png:
        color_dir.mkdir(exist_ok=True)
    if args.storage == "compressed":
        color_jpg_dir.mkdir(exist_ok=True)
    if args.storage == "legacy" or args.save_depth_npy:
        depth_dir.mkdir(exist_ok=True)
    if args.storage in ("compressed", "video") or args.save_depth_png:
        depth_png_dir.mkdir(exist_ok=True)



    camera = None
    frames_file = None
    metadata = None
    color_video = None
    color_video_rel = Path("color.mp4")
    recording = not args.space_toggle_recording
    active_segment_start = 0 if recording and args.space_toggle_recording else None
    recorded_segments: list[dict] = []
    space_events: list[dict] = []
    try:
        camera_name, camera_cfg, camera = _open_camera(Camera_Realsense, cfg.Camera, args.camera)
        init_delay = float(getattr(cfg.Camera, "init_delay", 0.0))
        if init_delay > 0.0:
            time.sleep(init_delay)

        for _ in range(max(0, int(args.warmup_frames))):
            camera.get_rgbd_image()

        probe_frame = camera.get_rgbd_image()
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
            "color_video_path": color_video_rel.as_posix() if args.storage == "video" else None,
            "space_toggle_recording": bool(args.space_toggle_recording),
            "bestman_root": str(bestman_root),
            "config": str(Path(args.config).resolve()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frames_file = open(output_dir / "frames.jsonl", "w", encoding="utf-8")

        start_s = time.time()
        index = 0
        while True:
            if args.num_frames > 0 and index >= args.num_frames:
                break
            if args.duration_s > 0.0 and time.time() - start_s >= args.duration_s:
                break

            frame = camera.get_rgbd_image()
            saved_this_frame = False
            if recording:
                _write_frame_record(
                    output_dir=output_dir,
                    args=args,
                    frame=frame,
                    index=index,
                    frames_file=frames_file,
                    color_video=color_video,
                    color_video_rel=color_video_rel,
                )
                index += 1
                saved_this_frame = True

            if args.show:
                preview = frame.color_bgr.copy()
                status = "REC" if recording else "PAUSED"
                status_color = (0, 0, 255) if recording else (0, 220, 255)
                cv2.putText(
                    preview,
                    f"{status} saved={index} segments={len(recorded_segments)}",
                    (16, 32),
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
                        (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("record RGB-D", preview)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                if args.space_toggle_recording and key == ord(" "):
                    if recording:
                        end_index = index - 1
                        if active_segment_start is not None and end_index >= active_segment_start:
                            recorded_segments.append({"start": int(active_segment_start), "end": int(end_index)})
                        space_events.append(
                            _space_event("pause", end_index, frame.timestamp_ms, frame.frame_number)
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
                            _write_frame_record(
                                output_dir=output_dir,
                                args=args,
                                frame=frame,
                                index=index,
                                frames_file=frames_file,
                                color_video=color_video,
                                color_video_rel=color_video_rel,
                            )
                            index += 1
                        print(f"Started recording at saved frame {active_segment_start}")

    except KeyboardInterrupt:
        pass
    finally:
        if args.space_toggle_recording and recording and active_segment_start is not None and index > active_segment_start:
            recorded_segments.append({"start": int(active_segment_start), "end": int(index - 1)})
            space_events.append(_space_event("stop", index - 1, None, None))
        if color_video is not None:
            color_video.release()
        if frames_file is not None:
            frames_file.close()
        if camera is not None:
            camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    if args.space_toggle_recording:
        _write_segments(output_dir, recorded_segments, space_events, metadata)
    print(f"Saved RGB-D sequence to {output_dir}")


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
) -> None:
    record = {
        "index": index,
        "timestamp_ms": frame.timestamp_ms,
        "frame_number": frame.frame_number,
        "intrinsics": _intrinsics_to_dict(frame.intrinsics),
    }

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


def _write_segments(output_dir: Path, segments: list[dict], space_events: list[dict], metadata: dict | None) -> None:
    segments_arg = ",".join(f"{segment['start']}:{segment['end']}" for segment in segments)
    (output_dir / "segments.txt").write_text(segments_arg + "\n", encoding="utf-8")
    payload = {
        "segments": segments,
        "segments_arg": segments_arg,
        "space_events": space_events,
        "build_command_hint": (
            f"python build_humanhand_hdf5_dataset.py --input {output_dir} "
            f"--segments {segments_arg}"
        )
        if segments_arg
        else None,
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


if __name__ == "__main__":
    main()
