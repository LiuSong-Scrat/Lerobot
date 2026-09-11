"""RLBench-style task artifacts; deliberately independent of control/inference."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

from . import libero_eval_visualization as vis

OUTPUT_DEFAULTS = {
    "save_action_records": True,
    "save_action_chunks": True,
    "save_action_visualizations": False,
    "save_frame_pointclouds": False,
    "save_umi_vis": False,
    "failure_artifacts_only": False,
    "frame_pointcloud_every_n_frames": 2,
    "action_vis_every_n_frames": 32,
    "action_vis_max_points": 50000,
    "action_vis_image_width": 768,
    "action_vis_point_mode": "prob",
    "video_fps": 20,
}


def add_output_arguments(parser):
    import argparse

    for key, default in OUTPUT_DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            flags = [flag, "--save_umi_vis"] if key == "save_umi_vis" else [flag]
            parser.add_argument(
                *flags, action=argparse.BooleanOptionalAction, default=None
            )
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=None)
        else:
            parser.add_argument(flag, choices=("full", "prob"), default=None)
    # Accept old commands, but no output-size setting may resize captured RGB.
    for key in ("video_width", "video_height"):
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=int,
            default=None,
            help="Deprecated/ignored: videos retain captured frame dimensions.",
        )


def configure_output(cfg, args):
    for key in ("video_width", "video_height"):
        if getattr(args, key, None) is not None or key in cfg:
            print(
                f"[output] {key} ignored; videos preserve captured RGB dimensions.",
                flush=True,
            )
        cfg.pop(key, None)
    for key, default in OUTPUT_DEFAULTS.items():
        value = getattr(args, key, None)
        cfg[key] = cfg.get(key, default) if value is None else value
        if isinstance(default, int) and not isinstance(default, bool) and cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    cfg["artifact_layout"] = "rlbench_style_libero_v1"
    cfg["video_size_mode"] = "native_capture"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def snapshot_row(infer, index=0):
    """Copy the exact batch row on its owning inference thread/process."""
    model = getattr(getattr(infer, "policy", None), "model", None)
    snapshot = getattr(model, "last_pointseg_visualization", None)
    if not isinstance(snapshot, dict) or not getattr(
        model, "capture_pointseg_visualization", False
    ):
        return None
    output = {}
    for key in ("point_cloud", "operation_prob", "point_is_pad"):
        value = snapshot.get(key)
        if value is None:
            continue
        expected = 3 if key == "point_cloud" else 2
        if not hasattr(value, "ndim"):
            value = np.asarray(value)
        if value.ndim == expected:
            value = value[index : index + 1]
        elif index != 0:
            return None
        # Slice on GPU BEFORE transferring. Copying the whole batch separately
        # for each worker would multiply visualization traffic by batch size.
        if hasattr(value, "detach"):
            value = value.detach()
            value = value.bool() if key == "point_is_pad" else value.float()
            value = value.cpu().numpy()
        output[key] = np.asarray(value).copy()
    return output


def enable_capture(infer, cfg):
    if cfg.get("save_action_visualizations", False):
        infer.policy.model.capture_pointseg_visualization = True


def inference_payload(action_chunk, snapshot):
    action = np.asarray(action_chunk)
    return (
        action
        if snapshot is None
        else {"action_chunk": action, "artifact_snapshot": snapshot}
    )


class EpisodeArtifacts:
    def __init__(self, task_dir, episode_index, cfg):
        self.root = Path(task_dir)
        self.episode = int(episode_index)
        self.prefix = f"episode_{self.episode:03d}"
        self.cfg = {**OUTPUT_DEFAULTS, **cfg}
        self.chunks = []
        self.visualizations = []
        self.visualization_files = []
        self.umi_visualizations = []
        self.next_visualization_frame = 0
        self.frame_pointcloud_count = 0

    def frame(self, frame_index, point_cloud):
        if not self.cfg["save_frame_pointclouds"]:
            return
        path = (
            self.root
            / "frame_pointclouds"
            / self.prefix
            / f"frame_{frame_index:06d}_model_input_raw.ply"
        )
        vis.write_model_input_ply_binary(
            path,
            point_cloud,
            comments=(
                "coordinate_frame current_virtual_eef",
                "source live_libero_model_input",
                f"video_frame_index {frame_index}",
                f"point_sampling_seed {frame_index}",
                "pointseg_probabilities unavailable_no_model_call_at_every_frame",
            ),
        )
        self.frame_pointcloud_count += 1

    def umi(self, frame_index, model_call, chunk, point_cloud, *, renderer):
        """Save the existing vis_umi_data result for EVERY prediction, headlessly."""
        if not self.cfg["save_umi_vis"]:
            return
        path = (
            self.root
            / "umi_vis"
            / self.prefix
            / f"frame_{frame_index:06d}_model_call_{model_call:04d}.ply"
        )
        renderer(
            chunk,
            point_cloud,
            save_path=path,
            max_points=int(self.cfg.get("trajectory_vis_max_points", 50000)),
        )
        self.umi_visualizations.append(str(path.relative_to(self.root)))

    def chunk(
        self,
        frame_index,
        model_call,
        chunk,
        point_cloud,
        anchor_world,
        env,
        raw_obs,
        snapshot,
        execution_start,
        execution_stop,
        width,
    ):
        stem = f"frame_{frame_index:06d}_model_call_{model_call:04d}"
        if self.cfg["save_action_chunks"]:
            path = self.root / "action_chunks" / self.prefix / (stem + ".npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(chunk, dtype=np.float32))
            self.chunks.append(str(path.relative_to(self.root)))
        if (
            not self.cfg["save_action_visualizations"]
            or frame_index < self.next_visualization_frame
        ):
            return
        while self.next_visualization_frame <= frame_index:
            self.next_visualization_frame += self.cfg["action_vis_every_n_frames"]
        directory = self.root / "action_visualizations" / self.prefix
        mode = self.cfg["action_vis_point_mode"]
        ply = directory / (stem + f"_{mode}.ply")
        png = directory / (stem + ".png")
        self.visualization_files.extend([ply, png])
        arrays = vis.build_action_chunk_ply_cloud(
            point_cloud,
            chunk,
            self.cfg["action_vis_max_points"],
            snapshot,
            mode,
            execution_start,
            execution_stop,
        )
        vis.write_colored_ply(ply, *arrays, point_mode=mode)
        # Prefer agentview as front; no extra rendering and no model input changes.
        camera = (
            "agentview" if "agentview_image" in raw_obs else self.cfg["camera_names"][0]
        )
        observation = vis.libero_front_observation(env, raw_obs, camera)
        cloud = np.asarray(point_cloud)
        # Scene-first / gripper-tail is the model adapter contract. Do not
        # identify gripper samples by red alone: tasks can contain red objects.
        mask = np.zeros(len(cloud), dtype=bool)
        count = min(len(cloud), int(self.cfg.get("gripper_points", 500)))
        if self.cfg.get("add_gripper_cloud", True) and count > 0:
            tail = cloud[-count:]
            if np.isclose(tail[:, 3:6], [255, 0, 0], atol=0.01).all():
                mask[-count:] = True
        current = cloud[mask, :3] @ anchor_world[:3, :3].T + anchor_world[:3, 3]
        boxes = []
        if len(current) and self.cfg.get("gripper_template", "rh20t_v3") in (
            "rh20t_v3",
            "reap",
        ):
            boxes = [
                b @ anchor_world[:3, :3].T + anchor_world[:3, 3]
                for b in vis.canonical_gripper_boxes(
                    width, self.cfg.get("gripper_qpos_max_width", 0.08)
                )
            ]
        targets = anchor_world @ vis.pose9_to_homo_np(np.asarray(chunk)[:, :9])
        vis.draw_action_chunk_on_front_image(
            png,
            observation,
            targets,
            self.cfg["action_vis_image_width"],
            execution_start,
            execution_stop,
            current,
            boxes,
        )
        scores = arrays[2][np.isfinite(arrays[2])]
        self.visualizations.append(
            {
                "frame_index": int(frame_index),
                "model_call": int(model_call),
                "ply": str(ply.relative_to(self.root)),
                "image": str(png.relative_to(self.root)),
                "point_mode": mode,
                "foreground_scores": len(scores),
                "foreground_score_mean": float(scores.mean()) if len(scores) else None,
                "planned_execution_window": [int(execution_start), int(execution_stop)],
                "execution_window_note": "Nominal window; actual holds/early termination/adaptive extension are in execution_index.",
                "target_gripper_geometry": False,
                "current_gripper_geometry": bool(len(current)),
                "current_gripper_points": len(current),
                "projection_camera": camera,
            }
        )

    def finish(self, success):
        if success and self.cfg["failure_artifacts_only"]:
            # Remove only files this recorder created, never a caller's whole tree.
            for path in self.visualization_files:
                path.unlink(missing_ok=True)
            self.visualizations = []
        return {
            "save_umi_vis": self.cfg["save_umi_vis"],
            "umi_vis": self.umi_visualizations,
            "umi_vis_count": len(self.umi_visualizations),
            "action_chunks": self.chunks,
            "action_visualizations": self.visualizations,
            "frame_pointcloud_count": self.frame_pointcloud_count,
            "frame_pointclouds_enabled": self.cfg["save_frame_pointclouds"],
            "frame_pointcloud_dir": (
                f"frame_pointclouds/{self.prefix}"
                if self.cfg["save_frame_pointclouds"]
                else None
            ),
        }


def save_action_records(task_dir, episode_index, result):
    root = Path(task_dir)
    prefix = f"episode_{episode_index:03d}"
    # These are captured online after each successful step, including holds,
    # oracle commands and manual rollback. Unknown model rows are NaN, not fake GT.
    model = np.asarray(result["artifact_model_rows"], dtype=np.float32).reshape(-1, 10)
    controller = np.asarray(
        result["artifact_controller_rows"], dtype=np.float32
    ).reshape(-1, 7)
    index = np.asarray(result["artifact_execution_indices"], dtype=np.int64).reshape(
        -1, 6
    )
    if not len(model) == len(controller) == len(index):
        raise ValueError("Executed model/controller/index lengths differ")
    arrays = {
        "executed_actions": (f"actions/{prefix}_actions.npy", controller),
        "model_chunks": (
            f"model_chunks/{prefix}_model_chunks.npy",
            np.asarray(result["predicted_action_chunks"], dtype=np.float32),
        ),
        "executed_model_actions_relative10": (
            f"executed_action_alignment/{prefix}_executed_model_actions_relative10.npy",
            model,
        ),
        "executed_controller_actions_world7": (
            f"executed_action_alignment/{prefix}_executed_controller_actions_world7.npy",
            controller,
        ),
        "executed_model10_controller7": (
            f"executed_action_alignment/{prefix}_executed_model10_controller7.npy",
            np.concatenate([model, controller], 1),
        ),
        "execution_index": (
            f"executed_action_alignment/{prefix}_execution_index.npy",
            index,
        ),
    }
    for path, array in arrays.values():
        path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    return {key: path for key, (path, _) in arrays.items()}


def finalize_alignment(task_dir, results):
    root = Path(task_dir)
    folder = root / "executed_action_alignment"
    keys = {
        "executed_model_actions_relative10": (10, np.float32),
        "executed_controller_actions_world7": (7, np.float32),
        "executed_model10_controller7": (17, np.float32),
        "execution_index": (6, np.int64),
    }
    rows = sorted(
        (r for r in results if r.get("execution_index")),
        key=lambda r: r["episode_index"],
    )
    if not rows:
        return None
    for key, (width, dtype) in keys.items():
        arrays = [np.load(root / r[key], allow_pickle=False) for r in rows]
        if any(a.ndim != 2 or a.shape[1] != width for a in arrays):
            raise ValueError(f"Invalid alignment shape for {key}")
        np.save(
            folder / f"all_{key}.npy",
            np.concatenate(arrays, 0).astype(dtype, copy=False),
        )
    manifest = {
        "schema": "libero_executed_action_alignment_v1",
        "model_action_columns": [
            "relative_x",
            "relative_y",
            "relative_z",
            "rotation_column_1_x",
            "rotation_column_1_y",
            "rotation_column_1_z",
            "rotation_column_2_x",
            "rotation_column_2_y",
            "rotation_column_2_z",
            "predicted_gripper_width_m",
        ],
        "controller_action_columns": [
            "world_x",
            "world_y",
            "world_z",
            "rotation_vector_x",
            "rotation_vector_y",
            "rotation_vector_z",
            "gripper_open_close_command",
        ],
        "execution_index_columns": [
            "episode_index",
            "model_call_1based",
            "chunk_row_index_0based",
            "execution_phase_code",
            "hold_attempt_1based_or_0",
            "environment_step_index_0based",
        ],
        "execution_phase_codes": {
            "0": "policy",
            "1": "waypoint_hold",
            "2": "manual_rollback",
            "3": "source_oracle",
        },
        "unknown_model_row": "NaN, with chunk_row=-1 for manual rollback/source oracle",
        "episode_summaries": [
            {
                "episode_index": r["episode_index"],
                "execution_index": r["execution_index"],
            }
            for r in rows
        ],
        "note": "Float32 copies of issued world7 OSC actions, not RLBench world8 quaternion actions. Rows repeat for holds; warmup/reset actions are excluded.",
    }
    write_json(folder / "manifest.json", manifest)
    (folder / "README.md").write_text(
        "逐执行步对齐：model relative10 / controller world7 / combined17 / execution_index6。\n字段与NaN语义见 manifest.json；all_* 按 episode 排序，不混合不同任务。\n",
        encoding="utf-8",
    )
    return manifest


def annotated_frames(frames, metadata, task_name, episode_index, success, cfg):
    for i, frame in enumerate(frames):
        # Annotations operate on a copy; preserve each camera's actual H x W.
        frame = np.asarray(frame, dtype=np.uint8).copy()
        record = metadata[i] if i < len(metadata) else {}
        frame = vis.annotate_video_frame(
            frame,
            task_name,
            episode_index,
            i,
            record.get("physics_frame_index", "unknown"),
            record.get("model_call", 0),
            record.get("chunk_row"),
        )
        yield vis.annotate_final_task_result(frame, success)
