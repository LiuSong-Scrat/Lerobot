#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "local.json"
DEFAULT_LIBERO_CONFIG = PROJECT_ROOT / "configs" / "libero.json"
ALL_STAGES = ("collect", "hdf5", "gripper", "check", "convert", "cache", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Song real-robot/LIBERO benchmark pipeline stages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--libero-config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument(
        "--stage",
        choices=("collect", "hdf5", "gripper", "check", "convert", "cache", "train", "infer", "libero_collect", "libero", "all"),
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--task", default=None)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--train-output-dir", default=None)
    parser.add_argument("--obs-path", default=None)
    parser.add_argument("--ply-path", default=None)
    parser.add_argument("--demo-root", default=None)
    parser.add_argument("--demo-file", action="append", default=None)
    parser.add_argument("--libero-output-root", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def get_path(cfg: dict[str, Any], key: str) -> str:
    return str(Path(cfg["paths"][key]).expanduser())


def maybe_append(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def command_for_stage(stage: str, cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    task = args.task or cfg.get("task", "")
    py = args.python

    if stage == "collect":
        collect = cfg["collect"]
        cmd = [
            py,
            str(SCRIPTS_DIR / "record_bestman_rgbd.py"),
            "--bestman-root",
            cfg["roots"]["bestman"],
            "--config",
            cfg["paths"]["bestman_config"],
            "--camera",
            collect["camera"],
            "--output",
            get_path(cfg, "raw_rgbd_dir"),
            "--storage",
            collect["storage"],
            "--warmup-frames",
            str(collect["warmup_frames"]),
            "--num-frames",
            str(collect["num_frames"]),
            "--duration-s",
            str(collect["duration_s"]),
        ]
        bool_flag(cmd, bool(collect.get("show", False)), "--show")
        return cmd

    if stage == "hdf5":
        hdf5 = cfg["hdf5"]
        cmd = [
            py,
            str(SCRIPTS_DIR / "build_humanhand_hdf5_dataset.py"),
            "--input",
            get_path(cfg, "raw_rgbd_dir"),
            "--output-dir",
            get_path(cfg, "raw_hdf5_dir"),
            "--task",
            task,
            "--camera-names",
            hdf5["camera_names"],
            "--max-points",
            str(hdf5["max_points"]),
            "--pose-frame",
            hdf5["pose_frame"],
            "--camera-to-world-preset",
            hdf5["camera_to_world_preset"],
        ]
        segments = args.segments if args.segments is not None else hdf5.get("segments", "")
        maybe_append(cmd, "--segments", segments)
        bool_flag(cmd, bool(hdf5.get("run_inference", False)), "--run-inference")
        bool_flag(cmd, bool(hdf5.get("no_interactive", False)), "--no-interactive")
        return cmd

    if stage == "gripper":
        gripper = cfg["gripper"]
        cmd = [
            py,
            str(SCRIPTS_DIR / "add_gripper_cloud_to_hdf5.py"),
            "--input-dir",
            get_path(cfg, "raw_hdf5_dir"),
            "--output-dir",
            get_path(cfg, "with_gripper_hdf5_dir"),
            "--gripper-points",
            str(gripper["gripper_points"]),
            "--drop-strategy",
            gripper["drop_strategy"],
            "--num-workers",
            str(gripper["num_workers"]),
        ]
        bool_flag(cmd, bool(gripper.get("overwrite", True)), "--overwrite")
        return cmd

    if stage == "check":
        check = cfg["check"]
        cmd = [
            py,
            str(SCRIPTS_DIR / "filter_continuous_hdf5.py"),
            get_path(cfg, "with_gripper_hdf5_dir"),
            "--output-dir",
            get_path(cfg, "clean_hdf5_dir"),
            "--threshold",
            str(check["threshold"]),
            "--mode",
            check["mode"],
        ]
        bool_flag(cmd, bool(check.get("overwrite", True)), "--overwrite")
        return cmd

    if stage == "convert":
        convert = cfg["convert"]
        output_root = args.dataset_root or get_path(cfg, "lerobot_dataset")
        cmd = [
            py,
            str(SCRIPTS_DIR / "song_lerobot_from_hdf5.py"),
            "--hdf5-folder",
            get_path(cfg, "clean_hdf5_dir"),
            "--output-root",
            output_root,
            "--repo-id",
            convert["repo_id"],
            "--fps",
            str(convert["fps"]),
            "--workers",
            str(convert["workers"]),
            "--task",
            task,
            "--point-cloud-key",
            convert["point_cloud_key"],
            "--num-points",
            str(convert.get("num_points", 10000)),
            "--gripper-points",
            str(cfg.get("gripper", {}).get("gripper_points", convert.get("gripper_points", 500))),
            "--point-cloud-storage",
            convert.get("point_cloud_storage", "zarr"),
            "--overwrite",
        ]
        return cmd

    if stage == "cache":
        cache = cfg["cache"]
        dataset_root = args.dataset_root or get_path(cfg, "lerobot_dataset")
        cache_dir = args.cache_dir or get_path(cfg, "pointseg_cache")
        return [
            py,
            str(SCRIPTS_DIR / "song_cache_pointseg_samples.py"),
            "--dataset.repo_id",
            dataset_root,
            "--output-dir",
            cache_dir,
            "--current-points",
            str(cache["current_points"]),
            "--future-points",
            str(cache["future_points"]),
            "--batch-size",
            str(cache["batch_size"]),
            "--num-workers",
            str(cache["num_workers"]),
            "--nn-chunk-size",
            str(cache["nn_chunk_size"]),
            "--shard-size",
            str(cache["shard_size"]),
            "--storage-dtype",
            cache["storage_dtype"],
            "--overwrite",
        ]

    if stage == "train":
        train = cfg["train"]
        dataset_root = args.dataset_root or get_path(cfg, "lerobot_dataset")
        cache_dir = args.cache_dir or get_path(cfg, "pointseg_cache")
        output_dir = args.train_output_dir or get_path(cfg, "train_output")
        policy_path = args.policy_path or cfg["paths"]["policy_checkpoint"]
        cmd = [py, str(SCRIPTS_DIR / "train_song_benchmark.py")]
        if train.get("mode", "resume") == "fresh":
            cmd.append("--policy.type=smolvla")
        else:
            cmd.append(f"--policy.path={policy_path}")
        cmd.extend(
            [
                "--policy.push_to_hub=false",
                f"--dataset.repo_id={dataset_root}",
                f"--pointseg_sample_cache_dir={cache_dir}",
                f"--policy.vlm_model_name={cfg['paths']['vlm_model']}",
                "--policy.load_vlm_weights=false",
                f"--batch_size={train['batch_size']}",
                f"--steps={train['steps']}",
                f"--log_freq={train['log_freq']}",
                f"--output_dir={output_dir}",
                f"--job_name={train['job_name']}",
                f"--policy.device={train['device']}",
                f"--wandb.enable={str(train['wandb_enable']).lower()}",
                "--wandb.disable_artifact=true",
                f"--save_freq={train['save_freq']}",
                f"--eval_freq={train['eval_freq']}",
                f"--num_workers={train['num_workers']}",
                "--policy.pointseg_enable=true",
                "--policy.pointseg_backbone_type=litept",
                "--policy.pointseg_grid_size=0.01",
                "--policy.pointseg_feature_dim=64",
                "--policy.pointseg_aux_loss_weight=0.002",
                "--policy.pointseg_foreground_ratio=0.08",
                "--policy.pointseg_background_ratio=0.08",
                "--policy.pointseg_min_foreground_points=4000",
                "--policy.pointseg_min_background_points=0",
                "--policy.pointseg_use_temporal_priors_as_input=false",
                "--policy.pointseg_use_pseudo_selection=false",
                "--policy.worldflow_enable=false",
                "--policy.worldflow_se3_head_enable=false",
                "--policy.se3_enable=false",
                "--policy.se3_final_correction_enable=false",
            ]
        )
        return cmd

    if stage == "infer":
        infer = cfg["infer"]
        obs_path = args.obs_path if args.obs_path is not None else infer.get("obs_path", "")
        ply_path = args.ply_path if args.ply_path is not None else infer.get("ply_path", "")
        if not obs_path and not ply_path:
            raise ValueError("Stage infer requires --obs-path or --ply-path, or a non-empty infer path in config.")
        cmd = [
            py,
            str(SCRIPTS_DIR / "smolvla_model_inference.py"),
            f"--policy.path={args.policy_path or cfg['paths']['policy_checkpoint']}",
            "--task",
            task,
            "--device",
            infer["device"],
            "--num-points",
            str(infer["num_points"]),
            "--output-dir",
            get_path(cfg, "infer_output"),
        ]
        maybe_append(cmd, "--obs.path", obs_path)
        maybe_append(cmd, "--ply.path", ply_path)
        bool_flag(cmd, bool(infer.get("visualize", False)), "--visualize")
        return cmd

    if stage == "libero":
        cmd = [py, str(SCRIPTS_DIR / "libero_pointcloud_eval.py"), "--config", str(args.libero_config)]
        maybe_append(cmd, "--suite", args.suite)
        maybe_append(cmd, "--task-id", args.task_id)
        maybe_append(cmd, "--episodes", args.episodes)
        maybe_append(cmd, "--policy.path", args.policy_path)
        return cmd

    if stage == "libero_collect":
        cmd = [py, str(SCRIPTS_DIR / "libero_collect_dataset.py"), "--config", str(args.libero_config)]
        maybe_append(cmd, "--suite", args.suite)
        maybe_append(cmd, "--task-id", args.task_id)
        maybe_append(cmd, "--episodes", args.episodes)
        maybe_append(cmd, "--demo-root", args.demo_root)
        maybe_append(cmd, "--output-root", args.libero_output_root)
        maybe_append(cmd, "--num-workers", args.num_workers)
        if args.demo_file:
            for demo_file in args.demo_file:
                maybe_append(cmd, "--demo-file", demo_file)
        return cmd

    raise ValueError(f"Unsupported stage: {stage}")


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    stages = ALL_STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        run_command(command_for_stage(stage, cfg, args), args.dry_run)


if __name__ == "__main__":
    main()
