#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from smolvla_model_inference import (
    SmolVLA_ModelInference,
    identity_pose9_gripper,
    write_trajectory_ply,
)
from libero_collect_dataset import (
    append_video_frames,
    export_episode_videos,
    resolve_suite_names,
    resolve_task_ids_for_suite,
)
from libero_pointcloud_utils import (
    action_pose9_to_libero,
    add_world_gripper_cloud_to_point_cloud,
    ensure_libero_config,
    get_task_init_states,
    gripper_width_percent_from_scalar,
    make_libero_env,
    observation_to_point_clouds,
    pointcloud_camera_names_from_config,
    render_camera_names_from_config,
)


def load_config(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str) -> Any:
    return cli_value if cli_value is not None else cfg[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the point-cloud SmolVLA policy on LIBERO.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "libero.json")
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default=None)
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def run_episode(
    *,
    infer: SmolVLA_ModelInference,
    env: Any,
    task_language: str,
    init_state: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    control = cfg["control"]
    max_steps = control.get("max_steps")
    raw_obs = env.set_init_state(init_state)
    for _ in range(5):
        raw_obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    infer.policy.reset()
    infer.policy_reset()
    action_queue: list[np.ndarray] = []
    libero_actions: list[np.ndarray] = []
    pose_actions: list[np.ndarray] = []
    rewards: list[float] = []
    video_frames: dict[str, list[np.ndarray]] = {}
    max_steps = int(max_steps or getattr(env, "horizon", 500))
    pc_camera_names = pointcloud_camera_names_from_config(cfg)

    for step in range(max_steps):
        if cfg.get("save_video", False):
            append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
        point_cloud, point_cloud_world, eef_pose = observation_to_point_clouds(
            env,
            raw_obs,
            pc_camera_names,
            int(cfg["observation_height"]),
            int(cfg["observation_width"]),
            int(cfg["num_points"]),
            seed=step,
        )
        if cfg.get("add_gripper_cloud", True):
            point_cloud = add_world_gripper_cloud_to_point_cloud(
                point_cloud_world,
                eef_pose,
                gripper_width_percent_from_scalar(
                    float(eef_pose[-1]),
                    max_physical_width=float(cfg.get("gripper_qpos_max_width", 0.04)),
                ),
                total_points=int(cfg["num_points"]),
                gripper_points=int(cfg.get("gripper_points", 500)),
                gripper_len=float(cfg.get("gripper_len", 0.06)),
                gripper_template=str(cfg.get("gripper_template", "reap")),
                seed=step,
                drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
                shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
            )
        if not action_queue:
            chunk = infer.predict_action_chunk_obs(
                {"point_cloud": point_cloud, "state": identity_pose9_gripper()},
                task=task_language,
                postprocess=True,
                state_pose_mode="identity",
            )[0].detach().cpu().numpy()
            action_queue.extend(list(chunk))

        pose_action = np.asarray(action_queue.pop(0), dtype=np.float32)
        libero_action = action_pose9_to_libero(
            pose_action,
            float(control["trans_scale"]),
            float(control["rot_scale"]),
            float(control["gripper_threshold"]),
        )
        raw_obs, reward, done, _ = env.step(libero_action)
        success = bool(env.check_success())
        libero_actions.append(libero_action)
        pose_actions.append(pose_action)
        rewards.append(float(reward))
        if done or success:
            break

    return {
        "success": bool(env.check_success()),
        "steps": step + 1,
        "sum_reward": float(np.sum(rewards)),
        "libero_actions": np.asarray(libero_actions, dtype=np.float32),
        "pose_actions": np.asarray(pose_actions, dtype=np.float32),
        "video_frames": video_frames,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["policy_path"] = cfg_get(cfg, args.policy_path, "policy_path")
    cfg["policy_repo_id"] = args.policy_repo_id if args.policy_repo_id is not None else cfg.get("policy_repo_id")
    suite_names = resolve_suite_names(args.suite, cfg)
    cfg["suites"] = suite_names
    cfg["all_tasks"] = bool(args.all_tasks if args.all_tasks is not None else cfg.get("all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes"))
    cfg["device"] = cfg_get(cfg, args.device, "device")
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points"))
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video"))
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
    output_dir = Path(args.output_dir or cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark

    infer = SmolVLA_ModelInference(
        policy_path=cfg["policy_path"],
        policy_repo_id=cfg.get("policy_repo_id"),
        device=cfg["device"],
    )

    all_results = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in suite_names:
        suite_cls = benchmark_dict[suite_name]
        suite = suite_cls()
        task_ids = resolve_task_ids_for_suite(
            suite_name=suite_name,
            task_count=len(suite.tasks),
            cli_task_ids=args.task_id,
            cfg=cfg,
        )
        for task_id in task_ids:
            env, task = make_libero_env(
                suite,
                int(task_id),
                int(cfg["observation_height"]),
                int(cfg["observation_width"]),
                render_camera_names_from_config(cfg),
            )
            init_states = get_task_init_states(suite, int(task_id))
            task_results = []
            try:
                for episode_idx in range(cfg["episodes"]):
                    episode_dir = output_dir / suite_name / f"task_{int(task_id):03d}" / f"episode_{episode_idx:03d}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    result = run_episode(
                        infer=infer,
                        env=env,
                        task_language=task.language,
                        init_state=init_states[episode_idx % len(init_states)],
                        cfg=cfg,
                    )
                    np.save(episode_dir / "libero_actions.npy", result["libero_actions"])
                    np.save(episode_dir / "pose_actions.npy", result["pose_actions"])
                    if cfg.get("save_trajectory", True):
                        write_trajectory_ply(episode_dir / "trajectory.ply", result["pose_actions"])
                    record = {
                        "episode_index": episode_idx,
                        "demo_name": "rollout",
                        "video_dir_name": episode_dir.name,
                    }
                    video_paths = export_episode_videos(result, episode_dir.parent, record, cfg)
                    task_results.append(
                        {
                            k: v
                            for k, v in result.items()
                            if not isinstance(v, np.ndarray) and k != "video_frames"
                        }
                    )
                    if video_paths:
                        task_results[-1]["videos"] = video_paths
                    print(
                        f"{suite_name} task={task_id} episode={episode_idx} "
                        f"success={result['success']} steps={result['steps']} reward={result['sum_reward']:.3f}"
                    )
            finally:
                env.close()
            success_rate = float(np.mean([item["success"] for item in task_results])) if task_results else 0.0
            all_results.append(
                {
                    "suite": suite_name,
                    "task_id": int(task_id),
                    "task_name": task.name,
                    "task_language": task.language,
                    "episodes": task_results,
                    "success_rate": success_rate,
                }
            )

    summary = {
        "created_unix_s": time.time(),
        "policy_path": cfg["policy_path"],
        "suites": suite_names,
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": pointcloud_camera_names_from_config(cfg),
        "render_camera_names": render_camera_names_from_config(cfg),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "results": all_results,
        "overall_success_rate": float(
            np.mean([item["success_rate"] for item in all_results]) if all_results else 0.0
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
