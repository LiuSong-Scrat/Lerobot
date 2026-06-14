#!/usr/bin/env python
"""Direct-IK LIBERO point-cloud evaluator.

This evaluator keeps the same observation -> point cloud -> SmolVLA chunk path
as libero_pointcloud_eval.py, but bypasses robosuite OSC rollout. Each selected
target pose is solved as a MuJoCo site IK problem and written directly into
qpos/qvel, so one benchmark step jumps to the requested end-effector pose and
gripper width.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from libero_collect_dataset import (
    append_video_frames,
    export_episode_videos,
    resolve_suite_names,
    resolve_task_ids_for_suite,
)
from libero_pointcloud_eval import (
    BENCHMARK_ROOT,
    VSCODE_DEBUG_DEFAULT_ENV,
    add_timing,
    build_eval_summary,
    canonical_gripper_width,
    cfg_get,
    configure_mujoco_render_backend,
    correct_rim_grasp_target_world,
    current_controller_eef_world,
    episode_result_record,
    goal_predicate_status,
    gripper_close_threshold_for_task,
    gripper_grasp_status,
    json_safe,
    make_task_summary,
    matrix_to_pose9,
    model_world_pose_to_controller_world,
    planned_world_poses_from_umi_chunk,
    pose_error_to_target,
    render_onscreen_env,
    summarize_timings,
    write_eval_reports,
    write_json_atomic,
)
from libero_pointcloud_utils import (
    add_world_gripper_cloud_to_point_cloud,
    eef_pose9_gripper_from_obs,
    ensure_libero_config,
    fast_inverse_homogeneous,
    get_task_init_states,
    gripper_scalar,
    gripper_width_percent_from_scalar,
    make_libero_env,
    observation_to_point_clouds,
    observation_to_world_point_cloud,
    pointcloud_camera_names_from_config,
    pose9_to_homo_np,
    render_camera_names_from_config,
)
from smolvla_model_inference import SmolVLA_ModelInference, identity_pose9_gripper, write_trajectory_ply


DIRECT_DEBUG_DEFAULT_ARGS = [
    "--config",
    str(BENCHMARK_ROOT / "configs" / "libero.json"),
    "--policy.path",
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/train_libero_fresh/checkpoints/last/pretrained_model",
    "--suite",
    "libero_spatial",
    "--suite",
    "libero_object",
    "--all-tasks",
    "--episodes",
    "10",
    "--action-index",
    "0",
    "--exec-action-steps",
    "16",
    "--no-replan-every-step",
    "--save-video",
    "--render-mode",
    "offscreen",
    "--output-dir",
    "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/eval_libero_4suite_direct_ik",
    "--control-freq",
    "20",
]


def load_config(path: Path) -> dict[str, Any]:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        for key, value in VSCODE_DEBUG_DEFAULT_ENV.items():
            os.environ.setdefault(key, value)
        argv = list(DIRECT_DEBUG_DEFAULT_ARGS)
        print("[debug] No CLI args detected; using direct-IK debug defaults.")

    parser = argparse.ArgumentParser(description="Evaluate point-cloud SmolVLA on LIBERO with direct IK state writes.")
    parser.add_argument("--control-freq", type=float, default=None)
    parser.add_argument("--config", type=Path, default=BENCHMARK_ROOT / "configs" / "libero.json")
    parser.add_argument("--policy.path", "--policy_path", dest="policy_path", default=None)
    parser.add_argument("--policy.repo_id", "--policy_repo_id", dest="policy_repo_id", default=None)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-mode", choices=("offscreen", "onscreen", "viewer3d"), default=None)
    parser.add_argument("--headed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-camera", default=None)
    parser.add_argument("--render-every-n-steps", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=None)
    parser.add_argument("--action-index", type=int, default=None)
    parser.add_argument("--exec-action-steps", type=int, default=None)
    parser.add_argument("--replan-every-step", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--gripper-close-threshold", type=float, default=None)
    parser.add_argument("--keyboard-vis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keyboard-vis-mode", choices=("ply", "window"), default=None)
    parser.add_argument("--ik-max-iters", type=int, default=80)
    parser.add_argument("--ik-pos-tolerance", type=float, default=0.002)
    parser.add_argument("--ik-rot-tolerance", type=float, default=0.03)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--ik-step-size", type=float, default=0.75)
    parser.add_argument("--ik-max-dq", type=float, default=0.10)
    parser.add_argument("--direct-gripper-mode", choices=("predicted_width", "binary"), default="predicted_width")
    parser.add_argument("--direct-attach-objects", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def unwrap_libero_env(env: Any) -> Any:
    inner_env = env
    while hasattr(inner_env, "env"):
        next_env = inner_env.env
        if next_env is inner_env:
            break
        inner_env = next_env
    return inner_env


def refresh_observation_after_state_write(env: Any) -> dict[str, Any]:
    env.sim.forward()
    for fn_name, kwargs in (
        ("check_success", {}),
        ("_post_process", {}),
        ("_update_observables", {"force": True}),
    ):
        fn = getattr(env, fn_name, None)
        if callable(fn):
            try:
                fn(**kwargs)
            except TypeError:
                fn()
            except Exception:
                pass
    inner_env = unwrap_libero_env(env)
    get_obs = getattr(inner_env, "_get_observations", None)
    if callable(get_obs):
        return get_obs()
    raise AttributeError("Could not refresh LIBERO observations after direct state write.")


def _joint_limits_for_robot(robot: Any, sim: Any) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
    limits = np.asarray(sim.model.jnt_range[joint_ids], dtype=np.float64)
    low = limits[:, 0]
    high = limits[:, 1]
    valid = np.isfinite(low) & np.isfinite(high) & (high > low)
    low = np.where(valid, low, -np.inf)
    high = np.where(valid, high, np.inf)
    return low, high


def _site_pose(sim: Any, site_name: str) -> np.ndarray:
    site_id = sim.model.site_name2id(site_name)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    pose[:3, :3] = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return pose


def _ik_pose_error(current_world: np.ndarray, target_world: np.ndarray) -> tuple[np.ndarray, float, float]:
    pos_error_vec = np.asarray(target_world[:3, 3] - current_world[:3, 3], dtype=np.float64)
    rot_delta = np.asarray(target_world[:3, :3] @ current_world[:3, :3].T, dtype=np.float64)
    rot_error_vec = R.from_matrix(rot_delta).as_rotvec().astype(np.float64)
    return np.concatenate([pos_error_vec, rot_error_vec]), float(np.linalg.norm(pos_error_vec)), float(np.linalg.norm(rot_error_vec))


def solve_site_ik(
    *,
    env: Any,
    target_world: np.ndarray,
    max_iters: int,
    pos_tolerance: float,
    rot_tolerance: float,
    damping: float,
    step_size: float,
    max_dq: float,
) -> dict[str, Any]:
    robot = env.robots[0]
    sim = env.sim
    controller = robot.controller
    site_name = controller.eef_name
    qpos_index = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    qvel_index = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    q_low, q_high = _joint_limits_for_robot(robot, sim)
    target_world = np.asarray(target_world, dtype=np.float64)
    damping = max(float(damping), 1e-8)
    step_size = float(step_size)
    max_dq = max(float(max_dq), 1e-8)

    last_pos_error = np.inf
    last_rot_error = np.inf
    success = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iters)) + 1):
        sim.forward()
        current_world = _site_pose(sim, site_name)
        error, last_pos_error, last_rot_error = _ik_pose_error(current_world, target_world)
        if last_pos_error <= pos_tolerance and last_rot_error <= rot_tolerance:
            success = True
            break

        jac_pos = np.asarray(sim.data.get_site_jacp(site_name), dtype=np.float64).reshape(3, -1)[:, qvel_index]
        jac_rot = np.asarray(sim.data.get_site_jacr(site_name), dtype=np.float64).reshape(3, -1)[:, qvel_index]
        jac = np.vstack([jac_pos, jac_rot])
        lhs = jac @ jac.T + (damping * damping) * np.eye(6, dtype=np.float64)
        try:
            dq = jac.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            dq = jac.T @ np.linalg.pinv(lhs) @ error
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            dq = dq * (max_dq / dq_norm)

        q_next = np.asarray(sim.data.qpos[qpos_index], dtype=np.float64) + step_size * dq
        q_next = np.clip(q_next, q_low, q_high)
        sim.data.qpos[qpos_index] = q_next
        sim.data.qvel[qvel_index] = 0.0

    sim.forward()
    controller.update(force=True)
    if hasattr(controller, "reset_goal"):
        controller.reset_goal()
    post_world = _site_pose(sim, site_name)
    _, last_pos_error, last_rot_error = _ik_pose_error(post_world, target_world)
    success = success or (last_pos_error <= pos_tolerance and last_rot_error <= rot_tolerance)
    return {
        "success": bool(success),
        "iterations": int(iterations),
        "pos_error": float(last_pos_error),
        "rot_error": float(last_rot_error),
        "qpos": np.asarray(sim.data.qpos[qpos_index], dtype=np.float32).copy(),
    }


def set_gripper_width(env: Any, width: float, *, max_physical_width: float) -> None:
    robot = env.robots[0]
    sim = env.sim
    qpos_indexes = list(getattr(robot, "_ref_gripper_joint_pos_indexes", []) or [])
    qvel_indexes = list(getattr(robot, "_ref_gripper_joint_vel_indexes", []) or [])
    if not qpos_indexes:
        return
    width = float(np.clip(width, 0.0, max(float(max_physical_width), 1e-6)))
    if len(qpos_indexes) >= 2:
        values = np.asarray(sim.data.qpos[qpos_indexes], dtype=np.float64)
        signs = np.sign(values)
        if np.count_nonzero(signs) < len(signs):
            signs = np.asarray([1.0, -1.0] + [1.0] * max(0, len(qpos_indexes) - 2), dtype=np.float64)
        target = signs * (width / float(len(qpos_indexes)))
    else:
        target = np.asarray([width], dtype=np.float64)

    qpos_to_joint_id = {}
    try:
        qpos_to_joint_id = {int(qpos_adr): int(joint_id) for joint_id, qpos_adr in enumerate(sim.model.jnt_qposadr)}
    except Exception:
        qpos_to_joint_id = {}
    for offset, qpos_idx in enumerate(qpos_indexes):
        joint_id = qpos_to_joint_id.get(int(qpos_idx))
        value = float(target[offset])
        if joint_id is not None:
            low, high = sim.model.jnt_range[joint_id]
            if np.isfinite(low) and np.isfinite(high) and high > low:
                value = float(np.clip(value, low, high))
        sim.data.qpos[qpos_idx] = value
    if qvel_indexes:
        sim.data.qvel[qvel_indexes] = 0.0


def object_pose_from_free_joint(env: Any, joint_name: str) -> np.ndarray | None:
    qpos = np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).reshape(-1)
    if qpos.size < 7:
        return None
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = qpos[:3]
    quat_wxyz = qpos[3:7]
    pose[:3, :3] = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    return pose


def set_object_pose_free_joint(env: Any, joint_name: str, pose: np.ndarray) -> None:
    quat_xyzw = R.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    env.sim.data.set_joint_qpos(joint_name, np.concatenate([pose[:3, 3], quat_wxyz]))


def maybe_update_direct_attachment(
    *,
    env: Any,
    attached: dict[str, Any] | None,
    should_close: bool,
    current_eef_world: np.ndarray,
    target_eef_world: np.ndarray,
    enable: bool,
) -> dict[str, Any] | None:
    if not enable:
        return None
    if attached is not None and should_close:
        object_world = np.asarray(target_eef_world, dtype=np.float64) @ attached["eef_to_object"]
        set_object_pose_free_joint(env, attached["joint_name"], object_world)
        env.sim.forward()
        return attached
    if attached is not None and not should_close:
        return None
    if not should_close:
        return None

    env.sim.forward()
    grasp_status = gripper_grasp_status(env)
    grasped_names = list(grasp_status.get("grasped_objects", []) or [])
    if not grasped_names:
        return None
    base_env = unwrap_libero_env(env)
    objects_dict = getattr(base_env, "objects_dict", {}) or {}
    obj = objects_dict.get(grasped_names[0])
    joints = list(getattr(obj, "joints", []) or []) if obj is not None else []
    if not joints:
        return None
    object_world = object_pose_from_free_joint(env, joints[0])
    if object_world is None:
        return None
    return {
        "object": str(grasped_names[0]),
        "joint_name": joints[0],
        "eef_to_object": fast_inverse_homogeneous(np.asarray(target_eef_world, dtype=np.float32)).astype(np.float64)
        @ object_world,
    }


def run_episode_direct_ik(
    *,
    infer: SmolVLA_ModelInference,
    env: Any,
    task_language: str,
    init_state: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    control = cfg["control"]
    direct_cfg = cfg["direct_ik"]
    max_steps = int(control.get("max_steps") or getattr(env, "horizon", 500))
    action_index = max(0, int(control.get("action_index", 0)))
    replan_every_step = bool(control.get("replan_every_step", False))
    gripper_max_width = float(cfg.get("gripper_qpos_max_width", 0.08))
    gripper_close_threshold = gripper_close_threshold_for_task(task_language, control)
    pc_camera_names = pointcloud_camera_names_from_config(cfg)
    save_video = bool(cfg.get("save_video", True))

    env.reset()
    raw_obs = env.set_init_state(init_state)
    infer.policy.reset()
    infer.policy_reset()

    action_queue: list[np.ndarray] = []
    action_target_indices: list[int] = []
    planned_world_poses: np.ndarray | None = None
    planned_step_index = 0
    attached_object: dict[str, Any] | None = None
    video_frames: dict[str, list[np.ndarray]] = {}
    if save_video:
        append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
    render_onscreen_env(env, cfg, 0, force=True)

    model_call_count = 0
    timing_totals: dict[str, float] = defaultdict(float)
    timing_counts: dict[str, int] = defaultdict(int)
    episode_start_s = time.perf_counter()

    model_pose_actions: list[np.ndarray] = []
    pose_actions: list[np.ndarray] = []
    controller_pose_actions: list[np.ndarray] = []
    controller_targets: list[np.ndarray] = []
    ik_success_flags: list[bool] = []
    ik_iterations: list[int] = []
    ik_pos_errors: list[float] = []
    ik_rot_errors: list[float] = []
    gripper_targets: list[float] = []
    gripper_actuals: list[float] = []
    gripper_width_errors: list[float] = []
    gripper_should_close_flags: list[bool] = []
    gripper_grasp_flags: list[bool] = []
    gripper_contact_flags: list[bool] = []
    rewards: list[float] = []
    success_ever = False
    max_reward = 0.0

    for step in range(max_steps):
        need_model_point_cloud = (replan_every_step or not action_queue)
        point_cloud_world: np.ndarray | None = None
        if need_model_point_cloud:
            timing_start = time.perf_counter()
            point_cloud, point_cloud_world, eef_pose = observation_to_point_clouds(
                env,
                raw_obs,
                pc_camera_names,
                int(cfg["observation_height"]),
                int(cfg["observation_width"]),
                int(cfg["num_points"]),
                seed=step,
            )
            add_timing(timing_totals, timing_counts, "pointcloud_from_obs", timing_start)
            if cfg.get("add_gripper_cloud", True):
                timing_start = time.perf_counter()
                point_cloud = add_world_gripper_cloud_to_point_cloud(
                    point_cloud_world,
                    eef_pose,
                    gripper_width_percent_from_scalar(
                        float(eef_pose[-1]),
                        max_physical_width=gripper_max_width,
                    ),
                    total_points=int(cfg["num_points"]),
                    gripper_points=int(cfg.get("gripper_points", 500)),
                    gripper_len=float(cfg.get("gripper_len", 0.06)),
                    gripper_template=str(cfg.get("gripper_template", "reap")),
                    seed=step,
                    drop_strategy=str(cfg.get("gripper_drop_strategy", "tail")),
                    shuffle_points=bool(cfg.get("gripper_shuffle_points", False)),
                )
                add_timing(timing_totals, timing_counts, "pointcloud_add_gripper", timing_start)

            timing_start = time.perf_counter()
            chunk = infer.predict_action_chunk_obs(
                {"point_cloud": point_cloud, "state": identity_pose9_gripper(float(eef_pose[-1]))},
                task=task_language,
                postprocess=True,
                state_pose_mode="identity",
            )[0].detach().cpu().numpy()
            add_timing(timing_totals, timing_counts, "model_predict_action_chunk", timing_start)
            model_call_count += 1

            selected_index = min(action_index, len(chunk) - 1)
            if replan_every_step:
                action_queue.clear()
                action_target_indices.clear()
                planned_chunk = chunk[selected_index : selected_index + 1].copy()
                action_queue.append(planned_chunk[0])
                action_target_indices.append(0)
            else:
                exec_action_steps = int(control.get("exec_action_steps", 16))
                if exec_action_steps <= 0:
                    exec_action_steps = len(chunk)
                end_index = min(len(chunk), selected_index + exec_action_steps)
                planned_chunk = chunk[selected_index:end_index].copy()
                action_queue.extend(list(planned_chunk))
                action_target_indices.extend(list(range(len(planned_chunk))))
            planned_world_poses = planned_world_poses_from_umi_chunk(eef_pose, planned_chunk)
            planned_step_index = 0
        else:
            eef_pose = eef_pose9_gripper_from_obs(raw_obs)

        model_pose_action = np.asarray(action_queue[0], dtype=np.float32)
        target_idx = action_target_indices[0] if action_target_indices else planned_step_index
        pose_action = model_pose_action.copy()
        controller_pose_action = model_pose_action.copy()
        current_model_world = pose9_to_homo_np(np.asarray(eef_pose, dtype=np.float32)[..., :9])

        timing_start = time.perf_counter()
        current_controller_world = current_controller_eef_world(env)
        add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)

        if planned_world_poses is not None and len(planned_world_poses) > 0:
            target_idx = min(int(target_idx), len(planned_world_poses) - 1)
            target_model_world = planned_world_poses[target_idx]
        else:
            target_model_world = current_model_world @ pose9_to_homo_np(model_pose_action[:9])

        gripper_target = float(model_pose_action[-1])
        gripper_should_close = gripper_target < gripper_close_threshold
        if gripper_should_close:
            if point_cloud_world is None and bool(control.get("gripper_rim_correction_enable", True)):
                timing_start = time.perf_counter()
                point_cloud_world, _ = observation_to_world_point_cloud(
                    env,
                    raw_obs,
                    pc_camera_names,
                    int(cfg["observation_height"]),
                    int(cfg["observation_width"]),
                    int(cfg["num_points"]),
                    seed=step,
                )
                add_timing(timing_totals, timing_counts, "pointcloud_world_for_correction", timing_start)
            if point_cloud_world is not None:
                target_model_world, _ = correct_rim_grasp_target_world(
                    target_model_world,
                    point_cloud_world,
                    enabled=bool(control.get("gripper_rim_correction_enable", True)),
                    min_points=max(1, int(control.get("gripper_rim_correction_min_points", 20))),
                    axis_limit=max(0.0, float(control.get("gripper_rim_correction_axis_limit", 0.075))),
                    depth_limit=max(0.0, float(control.get("gripper_rim_correction_depth_limit", 0.06))),
                    z_min=float(control.get("gripper_rim_correction_z_min", -0.11)),
                    z_max=float(control.get("gripper_rim_correction_z_max", 0.04)),
                    top_quantile=float(control.get("gripper_rim_correction_top_quantile", 0.7)),
                    center_alpha=float(control.get("gripper_rim_correction_center_alpha", 0.8)),
                    max_axis_shift=max(0.0, float(control.get("gripper_rim_correction_max_axis_shift", 0.018))),
                    max_depth_shift=max(0.0, float(control.get("gripper_rim_correction_max_depth_shift", 0.018))),
                    lift=float(control.get("gripper_rim_correction_lift", 0.012)),
                )

        controller_target_world = model_world_pose_to_controller_world(
            target_model_world,
            current_model_world,
            current_controller_world,
        )
        pose_action[:9] = matrix_to_pose9(target_model_world)
        controller_pose_action[:9] = matrix_to_pose9(controller_target_world)

        if str(direct_cfg["gripper_mode"]) == "binary":
            gripper_target_width = 0.0 if gripper_should_close else gripper_max_width
        else:
            gripper_target_width = canonical_gripper_width(gripper_target, max_physical_width=gripper_max_width)

        timing_start = time.perf_counter()
        ik_result = solve_site_ik(
            env=env,
            target_world=controller_target_world,
            max_iters=int(direct_cfg["max_iters"]),
            pos_tolerance=float(direct_cfg["pos_tolerance"]),
            rot_tolerance=float(direct_cfg["rot_tolerance"]),
            damping=float(direct_cfg["damping"]),
            step_size=float(direct_cfg["step_size"]),
            max_dq=float(direct_cfg["max_dq"]),
        )
        set_gripper_width(env, gripper_target_width, max_physical_width=gripper_max_width)
        attached_object = maybe_update_direct_attachment(
            env=env,
            attached=attached_object,
            should_close=bool(gripper_should_close),
            current_eef_world=current_controller_world,
            target_eef_world=controller_target_world,
            enable=bool(direct_cfg.get("attach_objects", False)),
        )
        raw_obs = refresh_observation_after_state_write(env)
        add_timing(timing_totals, timing_counts, "direct_state_write", timing_start)

        timing_start = time.perf_counter()
        post_controller_world = current_controller_eef_world(env)
        add_timing(timing_totals, timing_counts, "controller_pose_read", timing_start)
        pose_pos_error, pose_rot_error = pose_error_to_target(post_controller_world, controller_target_world)

        gripper_actual = float(gripper_scalar(raw_obs))
        gripper_actual_width = canonical_gripper_width(gripper_actual, max_physical_width=gripper_max_width)
        gripper_width_error = abs(gripper_actual_width - gripper_target_width)

        timing_start = time.perf_counter()
        try:
            success_now = bool(env.check_success())
        except Exception:
            success_now = False
        add_timing(timing_totals, timing_counts, "success_check", timing_start)
        success_ever = success_ever or success_now
        reward = float(success_now)
        max_reward = max(max_reward, reward)
        rewards.append(reward)

        grasp_status = {"any_contact": False, "any_grasped": False}
        if gripper_should_close:
            timing_start = time.perf_counter()
            grasp_status = gripper_grasp_status(env)
            add_timing(timing_totals, timing_counts, "gripper_grasp_check", timing_start)

        timing_start = time.perf_counter()
        render_onscreen_env(env, cfg, step + 1)
        add_timing(timing_totals, timing_counts, "render", timing_start)
        if save_video:
            timing_start = time.perf_counter()
            append_video_frames(video_frames, raw_obs, list(cfg["camera_names"]))
            add_timing(timing_totals, timing_counts, "video_frame_append", timing_start)

        model_pose_actions.append(model_pose_action.copy())
        pose_actions.append(pose_action.copy())
        controller_pose_actions.append(controller_pose_action.copy())
        controller_targets.append(matrix_to_pose9(controller_target_world))
        ik_success_flags.append(bool(ik_result["success"]))
        ik_iterations.append(int(ik_result["iterations"]))
        ik_pos_errors.append(float(pose_pos_error))
        ik_rot_errors.append(float(pose_rot_error))
        gripper_targets.append(float(gripper_target))
        gripper_actuals.append(float(gripper_actual))
        gripper_width_errors.append(float(gripper_width_error))
        gripper_should_close_flags.append(bool(gripper_should_close))
        gripper_grasp_flags.append(bool(grasp_status.get("any_grasped", False)))
        gripper_contact_flags.append(bool(grasp_status.get("any_contact", False)))

        action_queue.pop(0)
        if action_target_indices:
            action_target_indices.pop(0)
        planned_step_index += 1
        if success_now:
            break

    final_goal_status = goal_predicate_status(env)
    wall_s = time.perf_counter() - episode_start_s
    return {
        "success": bool(success_ever),
        "steps": int(len(ik_success_flags)),
        "model_call_count": int(model_call_count),
        "sum_reward": float(np.sum(rewards)),
        "max_reward": float(max_reward),
        "timings": summarize_timings(timing_totals, timing_counts, wall_s=wall_s),
        "goal_predicates_final": final_goal_status,
        "direct_ik": dict(direct_cfg),
        "ik_success_rate": float(np.mean(ik_success_flags)) if ik_success_flags else 0.0,
        "ik_pos_error_mean": float(np.mean(ik_pos_errors)) if ik_pos_errors else 0.0,
        "ik_rot_error_mean": float(np.mean(ik_rot_errors)) if ik_rot_errors else 0.0,
        "ik_iterations_mean": float(np.mean(ik_iterations)) if ik_iterations else 0.0,
        "gripper_close_threshold": float(gripper_close_threshold),
        "gripper_close_target_steps": int(np.sum(gripper_should_close_flags)),
        "gripper_grasp_detected_any": bool(any(gripper_grasp_flags)),
        "gripper_grasp_detected_steps": int(np.sum(gripper_grasp_flags)),
        "gripper_contact_detected_any": bool(any(gripper_contact_flags)),
        "gripper_contact_detected_steps": int(np.sum(gripper_contact_flags)),
        "model_pose_actions": np.asarray(model_pose_actions, dtype=np.float32),
        "pose_actions": np.asarray(pose_actions, dtype=np.float32),
        "controller_pose_actions": np.asarray(controller_pose_actions, dtype=np.float32),
        "controller_targets": np.asarray(controller_targets, dtype=np.float32),
        "ik_success_flags": np.asarray(ik_success_flags, dtype=bool),
        "ik_iterations": np.asarray(ik_iterations, dtype=np.int64),
        "ik_pos_errors": np.asarray(ik_pos_errors, dtype=np.float32),
        "ik_rot_errors": np.asarray(ik_rot_errors, dtype=np.float32),
        "gripper_targets": np.asarray(gripper_targets, dtype=np.float32),
        "gripper_actuals": np.asarray(gripper_actuals, dtype=np.float32),
        "gripper_width_errors": np.asarray(gripper_width_errors, dtype=np.float32),
        "gripper_should_close_flags": np.asarray(gripper_should_close_flags, dtype=bool),
        "gripper_grasp_flags": np.asarray(gripper_grasp_flags, dtype=bool),
        "gripper_contact_flags": np.asarray(gripper_contact_flags, dtype=bool),
        "video_frames": video_frames,
    }


def configure_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    cfg["policy_path"] = cfg_get(cfg, args.policy_path, "policy_path")
    cfg["policy_repo_id"] = args.policy_repo_id if args.policy_repo_id is not None else cfg.get("policy_repo_id")
    cfg["suites"] = resolve_suite_names(args.suite, cfg)
    cfg["all_tasks"] = bool(args.all_tasks if args.all_tasks is not None else cfg.get("all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes"))
    cfg["device"] = cfg_get(cfg, args.device, "device")
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points"))
    cfg["save_video"] = bool(args.save_video)
    if args.render_mode is not None:
        cfg["render_mode"] = str(args.render_mode)
    if args.headed is not None:
        cfg["render_mode"] = "viewer3d" if bool(args.headed) else "offscreen"
    if args.render_camera is not None:
        cfg["render_camera"] = str(args.render_camera)
    if args.render_every_n_steps is not None:
        cfg["render_every_n_steps"] = int(args.render_every_n_steps)
    if args.render_gpu_device_id is not None:
        cfg["render_gpu_device_id"] = int(args.render_gpu_device_id)

    cfg.setdefault("control", {})
    control = cfg["control"]
    control.setdefault("control_mode", "absolute_pose")
    control.setdefault("replan_every_step", False)
    control.setdefault("action_index", 0)
    control.setdefault("exec_action_steps", 16)
    control.setdefault("max_steps", 1000)
    control.setdefault("gripper_close_threshold", 0.07)
    control.setdefault("gripper_close_threshold_rules", [{"keywords": ["bowl", "plate"], "threshold": 0.03}])
    control.setdefault("gripper_rim_correction_enable", True)
    control.setdefault("gripper_rim_correction_min_points", 20)
    control.setdefault("gripper_rim_correction_axis_limit", 0.075)
    control.setdefault("gripper_rim_correction_depth_limit", 0.06)
    control.setdefault("gripper_rim_correction_z_min", -0.11)
    control.setdefault("gripper_rim_correction_z_max", 0.04)
    control.setdefault("gripper_rim_correction_top_quantile", 0.7)
    control.setdefault("gripper_rim_correction_center_alpha", 0.8)
    control.setdefault("gripper_rim_correction_max_axis_shift", 0.018)
    control.setdefault("gripper_rim_correction_max_depth_shift", 0.018)
    control.setdefault("gripper_rim_correction_lift", 0.012)
    if args.control_freq is not None:
        control["control_freq"] = float(args.control_freq)
    if args.action_index is not None:
        control["action_index"] = int(args.action_index)
    if args.exec_action_steps is not None:
        control["exec_action_steps"] = int(args.exec_action_steps)
    if args.replan_every_step is not None:
        control["replan_every_step"] = bool(args.replan_every_step)
    if args.warmup_steps is not None:
        control["warmup_steps"] = int(args.warmup_steps)
    if args.gripper_close_threshold is not None:
        control["gripper_close_threshold"] = float(args.gripper_close_threshold)
    if args.keyboard_vis is not None:
        cfg["keyboard_vis"] = bool(args.keyboard_vis)
    if args.keyboard_vis_mode is not None:
        cfg["keyboard_vis_mode"] = str(args.keyboard_vis_mode)

    cfg["direct_ik"] = {
        "max_iters": int(args.ik_max_iters),
        "pos_tolerance": float(args.ik_pos_tolerance),
        "rot_tolerance": float(args.ik_rot_tolerance),
        "damping": float(args.ik_damping),
        "step_size": float(args.ik_step_size),
        "max_dq": float(args.ik_max_dq),
        "gripper_mode": str(args.direct_gripper_mode),
        "attach_objects": bool(args.direct_attach_objects),
    }
    return cfg


def write_direct_reports(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    suite_names: list[str],
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = build_eval_summary(cfg=cfg, suite_names=suite_names, all_results=all_results)
    summary["direct_ik"] = dict(cfg["direct_ik"])
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "overall_report.json", {"overall": summary["overall"], "suites": summary["suite_reports"], "direct_ik": summary["direct_ik"]})
    for suite_report in summary["suite_reports"]:
        write_json_atomic(output_dir / str(suite_report["suite"]) / "suite_report.json", suite_report)
    return summary


def main() -> None:
    args = parse_args()
    cfg = configure_from_args(args, load_config(args.config))
    configure_mujoco_render_backend(cfg)
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))
    output_dir = Path(args.output_dir or cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    from libero.libero import benchmark

    infer = SmolVLA_ModelInference(
        policy_path=cfg["policy_path"],
        policy_repo_id=cfg.get("policy_repo_id"),
        device=cfg["device"],
    )

    all_results = []
    suite_names = list(cfg["suites"])
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
                render_mode=str(cfg.get("render_mode", "offscreen")),
                render_camera=str(cfg.get("render_camera", "agentview")),
                render_gpu_device_id=int(cfg.get("render_gpu_device_id", -1)),
                control_delta=False,
                control_freq=float(cfg.get("control_freq", cfg.get("control", {}).get("control_freq", 20))),
            )
            init_states = get_task_init_states(suite, int(task_id))
            task_results = []
            try:
                for episode_idx in range(cfg["episodes"]):
                    episode_dir = output_dir / suite_name / f"task_{int(task_id):03d}" / f"episode_{episode_idx:03d}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[direct-ik] start {suite_name} task={task_id} episode={episode_idx}")
                    try:
                        result = run_episode_direct_ik(
                            infer=infer,
                            env=env,
                            task_language=task.language,
                            init_state=init_states[episode_idx % len(init_states)],
                            cfg=cfg,
                        )
                        for array_name in (
                            "model_pose_actions",
                            "pose_actions",
                            "controller_pose_actions",
                            "controller_targets",
                            "ik_success_flags",
                            "ik_iterations",
                            "ik_pos_errors",
                            "ik_rot_errors",
                            "gripper_targets",
                            "gripper_actuals",
                            "gripper_width_errors",
                            "gripper_should_close_flags",
                            "gripper_grasp_flags",
                            "gripper_contact_flags",
                        ):
                            np.save(episode_dir / f"{array_name}.npy", result[array_name])
                        if cfg.get("save_trajectory", True):
                            write_trajectory_ply(episode_dir / "trajectory_direct_ik.ply", result["controller_pose_actions"])
                        record = {
                            "episode_index": episode_idx,
                            "demo_name": "direct_ik_rollout",
                            "video_dir_name": episode_dir.name,
                        }
                        video_paths = export_episode_videos(result, episode_dir.parent, record, cfg)
                        episode_record = episode_result_record(result, episode_idx)
                        if video_paths:
                            episode_record["videos"] = video_paths
                        write_json_atomic(episode_dir / "result.json", episode_record)
                        task_results.append(episode_record)
                        current_results = [
                            *all_results,
                            make_task_summary(
                                suite_name=suite_name,
                                task_id=int(task_id),
                                task_name=task.name,
                                task_language=task.language,
                                task_results=task_results,
                            ),
                        ]
                        write_direct_reports(
                            output_dir=output_dir,
                            cfg=cfg,
                            suite_names=suite_names,
                            all_results=current_results,
                        )
                        print(
                            f"{suite_name} task={task_id} episode={episode_idx} "
                            f"success={result['success']} steps={result['steps']} "
                            f"model_calls={result['model_call_count']} "
                            f"ik_success_rate={result['ik_success_rate']:.3f} "
                            f"wall_s={result['timings']['wall_s']:.3f}"
                        )
                    except Exception as exc:
                        failure = {
                            "episode_index": int(episode_idx),
                            "success": False,
                            "steps": 0,
                            "sum_reward": 0.0,
                            "error": repr(exc),
                        }
                        task_results.append(failure)
                        write_json_atomic(episode_dir / "result.json", failure)
                        current_results = [
                            *all_results,
                            make_task_summary(
                                suite_name=suite_name,
                                task_id=int(task_id),
                                task_name=task.name,
                                task_language=task.language,
                                task_results=task_results,
                            ),
                        ]
                        write_direct_reports(
                            output_dir=output_dir,
                            cfg=cfg,
                            suite_names=suite_names,
                            all_results=current_results,
                        )
                        print(f"[WARN] direct-IK {suite_name} task={task_id} episode={episode_idx} failed: {exc!r}")
            finally:
                try:
                    env.close()
                except Exception as exc:
                    print(f"[WARN] failed to close env for {suite_name} task={task_id}: {exc!r}")
            all_results.append(
                make_task_summary(
                    suite_name=suite_name,
                    task_id=int(task_id),
                    task_name=task.name,
                    task_language=task.language,
                    task_results=task_results,
                )
            )

    summary = write_direct_reports(output_dir=output_dir, cfg=cfg, suite_names=suite_names, all_results=all_results)
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
