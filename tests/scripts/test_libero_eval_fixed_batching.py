#!/usr/bin/env python

import json
import os
import queue
from types import SimpleNamespace

import numpy as np
import pytest

# Import the evaluator without loading a real policy or initializing CUDA.
os.environ.setdefault("SONG_LIBERO_ENV_WORKER", "1")

from benchmarks.song_real_libero.scripts.libero_setting import (  # noqa: E402
    libero_pointcloud_eval as eval_module,
)
from benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_eval import (  # noqa: E402
    FixedBatchInferenceCache,
    ProcessInferenceProxy,
    _ProcessInferenceRequest,
    _execute_process_inference_fixed_slots,
    _process_worker_environment,
    evaluate_task,
    initialize_realtime_suite_progress,
    load_resumable_episode_record,
)


class _FakeInference:
    def __init__(self) -> None:
        self.calls = []

    def predict_action_chunk_obs(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        values = np.asarray(observation["value"], dtype=np.float32).reshape(-1)
        return values[:, None, None]


def _resume_config() -> dict:
    return {
        "control": {"max_steps": 1000},
        "policy_noise_seed": 0,
        "strict_official_init": True,
        "dataset_domain_env": False,
        "dataset_domain_oracle_actions": False,
        "worldflow_action_fusion_override": None,
        "secondary_view_causal_ablation": False,
        "world_to_ego_causal_ablation": False,
        "episodes": 10,
        "episode_ids": None,
        "recreate_env_per_episode": True,
    }


def _write_resumable_result(output_dir, *, task_id: int, episode_index: int) -> None:
    episode_dir = (
        output_dir
        / "libero_10"
        / f"task_{task_id:03d}"
        / f"episode_{episode_index:03d}"
    )
    episode_dir.mkdir(parents=True)
    action_path = episode_dir / "actions.npz"
    action_path.write_bytes(b"actions")
    record = {
        "episode_index": episode_index,
        "success": True,
        "error": None,
        "steps": 42,
        "max_steps": 1000,
        "model_call_count": 2,
        "policy_forward_call_count": 2,
        "action_source": "policy_flow_matching_sample",
        "action_npz": str(action_path),
        "evaluation_protocol": {
            "name": "single_uninterrupted_rollout",
            "rollouts_per_initial_state": 1,
            "retry_failed_rollout": False,
            "action_samples_per_model_call": 1,
            "action_sample_selection": "none",
            "initial_state_source": "task_suite.get_task_init_states",
            "fixture_reset_sequence": "seeded_serial_episode_index",
            "benchmark_comparable": True,
        },
        "policy_noise_seed_base": 0,
        "strict_official_init": True,
        "environment_alignment": {
            "enabled": False,
            "benchmark_comparable": True,
            "initial_state_source": "task_suite.get_task_init_states",
        },
    }
    (episode_dir / "result.json").write_text(json.dumps(record), encoding="utf-8")


def test_realtime_progress_reuses_only_strictly_validated_episode(tmp_path):
    _write_resumable_result(tmp_path, task_id=6, episode_index=4)

    initialize_realtime_suite_progress(
        output_dir=tmp_path,
        suite_name="libero_10",
        task_ids=[6, 8],
        episodes_per_task=10,
        cfg=_resume_config(),
    )

    progress = json.loads((tmp_path / "libero_10" / "progress.json").read_text())
    assert progress["completed_episode_count"] == 1
    assert progress["success_count"] == 1
    assert progress["tasks"]["6"]["episodes"]["4"]["steps"] == 42


def test_resume_rejects_result_from_different_policy_seed(tmp_path):
    _write_resumable_result(tmp_path, task_id=6, episode_index=4)
    config = _resume_config()
    config["policy_noise_seed"] = 1

    with pytest.raises(RuntimeError, match="policy_noise_seed_base"):
        load_resumable_episode_record(
            output_dir=tmp_path,
            suite_name="libero_10",
            task_id=6,
            episode_index=4,
            cfg=config,
        )


def test_evaluate_task_skips_valid_completed_episode_without_creating_env(tmp_path):
    _write_resumable_result(tmp_path, task_id=6, episode_index=0)

    class _Suite:
        @staticmethod
        def get_task(task_id):
            assert task_id == 6
            return SimpleNamespace(name="task-6", language="do task 6")

    summary = evaluate_task(
        infer=None,
        suite=_Suite(),
        suite_name="libero_10",
        task_id=6,
        cfg=_resume_config(),
        output_dir=tmp_path,
        episode_indices=[0],
        task_init_states=np.zeros((1, 1), dtype=np.float32),
    )

    assert summary["episode_count"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["episodes"][0]["episode_index"] == 0


def test_stale_evaluation_claim_is_atomically_replaced(tmp_path, monkeypatch):
    claim_dir = tmp_path / ".evaluation_run.claim"
    claim_dir.mkdir()
    (claim_dir / "owner.json").write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "hostname": os.uname().nodename,
                "started_unix_s": 0,
                "argv": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_module, "_EVALUATION_RUN_LOCK_CLAIM_DIR", None)

    eval_module.acquire_evaluation_run_lock(tmp_path)

    owner = json.loads((claim_dir / "owner.json").read_text())
    assert owner["pid"] == os.getpid()
    assert len(list(tmp_path.glob(".evaluation_run.claim.stale.*"))) == 1


def test_process_proxy_preserves_worldflow_frame_contract():
    proxy = ProcessInferenceProxy(
        worker_id=0,
        request_queue=queue.Queue(),
        response_queue=queue.Queue(),
        vla_adapter_enable=True,
        image_feature_keys=["observation.images.agentview"],
        worldflow_enable=True,
        worldflow_reference_frame="robot_base",
    )

    assert proxy.policy.config.worldflow_enable is True
    assert proxy.policy.config.worldflow_reference_frame == "robot_base"


def _request(worker_id: int, value: float, seed: int) -> _ProcessInferenceRequest:
    return _ProcessInferenceRequest(
        worker_id=worker_id,
        request_id=100 + worker_id,
        observation={"value": np.asarray([value], dtype=np.float32)},
        task=f"task-{worker_id}",
        postprocess=True,
        state_pose_mode="identity",
        noise_seed=seed,
    )


def test_fixed_slot_batch_keeps_worker_rows_and_only_replies_to_real_requests():
    infer = _FakeInference()
    responses = {worker_id: queue.Queue() for worker_id in range(3)}
    prior_padding = {1: _request(1, 11.0, 111)}
    real = {0: _request(0, 10.0, 110), 2: _request(2, 12.0, 112)}

    _execute_process_inference_fixed_slots(
        infer,
        real,
        responses,
        slot_count=3,
        padding_requests_by_worker=prior_padding,
    )

    observation, kwargs = infer.calls[0]
    np.testing.assert_array_equal(observation["value"].reshape(-1), [10.0, 11.0, 12.0])
    assert kwargs["task"] == ["task-0", "task-1", "task-2"]
    assert kwargs["noise_seed"] == [110, 111, 112]
    assert responses[1].empty()
    assert responses[0].get_nowait()[:2] == ("ok", 100)
    assert responses[2].get_nowait()[:2] == ("ok", 102)


def test_fixed_slot_batch_retains_latest_request_for_later_padding():
    infer = _FakeInference()
    responses = {worker_id: queue.Queue() for worker_id in range(2)}
    padding = {}

    _execute_process_inference_fixed_slots(
        infer,
        {0: _request(0, 1.0, 1), 1: _request(1, 2.0, 2)},
        responses,
        slot_count=2,
        padding_requests_by_worker=padding,
    )
    responses[0].get_nowait()
    responses[1].get_nowait()
    _execute_process_inference_fixed_slots(
        infer,
        {0: _request(0, 3.0, 3)},
        responses,
        slot_count=2,
        padding_requests_by_worker=padding,
    )

    observation, kwargs = infer.calls[1]
    np.testing.assert_array_equal(observation["value"].reshape(-1), [3.0, 2.0])
    assert kwargs["noise_seed"] == [3, 2]
    assert responses[1].empty()


def test_fixed_slot_exact_cache_reuses_original_action_without_second_forward(tmp_path):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.safetensors").write_bytes(b"model")
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (policy / name).write_text("{}\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache = FixedBatchInferenceCache(cache_dir, policy_path=policy, mode="read_write")
    requests = {0: _request(0, 1.0, 1), 1: _request(1, 2.0, 2)}

    first_infer = _FakeInference()
    first_responses = {worker_id: queue.Queue() for worker_id in range(2)}
    _execute_process_inference_fixed_slots(
        first_infer,
        requests,
        first_responses,
        slot_count=2,
        padding_requests_by_worker={},
        inference_cache=cache,
    )
    first_actions = [first_responses[index].get_nowait()[2] for index in range(2)]
    assert len(first_infer.calls) == 1
    assert cache.write_count == 1

    readonly_cache = FixedBatchInferenceCache(cache_dir, policy_path=policy, mode="readonly")
    second_infer = _FakeInference()
    second_responses = {worker_id: queue.Queue() for worker_id in range(2)}
    _execute_process_inference_fixed_slots(
        second_infer,
        requests,
        second_responses,
        slot_count=2,
        padding_requests_by_worker={},
        inference_cache=readonly_cache,
    )
    second_actions = [second_responses[index].get_nowait()[2] for index in range(2)]
    assert second_infer.calls == []
    assert readonly_cache.hit_count == 1
    for first, second in zip(first_actions, second_actions, strict=True):
        np.testing.assert_array_equal(first, second)


def test_process_worker_environment_can_override_parent_cuda_and_egl_namespaces(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "3")
    monkeypatch.setenv("SONG_LIBERO_ENV_CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SONG_LIBERO_ENV_MUJOCO_EGL_DEVICE_ID", "0")

    environment = _process_worker_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"
