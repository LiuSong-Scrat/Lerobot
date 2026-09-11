"""Output contract tests; no checkpoint, renderer or environment rollout required."""

import argparse
import importlib
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from benchmarks.song_real_libero.scripts.libero_setting import (
    libero_eval_artifacts as artifacts,
    libero_eval_visualization as vis,
    libero_pointcloud_utils as utils,
)


def chunk(n=4):
    rows = np.zeros((n, 10), dtype=np.float32)
    rows[:, 3] = rows[:, 7] = 1
    rows[:, 0] = np.linspace(0, 0.06, n)
    rows[:, 9] = 0.04
    return rows


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setenv("SONG_LIBERO_ENV_WORKER", "1")
    return importlib.import_module(
        "benchmarks.song_real_libero.scripts.libero_setting.libero_pointcloud_eval"
    )


@pytest.fixture
def camera(monkeypatch):
    module = ModuleType("robosuite.utils.camera_utils")
    intrinsic = np.array([[70.0, 0, 31.0], [0, 60.0, 22.0], [0, 0, 1.0]])
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.1, 0.2, 0.3]
    module.get_camera_intrinsic_matrix = lambda *args: intrinsic.copy()
    module.get_camera_extrinsic_matrix = lambda *args: extrinsic.copy()
    module.get_real_depth_map = lambda sim, depth: depth.copy()
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", module)
    raw = {
        "agentview_image": np.arange(48 * 80 * 3, dtype=np.uint8).reshape(48, 80, 3),
        "agentview_depth": np.full((48, 80), 0.8),
    }
    return SimpleNamespace(sim=None), raw, extrinsic


def test_output_argument_defaults_and_overrides():
    parser = argparse.ArgumentParser()
    artifacts.add_output_arguments(parser)
    cfg = {"video_fps": 17}
    artifacts.configure_output(
        cfg,
        parser.parse_args(["--save-action-visualizations", "--no-save-action-records"]),
    )
    assert cfg["video_fps"] == 17 and cfg["save_action_visualizations"]
    assert not cfg["save_action_records"] and cfg["save_action_chunks"]
    with pytest.raises(ValueError):
        artifacts.configure_output(
            {}, parser.parse_args(["--action-vis-every-n-frames", "0"])
        )


def test_projection_inverts_actual_backprojection_and_saved_flip(camera):
    env, raw, extrinsic = camera
    before = raw["agentview_image"].copy()
    pc = utils.backproject_camera(env, raw, "agentview", 48, 80)
    world = pc[:, :3] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    observation = vis.libero_front_observation(env, raw)
    pixels, valid = vis.project_world_points_to_front_image(world, observation)
    y, x = np.meshgrid(np.arange(48), np.arange(80), indexing="ij")
    np.testing.assert_allclose(pixels, np.c_[79 - x.ravel(), y.ravel()], atol=1e-5)
    # Interior pixels avoid expected floating-point round-off at image bounds.
    assert valid.reshape(48, 80)[1:-1, 1:-1].all()
    np.testing.assert_array_equal(observation.front_rgb, before[:, ::-1])
    np.testing.assert_array_equal(raw["agentview_image"], before)


def test_episode_artifacts_and_failure_retention(tmp_path, camera):
    env, raw, _ = camera
    cloud = np.array(
        [
            [0, 0, 0, 255, 0, 0],
            [0.01, 0, 0, 20, 50, 30],
            [0, -0.02, 0, 255, 0, 0],
            [0, 0.02, 0, 255, 0, 0],
        ],
        dtype=np.float32,
    )
    snapshot = {
        "point_cloud": cloud[None],
        "operation_prob": np.array([[0.1, 0.3, 0.7, 0.9]]),
    }
    cfg = {
        **artifacts.OUTPUT_DEFAULTS,
        "save_action_visualizations": True,
        "save_frame_pointclouds": True,
        "failure_artifacts_only": True,
        "camera_names": ["agentview"],
        "gripper_points": 2,
    }
    recorder = artifacts.EpisodeArtifacts(tmp_path, 3, cfg)
    anchor = np.eye(4)
    anchor[:3, 3] = [0.1, 0.2, 1.1]
    original = cloud.copy()
    recorder.frame(0, cloud)
    recorder.chunk(0, 1, chunk(), cloud, anchor, env, raw, snapshot, 0, 2, 0.04)
    paths = recorder.visualizations[0]
    assert paths["current_gripper_points"] == 2  # Red scene point is not a finger.
    assert Image.open(tmp_path / paths["image"]).size == (768, 461)
    text = (tmp_path / paths["ply"]).read_text()
    assert (
        "property float operation_prob" in text
        and "property uchar action_phase" in text
    )
    np.testing.assert_array_equal(np.load(tmp_path / recorder.chunks[0]), chunk())
    np.testing.assert_array_equal(cloud, original)
    assert (
        (tmp_path / "frame_pointclouds/episode_003/frame_000000_model_input_raw.ply")
        .read_bytes()
        .startswith(b"ply\nformat binary_little_endian")
    )
    result = recorder.finish(True)
    assert not result["action_visualizations"]
    assert not (tmp_path / paths["ply"]).exists()
    assert (tmp_path / recorder.chunks[0]).exists()
    assert result["frame_pointcloud_count"] == 1


def test_action_alignment_preserves_holds_and_unknown_model_rows(tmp_path):
    records = []
    for episode in [3, 1]:
        model = chunk(2)
        model[1] = model[0]
        model = np.concatenate([model, np.full((1, 10), np.nan)])
        controller = np.arange(21, dtype=np.float32).reshape(3, 7)
        result = {
            "artifact_model_rows": model,
            "artifact_controller_rows": controller,
            "artifact_execution_indices": [
                [episode, 1, 0, 0, 1, 0],
                [episode, 1, 0, 1, 2, 1],
                [episode, 1, -1, 2, 0, 2],
            ],
            "predicted_action_chunks": chunk()[None],
        }
        saved = artifacts.save_action_records(tmp_path, episode, result)
        records.append({"episode_index": episode, **saved})
    manifest = artifacts.finalize_alignment(tmp_path, records)
    merged = np.load(
        tmp_path / "executed_action_alignment/all_executed_model10_controller7.npy"
    )
    assert merged.shape == (6, 17) and np.isnan(merged[[2, 5], :10]).all()
    indices = np.load(tmp_path / "executed_action_alignment/all_execution_index.npy")
    np.testing.assert_array_equal(indices[:, 0], [1, 1, 1, 3, 3, 3])
    assert manifest["execution_phase_codes"]["1"] == "waypoint_hold"
    np.testing.assert_array_equal(merged[:3, 10:], controller)


class FakeInference:
    def __init__(self):
        self.policy = SimpleNamespace(
            model=SimpleNamespace(capture_pointseg_visualization=True)
        )
        self.calls = 0

    def predict_action_chunk_obs(self, observation, **kwargs):
        self.calls += 1
        pc = np.asarray(observation["point_cloud"])
        if pc.ndim == 2:
            pc = pc[None]
        self.policy.model.last_pointseg_visualization = {
            "point_cloud": pc.copy(),
            "operation_prob": pc[:, :, 0].copy(),
        }
        return np.broadcast_to(chunk()[None], (len(pc), 4, 10)).copy()


def requests(evaluator):
    return [
        evaluator._ProcessInferenceRequest(
            worker_id=i,
            request_id=i + 10,
            observation={"point_cloud": np.full((5, 6), i + 0.2, dtype=np.float32)},
            task="test",
            postprocess=True,
            state_pose_mode="identity",
            noise_seed=42 + i,
        )
        for i in range(2)
    ]


def test_dynamic_and_fixed_slots_snapshot_rows_and_cache_hits(evaluator):
    infer = FakeInference()
    reqs = requests(evaluator)
    queues = {i: queue.Queue() for i in range(2)}
    evaluator._execute_process_inference_batch(infer, reqs, queues)
    for i in range(2):
        status, rid, payload = queues[i].get_nowait()
        assert status == "ok", payload
        np.testing.assert_allclose(
            payload["artifact_snapshot"]["operation_prob"], i + 0.2
        )
    evaluator._execute_process_inference_fixed_slots(
        infer,
        {1: reqs[1]},
        queues,
        slot_count=2,
        padding_requests_by_worker={0: reqs[0]},
    )
    status, _, payload = queues[1].get_nowait()
    assert status == "ok", payload
    np.testing.assert_allclose(payload["artifact_snapshot"]["operation_prob"], 1.2)
    assert queues[0].empty() and infer.calls == 2
    cache = SimpleNamespace(
        key=lambda slots: "hit", load=lambda key: np.stack([chunk(), chunk()])
    )
    evaluator._execute_process_inference_fixed_slots(
        infer,
        {1: reqs[1]},
        queues,
        slot_count=2,
        padding_requests_by_worker={0: reqs[0]},
        inference_cache=cache,
    )
    status, _, payload = queues[1].get_nowait()
    assert status == "ok", payload
    assert isinstance(payload, np.ndarray) and infer.calls == 2


def test_thread_batcher_returns_caller_local_snapshot(evaluator):
    infer = FakeInference()
    scheduler = evaluator.BatchedInferenceScheduler(
        infer, max_batch_size=2, batch_wait_ms=100
    )

    def predict(i):
        result = scheduler.predict_action_chunk_obs(
            {"point_cloud": np.full((5, 6), i + 0.2)}, task="test"
        )
        return result, scheduler.artifact_snapshot

    try:
        with ThreadPoolExecutor(2) as pool:
            rows = list(pool.map(predict, range(2)))
        for i, (result, snapshot) in enumerate(rows):
            np.testing.assert_array_equal(result[0], chunk())
            np.testing.assert_allclose(snapshot["operation_prob"], i + 0.2)
    finally:
        scheduler.close()


def test_video_output_layout_and_input_immutability(evaluator, monkeypatch, tmp_path):
    frames = [np.full((64, 96, 3), 150, dtype=np.uint8) for _ in range(3)]
    copied = [f.copy() for f in frames]
    written = []
    monkeypatch.setattr(
        evaluator,
        "_write_video",
        lambda path, images, fps: written.append((path, list(images), fps)),
    )
    record = {"episode_index": 2, "task_name": "test task"}
    result = {
        "video_frames": {"agentview_image": frames, "robot0_eye_in_hand_image": frames},
        "success": False,
    }
    paths = evaluator.export_episode_videos(
        result, tmp_path, record, {"save_video": True}
    )
    assert paths == [
        "videos/episode_002.mp4",
        "videos/robot0_eye_in_hand_image/episode_002.mp4",
    ]
    assert len(written[0][1]) == 3 and written[0][1][0].shape == (64, 96, 3)
    assert written[0][2] == 20
    for a, b in zip(frames, copied, strict=True):
        np.testing.assert_array_equal(a, b)


def test_video_ignores_legacy_resize_settings():
    parser = argparse.ArgumentParser()
    artifacts.add_output_arguments(parser)
    cfg = {"video_width": 512, "video_height": 512}
    artifacts.configure_output(cfg, parser.parse_args(["--video-width", "768"]))
    assert "video_width" not in cfg and "video_height" not in cfg
    assert cfg["video_size_mode"] == "native_capture"
    frame = np.zeros((256, 320, 3), dtype=np.uint8)
    output = next(
        artifacts.annotated_frames(
            [frame], [], "task", 0, False, {"video_width": 512, "video_height": 512}
        )
    )
    assert output.shape == frame.shape


def test_umi_saves_every_call_exact_existing_visualization(evaluator, tmp_path):
    parser = argparse.ArgumentParser()
    artifacts.add_output_arguments(parser)
    cfg = {"failure_artifacts_only": True, "action_vis_every_n_frames": 9999}
    artifacts.configure_output(cfg, parser.parse_args(["--save_umi_vis"]))
    assert cfg["save_umi_vis"]
    assert parser.parse_args(["--no-save-umi-vis"]).save_umi_vis is False
    assert parser.parse_args(["--save-umi-vis"]).save_umi_vis is True
    pc = np.zeros((100, 6), dtype=np.float32)
    pc[:, 3:] = [100, 200, 50]
    before = pc.copy()
    recorder = artifacts.EpisodeArtifacts(tmp_path, 2, cfg)
    for call, frame in enumerate([0, 1, 2], 1):
        recorder.umi(frame, call, chunk(), pc, renderer=evaluator.vis_umi_data)
    result = recorder.finish(True)
    assert result["umi_vis_count"] == 3  # Also retain successful episodes.
    expected = tmp_path / "expected.ply"
    evaluator.vis_umi_data(chunk(), pc, save_path=expected)
    for path in result["umi_vis"]:
        assert (tmp_path / path).read_bytes() == expected.read_bytes()
    np.testing.assert_array_equal(pc, before)
    assert evaluator._STANDALONE_UMI_VISUALIZER is None
    disabled = artifacts.EpisodeArtifacts(tmp_path / "disabled", 2, {})
    disabled.umi(0, 1, chunk(), pc, renderer=evaluator.vis_umi_data)
    assert disabled.finish(False)["umi_vis_count"] == 0
    assert not (tmp_path / "disabled").exists()
