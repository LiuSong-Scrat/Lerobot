from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    WorldFlowMemmapDataset as BenchmarkWorldFlowMemmapDataset,
    canonical_rgb_camera_name,
    exact_global_batch_active_ranks,
    exact_global_batch_manifest,
    exact_global_batch_rank_loss_scale,
    make_song_training_ddp_kwargs,
    plan_cuda_allocator_lease_chunks,
    prune_committed_training_checkpoints,
    resolve_checkpoint_retention,
    resolve_cuda_allocator_lease_config,
    resolve_exact_global_batch_plan,
    write_exact_global_batch_manifest,
)
from lerobot.scripts.train_song import WorldFlowMemmapDataset
from lerobot.scripts.train_song_libero import (
    WorldFlowMemmapDataset as LiberoWorldFlowMemmapDataset,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, frame_index: int = 1, chunk_size: int = 4) -> None:
        self.frame_index = frame_index
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        assert index == 0
        return {
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(self.frame_index),
            "action": torch.zeros(self.chunk_size, 10),
        }


def _pose_sequence(offset: float) -> np.ndarray:
    poses = np.zeros((3, 9), dtype=np.float32)
    poses[:, 0] = np.arange(3, dtype=np.float32) + offset
    poses[:, 3] = 1.0
    poses[:, 7] = 1.0
    return poses


def _write_robot_base_worldflow_meta(base_dir, target_dir) -> None:
    (base_dir / "meta.json").write_text(
        json.dumps({"coordinate_frame": "robot_base"}),
        encoding="utf-8",
    )
    (target_dir / "meta.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "robot_base",
                "target_semantics": "commanded_eef_pose",
            }
        ),
        encoding="utf-8",
    )

def test_training_rgb_camera_validation_uses_semantic_aliases() -> None:
    assert canonical_rgb_camera_name("observation.images.overhead") == "agentview"
    assert canonical_rgb_camera_name("overview") == "agentview"
    assert canonical_rgb_camera_name("external") == "agentview"
    assert canonical_rgb_camera_name("observation.images.hand") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("wrist") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("custom_camera") == "custom_camera"


@pytest.mark.parametrize(
    ("vlm_backend", "expected_full_optimization"),
    [
        ("molmo2_full", True),
        ("molmo2_text", False),
        ("smolvlm", False),
        (None, False),
    ],
)
def test_full_molmo_alone_disables_unused_scan_and_uses_ddp_gradient_bucket_views(
    vlm_backend: str | None,
    expected_full_optimization: bool,
) -> None:
    kwargs = make_song_training_ddp_kwargs(vlm_backend)

    assert kwargs.find_unused_parameters is not expected_full_optimization
    assert kwargs.gradient_as_bucket_view is expected_full_optimization
    assert kwargs.static_graph is False
    serialized = kwargs.to_kwargs()
    if expected_full_optimization:
        assert serialized["gradient_as_bucket_view"] is True
        assert "find_unused_parameters" not in serialized
        assert "static_graph" not in serialized
    else:
        assert serialized["find_unused_parameters"] is True
        # False is Accelerate's default, so legacy backends retain their
        # previous dynamic-graph and non-bucket-view behavior.
        assert "gradient_as_bucket_view" not in serialized
        assert "static_graph" not in serialized


def test_eight_rank_b16_exact_global_batch_is_a_192_sample_ddp_mean() -> None:
    plan = resolve_exact_global_batch_plan(
        global_batch_size=192,
        batch_size=16,
        gradient_accumulation_steps=2,
        world_size=8,
    )
    assert plan is not None
    assert plan.full_micro_steps == 1
    assert plan.partial_micro_step_index == 1
    assert plan.partial_active_ranks == 4
    assert plan.physical_forward_samples_per_optimizer_step == 256
    assert plan.discarded_samples_per_optimizer_step == 64
    assert plan.valid_loss_scale == pytest.approx(128 / 192)

    partial_rank_uses = [0] * 8
    for optimizer_step in range(8):
        assert exact_global_batch_active_ranks(
            plan,
            optimizer_step=optimizer_step,
            micro_step=0,
        ) == tuple(range(8))
        partial_ranks = exact_global_batch_active_ranks(
            plan,
            optimizer_step=optimizer_step,
            micro_step=1,
        )
        assert partial_ranks == tuple((optimizer_step + offset) % 8 for offset in range(4))
        for rank in partial_ranks:
            partial_rank_uses[rank] += 1

        active_slots = 0
        accumulated_rank_scales = 0.0
        for micro_step in range(2):
            for rank in range(8):
                loss_scale = exact_global_batch_rank_loss_scale(
                    plan,
                    optimizer_step=optimizer_step,
                    micro_step=micro_step,
                    rank=rank,
                )
                active_slots += int(loss_scale > 0.0)
                accumulated_rank_scales += loss_scale

        assert active_slots * 16 == 192
        # DDP divides the summed rank gradients by W=8. The resulting total
        # coefficient is exactly one, hence an exact mean over 192 samples.
        assert accumulated_rank_scales / 8 == pytest.approx(1.0)

    assert partial_rank_uses == [4] * 8


def test_exact_global_batch_preserves_legacy_default_and_supports_divisible_batches() -> None:
    assert (
        resolve_exact_global_batch_plan(
            global_batch_size=None,
            batch_size=1,
            gradient_accumulation_steps=28,
            world_size=7,
        )
        is None
    )

    plan = resolve_exact_global_batch_plan(
        global_batch_size=192,
        batch_size=1,
        gradient_accumulation_steps=24,
        world_size=8,
    )
    assert plan is not None
    assert plan.full_micro_steps == 24
    assert plan.partial_micro_step_index is None
    assert plan.partial_active_ranks == 0
    assert plan.physical_forward_samples_per_optimizer_step == 192
    assert plan.discarded_samples_per_optimizer_step == 0
    assert plan.valid_loss_scale == pytest.approx(1 / 24)
    assert exact_global_batch_active_ranks(
        plan,
        optimizer_step=100,
        micro_step=23,
    ) == tuple(range(8))


def test_exact_global_batch_rejects_unrepresentable_schedules() -> None:
    with pytest.raises(ValueError, match="requires gradient_accumulation_steps=28"):
        resolve_exact_global_batch_plan(
            global_batch_size=192,
            batch_size=1,
            gradient_accumulation_steps=27,
            world_size=7,
        )
    with pytest.raises(ValueError, match="cannot split a rank-local micro-batch"):
        resolve_exact_global_batch_plan(
            global_batch_size=193,
            batch_size=2,
            gradient_accumulation_steps=14,
            world_size=7,
        )
    with pytest.raises(ValueError, match="positive integer"):
        resolve_exact_global_batch_plan(
            global_batch_size=True,
            batch_size=1,
            gradient_accumulation_steps=1,
            world_size=1,
        )


def test_exact_global_batch_manifest_is_stable_and_resume_safe(tmp_path) -> None:
    plan = resolve_exact_global_batch_plan(
        global_batch_size=192,
        batch_size=1,
        gradient_accumulation_steps=28,
        world_size=7,
    )
    assert plan is not None

    manifest = exact_global_batch_manifest(plan)
    assert manifest["global_batch_size"] == 192
    assert manifest["physical_forward_samples_per_optimizer_step"] == 196
    assert manifest["discarded_for_gradient_samples_per_optimizer_step"] == 4
    assert manifest["valid_loss_scale_fraction"] == "7/192"
    assert manifest["all_ranks_forward_backward_every_micro_step"] is True
    assert manifest["sample_counter_increment_per_optimizer_step"] == 192
    assert manifest["scheduler_steps_per_optimizer_step"] == 1
    assert manifest["logged_loss_reduction"].startswith("global mean")

    path = write_exact_global_batch_manifest(tmp_path, plan)
    assert json.loads(path.read_text()) == manifest
    assert write_exact_global_batch_manifest(tmp_path, plan) == path
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    incompatible_plan = resolve_exact_global_batch_plan(
        global_batch_size=191,
        batch_size=1,
        gradient_accumulation_steps=28,
        world_size=7,
    )
    assert incompatible_plan is not None
    with pytest.raises(RuntimeError, match="manifest is incompatible"):
        write_exact_global_batch_manifest(tmp_path, incompatible_plan)


def test_cuda_allocator_lease_is_strictly_scoped_and_opt_in() -> None:
    malformed_env = {"MOLMO_FULL_CUDA_LEASE_ENABLE": "definitely"}
    assert (
        resolve_cuda_allocator_lease_config(
            environ=malformed_env,
            vlm_backend="molmo2_text",
            num_processes=8,
            device_type="cuda",
        )
        is None
    )
    assert (
        resolve_cuda_allocator_lease_config(
            environ=malformed_env,
            vlm_backend="molmo2_full",
            num_processes=1,
            device_type="cuda",
        )
        is None
    )
    assert (
        resolve_cuda_allocator_lease_config(
            environ={},
            vlm_backend="molmo2_full",
            num_processes=8,
            device_type="cuda",
        )
        is None
    )
    with pytest.raises(ValueError, match="explicit boolean"):
        resolve_cuda_allocator_lease_config(
            environ=malformed_env,
            vlm_backend="molmo2_full",
            num_processes=8,
            device_type="cuda",
        )


def test_checkpoint_retention_only_deletes_older_committed_numeric_directories(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint_paths = []
    for step in (100, 200, 300):
        checkpoint_path = checkpoints / f"{step:06d}"
        checkpoint_path.mkdir()
        (checkpoint_path / "sentinel").write_text(str(step))
        checkpoint_paths.append(checkpoint_path)
    (checkpoints / "notes").mkdir()
    (checkpoints / "last").symlink_to("000300")

    assert resolve_checkpoint_retention({}) is None
    assert resolve_checkpoint_retention({"MOLMO_CHECKPOINTS_TO_KEEP": "1"}) == 1
    with pytest.raises(ValueError, match="positive integer"):
        resolve_checkpoint_retention({"MOLMO_CHECKPOINTS_TO_KEEP": "0"})

    deleted = prune_committed_training_checkpoints(
        committed_checkpoint=checkpoint_paths[-1],
        keep=1,
    )
    assert deleted == (checkpoint_paths[1], checkpoint_paths[0])
    assert checkpoint_paths[-1].is_dir()
    assert (checkpoints / "last").resolve() == checkpoint_paths[-1]
    assert (checkpoints / "notes").is_dir()


def test_cuda_allocator_lease_config_and_chunk_boundary_are_cpu_only() -> None:
    config = resolve_cuda_allocator_lease_config(
        environ={
            "MOLMO_FULL_CUDA_LEASE_ENABLE": "1",
            "MOLMO_FULL_CUDA_LEASE_TARGET_GIB": "0.00000000931322574615478515625",
            "MOLMO_FULL_CUDA_LEASE_CHUNK_MIB": "0.000003814697265625",
            "MOLMO_FULL_CUDA_LEASE_HEADROOM_MIB": "0",
        },
        vlm_backend="molmo2_full",
        num_processes=8,
        device_type="cuda",
    )
    assert config is not None
    assert config.target_bytes == 10
    assert config.chunk_bytes == 4
    assert config.headroom_bytes == 0

    # Eight temporary bytes bring allocated from 2 to the target of 10.
    # Only six new driver bytes are needed because two cached bytes already
    # exist; the final chunk exercises the non-divisible boundary.
    assert plan_cuda_allocator_lease_chunks(
        allocated_bytes=2,
        reserved_bytes=4,
        free_bytes=7,
        target_bytes=10,
        chunk_bytes=3,
        headroom_bytes=1,
    ) == (3, 3, 2)
    assert plan_cuda_allocator_lease_chunks(
        allocated_bytes=2,
        reserved_bytes=10,
        free_bytes=0,
        target_bytes=10,
        chunk_bytes=3,
        headroom_bytes=1,
    ) == ()


def test_cuda_allocator_lease_rejects_one_byte_below_required_free_boundary() -> None:
    with pytest.raises(RuntimeError, match="Insufficient rank-local CUDA memory"):
        plan_cuda_allocator_lease_chunks(
            allocated_bytes=2,
            reserved_bytes=4,
            free_bytes=6,
            target_bytes=10,
            chunk_bytes=3,
            headroom_bytes=1,
        )

    with pytest.raises(ValueError, match="allocated_bytes"):
        plan_cuda_allocator_lease_chunks(
            allocated_bytes=5,
            reserved_bytes=4,
            free_bytes=10,
            target_bytes=10,
            chunk_bytes=3,
            headroom_bytes=1,
        )


def test_worldflow_dataset_uses_achieved_current_and_commanded_future_targets(tmp_path):
    achieved = _pose_sequence(offset=10.0)
    commanded = _pose_sequence(offset=20.0)
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", achieved)
    np.save(target_dir / "episode_000000.npy", commanded)

    wrapped = WorldFlowMemmapDataset(_TinyDataset(), tmp_path, chunk_size=4)
    item = wrapped[0]

    assert torch.equal(
        item["worldflow.current_ee_pose"],
        torch.from_numpy(achieved[1]),
    )
    assert torch.equal(
        item["worldflow.ee_poses"],
        torch.from_numpy(commanded[[1, 2, 2, 2]]),
    )
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


@pytest.mark.parametrize(
    "dataset_cls",
    [WorldFlowMemmapDataset, BenchmarkWorldFlowMemmapDataset, LiberoWorldFlowMemmapDataset],
)
def test_worldflow_dataset_applies_same_causal_action_offset_as_ego_chunk(tmp_path, dataset_cls):
    achieved = _pose_sequence(offset=10.0)
    commanded = _pose_sequence(offset=20.0)
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", achieved)
    np.save(target_dir / "episode_000000.npy", commanded)

    wrapped = dataset_cls(
        _TinyDataset(frame_index=0),
        tmp_path,
        chunk_size=4,
        action_start_offset=1,
    )
    item = wrapped[0]

    # Current observation remains frame 0; only Ego/World action targets shift.
    assert torch.equal(item["worldflow.current_ee_pose"], torch.from_numpy(achieved[0]))
    assert torch.equal(
        item["worldflow.ee_poses"],
        torch.from_numpy(commanded[[1, 2, 2, 2]]),
    )
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


def test_worldflow_dataset_rejects_mismatched_achieved_and_target_lengths(tmp_path):
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", _pose_sequence(offset=10.0))
    np.save(target_dir / "episode_000000.npy", _pose_sequence(offset=20.0)[:2])

    wrapped = WorldFlowMemmapDataset(_TinyDataset(), tmp_path, chunk_size=4)
    with pytest.raises(ValueError, match="achieved/target lengths differ"):
        wrapped[0]


def test_world_eef_trajectory_uses_robot_base_current_and_commanded_targets(tmp_path):
    base_poses = _pose_sequence(offset=30.0)
    base_targets = _pose_sequence(offset=40.0)
    base_dir = tmp_path / "world_base_ee_poses"
    target_dir = tmp_path / "world_base_action_target_ee_poses"
    base_dir.mkdir()
    target_dir.mkdir()
    _write_robot_base_worldflow_meta(base_dir, target_dir)
    np.save(base_dir / "episode_000000.npy", base_poses)
    np.save(target_dir / "episode_000000.npy", base_targets)

    wrapped = BenchmarkWorldFlowMemmapDataset(
        _TinyDataset(frame_index=1, chunk_size=4),
        tmp_path,
        chunk_size=4,
        target_type="world_eef_trajectory",
    )
    item = wrapped[0]

    assert torch.equal(item["worldflow.current_ee_pose"], torch.from_numpy(base_poses[1]))
    assert torch.equal(
        item["worldflow.eef_trajectory"],
        torch.from_numpy(base_targets[[1, 2, 2, 2]]),
    )
    assert "worldflow.ee_poses" not in item
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


def test_world_eef_trajectory_rejects_missing_base_target_sidecar(tmp_path):
    base_dir = tmp_path / "world_base_ee_poses"
    base_dir.mkdir()
    np.save(base_dir / "episode_000000.npy", _pose_sequence(offset=30.0))
    (base_dir / "meta.json").write_text(
        json.dumps({"coordinate_frame": "robot_base"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="fallbacks are forbidden"):
        BenchmarkWorldFlowMemmapDataset(
            _TinyDataset(),
            tmp_path,
            chunk_size=4,
            target_type="world_eef_trajectory",
        )


@pytest.mark.parametrize(
    "dataset_cls",
    [WorldFlowMemmapDataset, BenchmarkWorldFlowMemmapDataset, LiberoWorldFlowMemmapDataset],
)
def test_worldflow_dataset_can_require_command_target_sidecar(tmp_path, dataset_cls):
    achieved_dir = tmp_path / "world_ee_poses"
    achieved_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", _pose_sequence(offset=10.0))

    with pytest.raises(FileNotFoundError, match="action_target_ee_poses"):
        dataset_cls(
            _TinyDataset(),
            tmp_path,
            chunk_size=4,
            require_action_target_sidecar=True,
        )
