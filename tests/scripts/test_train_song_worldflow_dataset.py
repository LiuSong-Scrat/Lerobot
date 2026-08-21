from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    WorldFlowMemmapDataset as BenchmarkWorldFlowMemmapDataset,
    canonical_rgb_camera_name,
    make_song_training_ddp_kwargs,
    plan_cuda_allocator_lease_chunks,
    prune_committed_training_checkpoints,
    resolve_checkpoint_retention,
    resolve_cuda_allocator_lease_config,
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


def test_training_rgb_camera_validation_uses_semantic_aliases() -> None:
    assert canonical_rgb_camera_name("observation.images.overhead") == "agentview"
    assert canonical_rgb_camera_name("overview") == "agentview"
    assert canonical_rgb_camera_name("external") == "agentview"
    assert canonical_rgb_camera_name("observation.images.hand") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("wrist") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("custom_camera") == "custom_camera"


@pytest.mark.parametrize(
    ("vlm_backend", "expected_bucket_view"),
    [
        ("molmo2_full", True),
        ("molmo2_text", False),
        ("smolvlm", False),
        (None, False),
    ],
)
def test_full_molmo_alone_uses_ddp_gradient_bucket_views(
    vlm_backend: str | None,
    expected_bucket_view: bool,
) -> None:
    kwargs = make_song_training_ddp_kwargs(vlm_backend)

    assert kwargs.find_unused_parameters is True
    assert kwargs.gradient_as_bucket_view is expected_bucket_view
    serialized = kwargs.to_kwargs()
    assert serialized["find_unused_parameters"] is True
    if expected_bucket_view:
        assert serialized["gradient_as_bucket_view"] is True
    else:
        # False is Accelerate's default, so legacy backends retain the exact
        # pre-existing kwargs passed to DistributedDataParallel.
        assert "gradient_as_bucket_view" not in serialized


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
