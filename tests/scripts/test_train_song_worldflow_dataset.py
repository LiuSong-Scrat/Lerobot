from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    WorldFlowMemmapDataset as BenchmarkWorldFlowMemmapDataset,
    _paired_pointseg_cache_contract_mismatches,
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


def _pointseg_manifest(points: int, *, nn_chunk_size: int = 512) -> dict:
    return {
        "version": 7,
        "num_samples": 20_744,
        "future_offsets": [1, 2, 4, 8, 16, 31],
        "temporal_offsets": [0, 1, 2, 4, 8, 16, 31],
        "trajectory_mode": "full_episode",
        "trajectory_offset_filtering": "future_only",
        "current_points": points,
        "future_points": points,
        "gripper_points": 500,
        "pseudo_label_policy": "soft_geometric_prior",
        "pseudo_label_config": {"held_sigma": 0.025, "nn_chunk_size": nn_chunk_size},
    }


def test_full_union_paired_cache_contract_allows_native_primary_point_count_and_compute_chunking():
    all_view = _pointseg_manifest(19_500, nn_chunk_size=512)
    primary = _pointseg_manifest(10_000, nn_chunk_size=1024)

    mismatches = _paired_pointseg_cache_contract_mismatches(
        all_view,
        primary,
        camera_view_fusion="full_union",
        num_views=2,
        gripper_points=500,
    )

    assert mismatches == {}


def test_full_union_paired_cache_contract_rejects_point_loss_or_semantic_prior_drift():
    all_view = _pointseg_manifest(19_499, nn_chunk_size=512)
    primary = _pointseg_manifest(10_000, nn_chunk_size=1024)
    primary["pseudo_label_config"]["held_sigma"] = 0.03

    mismatches = _paired_pointseg_cache_contract_mismatches(
        all_view,
        primary,
        camera_view_fusion="full_union",
        num_views=2,
        gripper_points=500,
    )

    assert mismatches["current_points"] == (19_499, 10_000)
    assert mismatches["future_points"] == (19_499, 10_000)
    assert "pseudo_label_config" in mismatches
