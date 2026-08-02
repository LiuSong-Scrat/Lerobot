from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    WorldFlowMemmapDataset as BenchmarkWorldFlowMemmapDataset,
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
