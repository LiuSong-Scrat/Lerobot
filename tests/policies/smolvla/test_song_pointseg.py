#!/usr/bin/env python

import json

import numpy as np
import torch

from lerobot.policies.smolvla.song_pointseg import (
    POINTSEG_CACHE_FIELDS,
    POINTSEG_CACHE_LABEL_FIELDS,
    POINTSEG_CACHE_VERSION,
    ROLE_FOREGROUND,
    ROLE_IGNORE,
    PseudoLabelConfig,
    SongPointSegCachedDataset,
    SongPointSegLoss,
    SongPointSegLossConfig,
    SongPointSegNet,
    SongTemporalPointCloudDataset,
    force_small_current_clouds_foreground,
    generate_pseudo_labels,
    generate_pseudo_labels_from_priors,
    matrix_to_pose9,
    pose9_to_matrix,
    refine_pseudo_labels_with_teacher,
    relative_poses_to_first,
)


def _pose_from_translation(xyz):
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
    return matrix_to_pose9(transform)


def test_relative_poses_to_first_has_identity_anchor():
    poses = torch.stack(
        [
            _pose_from_translation([0.2, -0.1, 0.0]),
            _pose_from_translation([0.5, -0.1, 0.1]),
            _pose_from_translation([0.5, 0.2, 0.1]),
        ],
        dim=0,
    )

    relative = relative_poses_to_first(poses.unsqueeze(0)).squeeze(0)
    first_matrix = pose9_to_matrix(relative[0])

    assert torch.allclose(first_matrix, torch.eye(4), atol=1e-5)
    assert torch.allclose(relative[1, :3], torch.tensor([0.3, 0.0, 0.1]), atol=1e-5)


def test_motion_residuals_separate_held_and_static_points():
    held_current = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32)
    static_current = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)
    current_xyz = torch.cat([held_current, static_current], dim=0)
    current_pc = torch.cat([current_xyz, torch.zeros(2, 3)], dim=-1).unsqueeze(0)

    future_pose = _pose_from_translation([0.20, 0.0, 0.0])
    future_poses = torch.stack([_pose_from_translation([0.0, 0.0, 0.0]), future_pose], dim=0).unsqueeze(0)
    static_future = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    future_xyz = torch.cat([held_current, static_future], dim=0)
    future_pc = torch.stack(
        [
            current_pc.squeeze(0),
            torch.cat([future_xyz, torch.zeros(2, 3)], dim=-1),
        ],
        dim=0,
    ).unsqueeze(0)
    future_is_pad = torch.zeros(1, 2, dtype=torch.bool)

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(nn_chunk_size=8, min_confidence=0.0),
    )

    held_res = pseudo["held_residual"].squeeze(0)
    static_res = pseudo["static_residual"].squeeze(0)
    assert held_res[0] < static_res[0]
    assert static_res[1] < held_res[1]


def test_generate_pseudo_labels_from_existing_priors():
    priors = torch.zeros(2, 16, 8, dtype=torch.float32)
    priors[..., 0] = torch.linspace(0.0, 0.4, 16)
    priors[..., 1] = priors[..., 0]
    priors[..., 3] = 0.05
    priors[..., 4] = 0.05
    priors[..., 6] = torch.exp(-priors[..., 3] / 0.05)
    priors[..., 7] = torch.exp(-priors[..., 4] / 0.05)

    pseudo = generate_pseudo_labels_from_priors(
        priors,
        config=PseudoLabelConfig(min_confidence=0.0, forced_foreground_min_score=0.0),
    )

    assert pseudo["priors"].shape == priors.shape
    assert pseudo["labels"].shape == priors.shape[:2]
    assert pseudo["weights"].shape == priors.shape[:2]
    assert pseudo["class_scores"].shape == (*priors.shape[:2], 2)


def test_small_cloud_binary_override_preserves_automatic_motion_roles():
    current_pc = torch.randn(1, 4, 6)
    original_roles = torch.tensor(
        [[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8], [0.2, 0.5, 0.3]]],
        dtype=torch.float32,
    )
    pseudo = {
        "labels": torch.zeros(1, 4, dtype=torch.long),
        "weights": torch.ones(1, 4),
        "class_scores": torch.zeros(1, 4, 2),
        "role_scores": original_roles.clone(),
        "foreground_score": torch.zeros(1, 4),
    }

    forced = force_small_current_clouds_foreground(pseudo, current_pc, configured_current_points=8)

    assert forced["labels"].eq(ROLE_FOREGROUND).all()
    assert torch.equal(forced["role_scores"], original_roles)


class _FakeBaseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_lengths):
        self.episode_lengths = episode_lengths
        self.index = []
        for episode_index, length in enumerate(episode_lengths):
            for frame_index in range(length):
                self.index.append((episode_index, frame_index))
        self.actions = []
        for episode_index, length in enumerate(episode_lengths):
            poses = []
            for frame_index in range(length):
                poses.append(torch.cat([_pose_from_translation([episode_index + frame_index * 0.1, 0.0, 0.0]), torch.zeros(1)]))
            self.actions.append(torch.stack(poses, dim=0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        episode_index, frame_index = self.index[idx]
        poses = self.actions[episode_index]
        chunk = []
        for offset in range(3):
            chunk.append(poses[min(frame_index + offset, poses.shape[0] - 1)])
        return {
            "episode_index": torch.tensor(episode_index),
            "frame_index": torch.tensor(frame_index),
            "action": torch.stack(chunk, dim=0),
        }


def test_temporal_point_cloud_dataset_shapes_and_episode_clamping(tmp_path):
    point_cloud_dir = tmp_path / "point_clouds"
    point_cloud_dir.mkdir()
    for episode_index, length in enumerate([3, 2]):
        clouds = np.zeros((length, 10, 6), dtype=np.float32)
        for frame_index in range(length):
            clouds[frame_index, :, 0] = episode_index * 10 + frame_index
        np.save(point_cloud_dir / f"episode_{episode_index:06d}.npy", clouds)

    dataset = SongTemporalPointCloudDataset(
        _FakeBaseDataset([3, 2]),
        point_cloud_dir=point_cloud_dir,
        future_offsets=(1, 2),
        current_points=4,
        future_points=5,
        seed=123,
    )

    last_ep0 = dataset[2]
    assert tuple(last_ep0["observation.point_cloud"].shape) == (4, 6)
    assert tuple(last_ep0["observation.point_cloud_future"].shape) == (3, 5, 6)
    assert tuple(last_ep0["future_ee_poses"].shape) == (3, 9)
    assert last_ep0["future_is_pad"].tolist() == [False, True, True]
    assert torch.allclose(pose9_to_matrix(last_ep0["future_ee_poses"][0]), torch.eye(4), atol=1e-5)

    first_ep1 = dataset[3]
    assert float(first_ep1["observation.point_cloud"][0, 0]) >= 10.0


def test_cached_pointseg_dataset_reads_sharded_memmap(tmp_path):
    cache_dir = tmp_path / "cache"
    shard_dir = cache_dir / "shard_000000"
    shard_dir.mkdir(parents=True)
    num_samples = 2
    n_points = 4

    np.save(shard_dir / "point_cloud.npy", np.ones((num_samples, n_points, 6), dtype=np.float16))
    np.save(shard_dir / "priors.npy", np.ones((num_samples, n_points, 8), dtype=np.float16) * 2)
    np.save(shard_dir / "labels.npy", np.array([[0, 1, -100, 0], [1, 1, 0, -100]], dtype=np.int16))
    np.save(shard_dir / "weights.npy", np.ones((num_samples, n_points), dtype=np.float16))
    np.save(shard_dir / "class_scores.npy", np.ones((num_samples, n_points, 2), dtype=np.float16))
    np.save(shard_dir / "role_scores.npy", np.ones((num_samples, n_points, 3), dtype=np.float16))
    np.save(shard_dir / "foreground_score.npy", np.ones((num_samples, n_points), dtype=np.float16))
    np.save(shard_dir / "episode_index.npy", np.array([3, 4], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.array([10, 11], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.array([0, 1], dtype=np.int64))
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "version": POINTSEG_CACHE_VERSION,
                "fields": list(POINTSEG_CACHE_FIELDS),
                "shards": [{"path": "shard_000000", "length": num_samples}],
            },
            f,
        )

    dataset = SongPointSegCachedDataset(cache_dir)
    sample = dataset[1]

    assert len(dataset) == num_samples
    assert tuple(sample["observation.point_cloud"].shape) == (n_points, 6)
    assert sample["observation.point_cloud"].dtype == torch.float32
    assert sample["pointseg.labels"].dtype == torch.int64
    assert sample["pointseg.labels"].tolist() == [1, 1, 0, -100]
    assert tuple(sample["pointseg.role_scores"].shape) == (n_points, 3)
    assert sample["episode_index"].item() == 4
    assert sample["dataset_index"].item() == 1


def test_index_cache_v3_preserves_automatic_role_scores(tmp_path):
    cache_dir = tmp_path / "cache"
    shard_dir = cache_dir / "shard_000000"
    shard_dir.mkdir(parents=True)
    np.save(shard_dir / "sample_offsets.npy", np.array([0, 3], dtype=np.int64))
    np.save(shard_dir / "point_indices.npy", np.array([2, 5, 7], dtype=np.int64))
    np.save(shard_dir / "labels.npy", np.array([1, 1, 0], dtype=np.int16))
    np.save(shard_dir / "weights.npy", np.ones(3, dtype=np.float16))
    np.save(shard_dir / "class_scores.npy", np.ones((3, 2), dtype=np.float16))
    role_scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=np.float16)
    np.save(shard_dir / "role_scores.npy", role_scores)
    np.save(shard_dir / "foreground_score.npy", role_scores.max(axis=-1))
    np.save(shard_dir / "episode_index.npy", np.array([0], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.array([4], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.array([9], dtype=np.int64))
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "version": POINTSEG_CACHE_VERSION,
                "cache_mode": "indices",
                "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
                "shards": [{"path": "shard_000000", "length": 1}],
            },
            f,
        )

    sample = SongPointSegCachedDataset(cache_dir)[0]

    assert sample["observation.point_cloud_indices"].tolist() == [2, 5, 7]
    assert torch.allclose(sample["pointseg.role_scores"], torch.from_numpy(role_scores.astype(np.float32)))


def test_song_pointseg_mlp_smoke_backward():
    bsize, n_points, future_points = 2, 64, 128
    current_pc = torch.rand(bsize, n_points, 6)
    current_pc[..., 3:] *= 255.0
    future_pc = torch.rand(bsize, 2, future_points, 6)
    future_pc[..., 3:] *= 255.0
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.1, 0.0, 0.0])],
        dim=0,
    )[None].repeat(bsize, 1, 1)
    future_is_pad = torch.zeros(bsize, 2, dtype=torch.bool)

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(nn_chunk_size=32, min_confidence=0.0, forced_foreground_min_score=0.0),
    )
    assert pseudo["labels"].shape == (bsize, n_points)
    assert not torch.equal(pseudo["labels"], torch.full_like(pseudo["labels"], ROLE_IGNORE))

    model = SongPointSegNet(backbone_type="mlp", hidden_dim=32)
    outputs = model(current_pc, future_pc, future_poses, future_is_pad, priors=pseudo["priors"])
    assert outputs["role_logits"].shape == (bsize, n_points, 2)
    assert outputs["operation_prob"].shape == (bsize, n_points)

    criterion = SongPointSegLoss(SongPointSegLossConfig(smooth_voxel_size=0.2))
    loss, _metrics = criterion(outputs, pseudo, current_pc)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_teacher_background_cannot_overwrite_uncertain_foreground_geometry():
    pseudo = {
        "labels": torch.tensor([[ROLE_FOREGROUND, ROLE_IGNORE, 0]], dtype=torch.long),
        "weights": torch.tensor([[0.7, 0.0, 0.1]], dtype=torch.float32),
        "foreground_score": torch.tensor([[0.6, 0.4, 0.01]], dtype=torch.float32),
        "class_scores": torch.tensor(
            [
                [
                    [0.1, 0.6],
                    [0.2, 0.4],
                    [0.9, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
    }
    teacher_logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]])

    refined = refine_pseudo_labels_with_teacher(
        pseudo,
        teacher_logits,
        config=PseudoLabelConfig(teacher_confidence=0.8),
    )

    assert refined["labels"][0, 0].item() == ROLE_FOREGROUND
    assert refined["labels"][0, 1].item() == ROLE_IGNORE
    assert refined["labels"][0, 2].item() == 0
