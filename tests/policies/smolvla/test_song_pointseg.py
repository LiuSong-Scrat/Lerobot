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
    _aggregate_motion_hypotheses,
    compose_point_cloud_views,
    force_small_current_clouds_foreground,
    generate_pseudo_labels,
    generate_pseudo_labels_from_priors,
    matrix_to_pose9,
    parse_camera_views,
    pose9_to_matrix,
    refine_pseudo_labels_with_teacher,
    relative_poses_to_first,
)


def test_multiview_point_cloud_composition_keeps_one_gripper_tail():
    first = np.zeros((10, 6), dtype=np.float32)
    second = np.ones((10, 6), dtype=np.float32)
    first[-2:, :3] = 7.0
    second[-2:, :3] = 9.0

    fused = compose_point_cloud_views([first, second], gripper_points=2, seed=11)

    assert fused.shape == first.shape
    assert np.all(fused[-2:, :3] == 7.0)
    assert np.any(np.all(fused[:-2] == 0.0, axis=1))
    assert np.any(np.all(fused[:-2] == 1.0, axis=1))


def test_parse_camera_views_accepts_overhead_and_hand_pair():
    assert parse_camera_views("agentview,robot0_eye_in_hand") == (
        "agentview",
        "robot0_eye_in_hand",
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


def test_motion_evidence_compares_hypotheses_in_the_same_future_frames():
    held = torch.tensor([[[0.01], [0.20], [0.01]]], dtype=torch.float32)
    static = torch.tensor([[[0.20], [0.01], [0.15]]], dtype=torch.float32)
    motion_weights = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32)
    valid = torch.ones(1, 3, dtype=torch.bool)

    held_residual, static_residual, residual_gap = _aggregate_motion_hypotheses(
        held,
        static,
        motion_weights,
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )

    # Independent minima would produce zero evidence because both minima are 0.01,
    # even though two same-frame comparisons favor the held hypothesis.
    assert torch.allclose(static.amin(dim=1) - held.amin(dim=1), torch.zeros(1, 1))
    assert residual_gap.item() > 0.4
    assert held_residual.item() < static_residual.item()


def test_motion_evidence_is_suppressed_when_pose_motion_is_uninformative():
    held = torch.tensor([[[0.01], [0.01]]], dtype=torch.float32)
    static = torch.tensor([[[0.20], [0.20]]], dtype=torch.float32)
    valid = torch.ones(1, 2, dtype=torch.bool)

    _held, _static, strong_gap = _aggregate_motion_hypotheses(
        held,
        static,
        torch.ones(1, 2),
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )
    _held, _static, weak_gap = _aggregate_motion_hypotheses(
        held,
        static,
        torch.full((1, 2), 0.01),
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )

    assert strong_gap.item() > 0.8
    assert weak_gap.item() < 0.02


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


def test_approached_target_does_not_require_static_motion():
    priors = torch.zeros(1, 2, 8, dtype=torch.float32)
    priors[..., 0] = 0.20
    priors[..., 1] = torch.tensor([[0.03, 0.40]])
    priors[..., 2] = torch.tensor([[0.12, 0.0]])
    priors[..., 3] = 0.20
    # Deliberately make the approached point inconsistent with the static
    # hypothesis.  Approach/contact evidence must still make it foreground.
    priors[..., 4] = 1.0
    priors[..., 5] = 0.0
    priors[..., 6] = 0.5
    priors[..., 7] = 1.0

    pseudo = generate_pseudo_labels_from_priors(priors)

    assert pseudo["foreground_score"][0, 0] > 0.7
    assert pseudo["foreground_score"][0, 0] > pseudo["foreground_score"][0, 1]
    assert torch.equal(pseudo["labels"], torch.full_like(pseudo["labels"], ROLE_IGNORE))


def test_near_contact_soft_score_survives_terminal_future_padding():
    current_xyz = torch.tensor([[[0.05, 0.0, 0.0], [0.50, 0.0, 0.0]]], dtype=torch.float32)
    current_pc = torch.cat([current_xyz, torch.zeros(1, 2, 3)], dim=-1)
    # The context cloud is padded, reproducing the final frame of an episode.
    future_pc = current_pc[:, None].repeat(1, 2, 1, 1)
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.0, 0.0, 0.0])]
    ).unsqueeze(0)
    future_is_pad = torch.tensor([[False, True]])

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(nn_chunk_size=8),
    )

    assert pseudo["foreground_score"][0, 0] > 0.3
    assert pseudo["foreground_score"][0, 0] > pseudo["foreground_score"][0, 1]


def test_small_cloud_fallback_does_not_harden_soft_labels():
    role_scores = torch.tensor(
        [
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.2],
                [0.0, 0.2, 0.7],
            ]
        ],
        dtype=torch.float32,
    )
    pseudo = {
        "labels": torch.full((1, 3), ROLE_IGNORE, dtype=torch.long),
        "weights": torch.zeros(1, 3),
        "class_scores": torch.zeros(1, 3, 2),
        "role_scores": role_scores.clone(),
        "foreground_score": role_scores.amax(dim=-1),
    }
    current_pc = torch.zeros(1, 3, 6)

    out = force_small_current_clouds_foreground(pseudo, current_pc, configured_current_points=8)

    assert torch.equal(out["labels"], torch.full((1, 3), ROLE_IGNORE, dtype=torch.long))
    assert torch.equal(out["weights"], pseudo["weights"])
    assert torch.equal(out["foreground_score"], pseudo["foreground_score"])
    assert torch.allclose(out["role_scores"], role_scores)


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
    assert last_ep0["pointseg_source_num_points"].item() == 10
    assert last_ep0["pointseg_sample_num_points"].item() == 4
    assert tuple(last_ep0["observation.point_cloud_future"].shape) == (5, 5, 6)
    assert tuple(last_ep0["future_ee_poses"].shape) == (5, 9)
    assert last_ep0["future_offsets"].tolist() == [0, -2, -1, 1, 2]
    assert last_ep0["future_is_pad"].tolist() == [False, False, False, True, True]
    assert torch.allclose(pose9_to_matrix(last_ep0["future_ee_poses"][0]), torch.eye(4), atol=1e-5)
    assert tuple(last_ep0["pointseg_trajectory_ee_poses"].shape) == (33, 9)
    assert tuple(last_ep0["pointseg_trajectory_offsets"].shape) == (33,)
    assert last_ep0["pointseg_trajectory_offsets"][0].item() == 0
    assert last_ep0["pointseg_trajectory_offsets"].min().item() == -2
    assert last_ep0["pointseg_trajectory_offsets"].max().item() == 0
    assert torch.allclose(
        pose9_to_matrix(last_ep0["pointseg_trajectory_ee_poses"][0]), torch.eye(4), atol=1e-5
    )

    first_ep1 = dataset[3]
    assert float(first_ep1["observation.point_cloud"][0, 0]) >= 10.0


def test_temporal_point_cloud_motion_prior_uses_achieved_state_not_action_target(tmp_path):
    point_cloud_dir = tmp_path / "point_clouds"
    point_cloud_dir.mkdir()
    np.save(
        point_cloud_dir / "episode_000000.npy",
        np.zeros((3, 4, 6), dtype=np.float32),
    )
    base = _FakeBaseDataset([3])
    base.observation_states = [
        torch.stack(
            [
                torch.cat([_pose_from_translation([x, 0.0, 0.0]), torch.zeros(1)])
                for x in (0.0, 0.1, 0.2)
            ]
        )
    ]
    base.actions = [
        torch.stack(
            [
                torch.cat([_pose_from_translation([x, 0.0, 0.0]), torch.zeros(1)])
                for x in (0.0, 1.0, 2.0)
            ]
        )
    ]
    dataset = SongTemporalPointCloudDataset(
        base,
        point_cloud_dir=point_cloud_dir,
        future_offsets=(1,),
        current_points=4,
        future_points=4,
    )

    relative = dataset._relative_temporal_poses(0, [0, 1])

    assert torch.allclose(relative[1, :3], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)


def test_full_episode_pose_trajectory_extends_target_evidence_without_changing_local_motion():
    current_xyz = torch.tensor([[[0.50, 0.0, 0.0], [0.80, 0.0, 0.0]]], dtype=torch.float32)
    current_pc = torch.cat([current_xyz, torch.zeros(1, 2, 3)], dim=-1)
    future_pc = current_pc[:, None].repeat(1, 2, 1, 1)
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.10, 0.0, 0.0])]
    ).unsqueeze(0)
    future_is_pad = torch.zeros(1, 2, dtype=torch.bool)
    trajectory_poses = torch.stack(
        [
            _pose_from_translation([0.0, 0.0, 0.0]),
            _pose_from_translation([0.25, 0.0, 0.0]),
            _pose_from_translation([0.50, 0.0, 0.0]),
        ]
    ).unsqueeze(0)

    local = generate_pseudo_labels(current_pc, future_pc, future_poses, future_is_pad)
    full = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        trajectory_poses=trajectory_poses,
    )

    assert torch.allclose(local["held_residual"], full["held_residual"])
    assert torch.allclose(local["static_residual"], full["static_residual"])
    assert full["min_traj_dist"][0, 0] < local["min_traj_dist"][0, 0]
    assert full["foreground_score"][0, 0] > local["foreground_score"][0, 0]
    assert full["foreground_score"][0, 0] > full["foreground_score"][0, 1]


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


def test_cached_pointseg_dataset_reads_index_cache_role_scores(tmp_path):
    cache_dir = tmp_path / "cache"
    shard_dir = cache_dir / "shard_000000"
    shard_dir.mkdir(parents=True)
    n_points = 3
    np.save(shard_dir / "sample_offsets.npy", np.array([0, n_points], dtype=np.int64))
    np.save(shard_dir / "point_indices.npy", np.array([1, 3, 5], dtype=np.int64))
    np.save(shard_dir / "labels.npy", np.array([1, 0, -100], dtype=np.int16))
    np.save(shard_dir / "weights.npy", np.ones(n_points, dtype=np.float16))
    np.save(shard_dir / "class_scores.npy", np.ones((n_points, 2), dtype=np.float16))
    role_scores = np.arange(n_points * 3, dtype=np.float16).reshape(n_points, 3)
    np.save(shard_dir / "role_scores.npy", role_scores)
    np.save(shard_dir / "foreground_score.npy", np.ones(n_points, dtype=np.float16))
    np.save(shard_dir / "episode_index.npy", np.array([0], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.array([2], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.array([7], dtype=np.int64))
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "version": POINTSEG_CACHE_VERSION,
                "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
                "cache_mode": "indices",
                "variable_num_points": True,
                "shards": [{"path": "shard_000000", "length": 1}],
            },
            f,
        )

    sample = SongPointSegCachedDataset(cache_dir)[0]

    assert sample["observation.point_cloud_indices"].tolist() == [1, 3, 5]
    assert torch.allclose(sample["pointseg.role_scores"], torch.as_tensor(role_scores, dtype=torch.float32))


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
        config=PseudoLabelConfig(
            nn_chunk_size=32,
            min_confidence=0.0,
            background_min_confidence=0.0,
            background_foreground_max=1.0,
            forced_foreground_min_score=0.0,
        ),
    )
    assert pseudo["labels"].shape == (bsize, n_points)
    assert torch.equal(pseudo["labels"], torch.full_like(pseudo["labels"], ROLE_IGNORE))
    assert torch.allclose(pseudo["class_scores"].sum(dim=-1), torch.ones(bsize, n_points), atol=1e-6)
    assert bool(((pseudo["foreground_score"] > 0) & (pseudo["foreground_score"] < 1)).any())

    model = SongPointSegNet(backbone_type="mlp", hidden_dim=32)
    outputs = model(current_pc, future_pc, future_poses, future_is_pad, priors=pseudo["priors"])
    assert outputs["role_logits"].shape == (bsize, n_points, 2)
    assert outputs["operation_prob"].shape == (bsize, n_points)

    criterion = SongPointSegLoss(SongPointSegLossConfig(smooth_voxel_size=0.2))
    loss, _metrics = criterion(outputs, pseudo, current_pc)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_teacher_refinement_preserves_soft_labels_and_never_hardens_targets():
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

    assert torch.equal(refined["labels"], pseudo["labels"])
    assert torch.allclose(refined["class_scores"].sum(dim=-1), torch.ones(1, 3), atol=1e-6)
    assert torch.all((refined["foreground_score"] >= 0) & (refined["foreground_score"] <= 1))
