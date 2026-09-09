import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.merged_camera import (
    FRAME_GRID,
    OPTICAL_AXES,
    align_merged_rgb_features,
    merged_source_group_ids,
    validate_merged_camera_reference,
)
from lerobot.datasets.sampler import TaskBalancedFrameSampler
from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    WorldFlowMemmapDataset,
    maybe_wrap_pointseg_cache_dataset,
)


def write(root, rel, value):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def merged(tmp_path):
    sources = [
        dict(source_id="real", episodes=1, episode_offset=0, action_semantics="commanded_target"),
        dict(
            source_id="generated",
            episodes=1,
            episode_offset=1,
            action_semantics="recorded_demonstration_pose_sequence",
        ),
    ]
    info = dict(
        fps_semantics=FRAME_GRID,
        mixed_source_fps=True,
        total_episodes=2,
        total_frames=6,
        camera_reference_alignment="aligned",
    )
    write(tmp_path, "meta/info.json", info)
    write(tmp_path, "merge_manifest.json", dict(status="complete", sources=sources))
    write(
        tmp_path,
        "camera_alignment/report.json",
        dict(
            status="complete",
            alignment_id="aligned",
            all_sources_stored_rgb_axes_aligned=True,
            source_conventions={
                s["source_id"]: dict(reference="fixed_camera", stored_rgb_axes=OPTICAL_AXES) for s in sources
            },
        ),
    )
    for folder in ("world_ee_poses", "action_target_ee_poses", "episode_reference_poses"):
        write(
            tmp_path,
            folder + "/meta.json",
            dict(
                coordinate_frame="fixed_primary_camera_reference",
                camera_axes=OPTICAL_AXES,
                length_unit="meter",
            ),
        )
    rows = [
        dict(
            episode_index=i,
            source_id=s["source_id"],
            merged_reference_frame="fixed_camera",
            source_action_semantics=s["action_semantics"],
        )
        for i, s in enumerate(sources)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "meta/source_episodes.parquet")
    poses = np.zeros((3, 9), np.float32)
    poses[:, 3] = 1
    poses[:, 7] = 1
    poses[:, 0] = np.arange(3)
    for e in range(2):
        np.save(tmp_path / f"world_ee_poses/episode_{e:06d}.npy", poses)
        target = poses.copy()
        target[:, 1] = 0.2
        np.save(tmp_path / f"action_target_ee_poses/episode_{e:06d}.npy", target)
    return tmp_path


class Tiny(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return dict(episode_index=torch.tensor(1), frame_index=torch.tensor(1), action=torch.zeros(4, 10))


def test_explicit_merged_camera_target_and_padding(merged):
    ds = WorldFlowMemmapDataset(
        Tiny(),
        merged,
        chunk_size=4,
        target_type="world_eef_trajectory",
        reference_frame="pointcloud_reference_camera",
        action_start_offset=1,
        require_action_target_sidecar=True,
    )
    item = ds[0]
    assert ds.pose_dir == merged / "world_ee_poses"
    assert item["worldflow.current_ee_pose"][0] == 1
    assert torch.all(item["worldflow.eef_trajectory"][:, 0] == 2)
    assert torch.allclose(item["worldflow.eef_trajectory"][:, 1], torch.full((4,), 0.2))
    assert item["worldflow.step_is_pad"].tolist() == [False, True, True, True]


@pytest.mark.parametrize(
    "target,reference",
    [("world_eef_trajectory", "robot_base"), ("legacy_eef", "pointcloud_reference_camera")],
)
def test_no_robot_base_or_legacy_reinterpretation(merged, target, reference):
    with pytest.raises(ValueError, match="reinterpretation"):
        WorldFlowMemmapDataset(Tiny(), merged, chunk_size=4, target_type=target, reference_frame=reference)


@pytest.mark.parametrize(
    "mutation", ["axes", "version", "incomplete", "unit", "provenance", "missing_target"]
)
def test_camera_contract_fail_closed(merged, mutation):
    if mutation in ("axes", "version", "incomplete"):
        path = merged / "camera_alignment/report.json"
        d = json.loads(path.read_text())
        if mutation == "axes":
            d["all_sources_stored_rgb_axes_aligned"] = False
        if mutation == "version":
            d["alignment_id"] = "old"
        if mutation == "incomplete":
            d["status"] = "publishing"
        path.write_text(json.dumps(d))
    elif mutation == "unit":
        path = merged / "world_ee_poses/meta.json"
        d = json.loads(path.read_text())
        d["length_unit"] = "mm"
        path.write_text(json.dumps(d))
    elif mutation == "provenance":
        p = merged / "meta/source_episodes.parquet"
        rows = pq.read_table(p).to_pylist()
        rows[1]["merged_reference_frame"] = "robot_base"
        pq.write_table(pa.Table.from_pylist(rows), p)
    else:
        (merged / "action_target_ee_poses/meta.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_merged_camera_reference(merged)


def test_feature_alias_changes_no_dimensions():
    state = PolicyFeature(type=FeatureType.STATE, shape=(10,))
    image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    cfg = SimpleNamespace(input_features={"observation.state": state, "observation.images.agentview": image})
    align_merged_rgb_features(cfg, {"observation.state": state, "observation.images.front": image})
    assert cfg.input_features == {"observation.state": state, "observation.images.front": image}
    cfg.input_features = {"observation.images.robot0_eye_in_hand": image}
    with pytest.raises(ValueError, match="uniquely"):
        align_merged_rgb_features(cfg, {"observation.images.front": image})


def test_source_sampling_not_frame_frequency(merged):
    groups = merged_source_group_ids(merged, [0, 1])
    sampler = TaskBalancedFrameSampler([0, 2], [2, 10], groups, shuffle=False)
    indices = list(sampler)
    assert sum(i < 2 for i in indices) == 5
    assert sum(i >= 2 for i in indices) == 5


def test_old_cache_rejected_before_loading(merged):
    cache = merged / "old_cache"
    write(cache, "manifest.json", {"num_samples": 6})
    ds = SimpleNamespace(meta=SimpleNamespace(info=json.loads((merged / "meta/info.json").read_text())))
    with pytest.raises(ValueError, match="not certified"):
        maybe_wrap_pointseg_cache_dataset(ds, str(cache), SimpleNamespace(pointseg_enable=True))


def test_episode_states_read_numeric_shard_only(tmp_path):
    from lerobot.datasets.merged_camera import _EpisodeObservationStates

    rows = [
        dict(episode_index=e, **{"observation.state": [float(e)] * 10, "unused_image": b"not an image"})
        for e in (0, 1, 1)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "shard.parquet")
    meta = SimpleNamespace(
        get_data_file_path=lambda e: "shard.parquet", episodes={0: {"length": 1}, 1: {"length": 2}}
    )
    states = _EpisodeObservationStates(tmp_path, meta)
    np.testing.assert_array_equal(states[1], np.ones((2, 10), dtype=np.float32))
    assert states[1] is states[1]
    assert states.__getstate__()["cache"] == {}
    assert states[0].shape == (1, 10)
    meta.episodes[2] = {"length": 1}
    with pytest.raises(ValueError, match="Incomplete episode"):
        states[2]
