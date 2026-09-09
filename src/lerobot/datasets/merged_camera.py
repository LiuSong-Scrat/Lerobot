"""Explicit contracts for materialized, mixed-source camera-frame datasets.

The nominal FPS encodes integer frame offsets, not acquisition time. Source
camera poses are per-episode references, never one fictitious shared camera.
"""

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from datasets import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch

FRAME_GRID = "nominal_delta_index_grid_only_not_acquisition_fps"
OPTICAL_AXES = ["right", "down", "forward"]


def _read(root: Path, rel: str) -> dict:
    with (root / rel).open() as stream:
        return json.load(stream)


def is_merged_camera_metadata(info: dict) -> bool:
    return info.get("fps_semantics") == FRAME_GRID


def merged_cache_contract(info: dict) -> dict:
    return {
        key: info[key]
        for key in ("total_episodes", "total_frames", "fps_semantics", "camera_reference_alignment")
    }


def align_merged_rgb_features(policy_cfg, features: dict) -> None:
    """Retarget serialized external-camera feature names, not learned weights."""
    from lerobot.configs.types import FeatureType
    from lerobot.policies.smolvla.configuration_smolvla import canonical_rgb_camera_view_name

    if not policy_cfg.input_features:
        return
    aligned = {}
    for key, feature in policy_cfg.input_features.items():
        if feature.type is not FeatureType.VISUAL:
            aligned[key] = feature
            continue
        canonical = canonical_rgb_camera_view_name(key.rsplit(".", 1)[-1])
        matches = [
            name
            for name, value in features.items()
            if value.type is FeatureType.VISUAL
            and canonical_rgb_camera_view_name(name.rsplit(".", 1)[-1]) == canonical
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map pretrained image {key} into merged data: {matches}")
        aligned[matches[0]] = features[matches[0]]
    policy_cfg.input_features = aligned


def validate_merged_camera_reference(root: str | Path) -> dict:
    """Require positive, versioned evidence before accepting generic world sidecars."""
    root = Path(root)
    info = _read(root, "meta/info.json")
    manifest = _read(root, "merge_manifest.json")
    alignment = _read(root, "camera_alignment/report.json")
    if not is_merged_camera_metadata(info) or not info.get("mixed_source_fps"):
        raise ValueError("Expected the explicit merged frame-index dataset contract.")
    if manifest.get("status") != "complete" or alignment.get("status") != "complete":
        raise ValueError("Merged camera dataset publication is not complete.")
    if not alignment.get("all_sources_stored_rgb_axes_aligned"):
        raise ValueError("Not all merged source camera axes are aligned to saved RGB.")
    if info.get("camera_reference_alignment") != alignment.get("alignment_id"):
        raise ValueError("Merged camera alignment metadata versions disagree.")
    conventions = alignment.get("source_conventions", {})
    sources = manifest.get("sources", [])
    if not sources or sum(s["episodes"] for s in sources) != info["total_episodes"]:
        raise ValueError("Merged source episode counts are inconsistent.")
    for source in sources:
        if conventions.get(source["source_id"], {}).get("stored_rgb_axes") != OPTICAL_AXES:
            raise ValueError(f"Source {source['source_id']} is not right/down/forward optical.")
    for folder in ("world_ee_poses", "action_target_ee_poses", "episode_reference_poses"):
        meta = _read(root, folder + "/meta.json")
        if (
            meta.get("coordinate_frame") != "fixed_primary_camera_reference"
            or meta.get("camera_axes") != OPTICAL_AXES
            or meta.get("length_unit") != "meter"
        ):
            raise ValueError(f"Invalid fixed-camera frame/axes/unit in {folder}/meta.json.")
    rows = pq.ParquetFile(root / "meta/source_episodes.parquet").read().to_pylist()
    if sorted(r["episode_index"] for r in rows) != list(range(info["total_episodes"])):
        raise ValueError("Merged provenance must contain every episode exactly once.")
    by_source = {s["source_id"]: s for s in sources}
    for row in rows:
        source = by_source[row["source_id"]]
        e = row["episode_index"]
        if not source["episode_offset"] <= e < source["episode_offset"] + source["episodes"]:
            raise ValueError(f"Source/provenance range mismatch for episode {e}.")
        if row["merged_reference_frame"] != conventions[row["source_id"]]["reference"]:
            raise ValueError(f"Source/provenance camera reference mismatch for episode {e}.")
        if row["source_action_semantics"] != source["action_semantics"]:
            raise ValueError(f"Source target semantics mismatch for episode {e}.")
    # Recorded-demonstration targets remain explicitly distinguished from commands.
    return {"alignment_id": alignment["alignment_id"], "sources": sources}


def merged_source_group_ids(root: str | Path, episode_ids: list[int]) -> list[str]:
    rows = (
        pq.ParquetFile(Path(root) / "meta/source_episodes.parquet")
        .read(columns=["episode_index", "source_id"])
        .to_pylist()
    )
    groups = {int(r["episode_index"]): str(r["source_id"]) for r in rows}
    if len(groups) != len(rows):
        raise ValueError("Duplicate episode in source provenance.")
    return [groups[int(e)] for e in episode_ids]


class MergedCameraLeRobotDataset(LeRobotDataset):
    """Use native integer delta_indices; efficiently filter actual Parquet shards."""

    @cached_property
    def observation_states(self):
        # PointSeg's motion prior must not materialize a multi-million-row HF
        # state column for every requested episode. Read only its numeric shard.
        return _EpisodeObservationStates(self.root, self.meta)

    def load_hf_dataset(self) -> Dataset:
        if not is_merged_camera_metadata(self.meta.info):
            raise ValueError("MergedCameraLeRobotDataset requires frame-grid metadata.")
        if self.episodes is None:
            return super().load_hf_dataset()
        paths = sorted({str(self.root / self.meta.get_data_file_path(e)) for e in self.episodes})
        dataset = Dataset.from_parquet(
            paths,
            filters=pa_ds.field("episode_index").isin(self.episodes),
            features=get_hf_features_from_features(self.features),
        )
        dataset.set_transform(hf_transform_to_torch)
        return dataset


class _EpisodeObservationStates:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.metadata = metadata
        self.cache = {}

    def __getstate__(self):
        return {**self.__dict__, "cache": {}}

    def __getitem__(self, episode_index):
        e = int(episode_index)
        if e not in self.cache:
            table = pq.ParquetFile(self.root / self.metadata.get_data_file_path(e)).read(
                columns=["episode_index", "observation.state"]
            )
            table = table.filter(pc.equal(table["episode_index"], e))
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            if states.shape != (int(self.metadata.episodes[e]["length"]), 10):
                raise ValueError(f"Incomplete episode {e} in merged numeric state shard: {states.shape}")
            self.cache[e] = states
        return self.cache[e]
