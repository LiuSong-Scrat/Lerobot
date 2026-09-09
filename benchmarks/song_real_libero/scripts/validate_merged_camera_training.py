"""Bounded real-data / checkpoint forward-backward smoke; no optimizer or saved weights."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.merged_camera import align_merged_rgb_features
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import validate_smolvla_worldflow_preprocessor
from lerobot.policies.smolvla.song_pointseg import pose9_to_matrix
from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    maybe_wrap_point_cloud_memmap_dataset,
    maybe_wrap_worldflow_dataset,
    maybe_wrap_pointseg_cache_dataset,
    make_song_train_collate_fn,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--online-labels", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    config = PreTrainedConfig.from_pretrained(
        str(args.policy_path),
        cli_overrides=[
            "--camera_views=front",
            "--rgb_camera_views=front",
            "--worldflow_reference_frame=pointcloud_reference_camera",
        ],
    )
    config.pretrained_path = args.policy_path
    episodes = [0, 8301, 10301, 11301, 11873, 13087]
    pipeline = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/merged_camera",
            root=str(args.dataset_root),
            episodes=episodes,
            use_imagenet_stats=False,
        ),
        policy=config,
    )
    raw = make_dataset(pipeline)
    dataset = maybe_wrap_pointseg_cache_dataset(raw, "", config) if args.online_labels else raw
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset, config)
    dataset = maybe_wrap_worldflow_dataset(dataset, config)
    collate = make_song_train_collate_fn(dataset)
    ei = np.asarray(raw.hf_dataset["episode_index"])
    checks = []
    examples = []
    for e in episodes:
        indices = np.flatnonzero(ei == e)
        n = len(indices)
        anchor = torch.from_numpy(
            np.load(args.dataset_root / f"episode_reference_poses/episode_{e:06d}.npy")
        ).float()
        for f in sorted({0, n // 2, n - 1}):
            item = dataset[int(indices[f])]
            current = pose9_to_matrix(item["worldflow.current_ee_pose"])
            targets = pose9_to_matrix(item["worldflow.eef_trajectory"])
            action = pose9_to_matrix(item["action"][:, :9])
            state = pose9_to_matrix(item["observation.state"][:9])
            err = max(
                float((anchor @ state - current).abs().max()), float((anchor @ action - targets).abs().max())
            )
            assert err < 2e-5 and torch.equal(item["action_is_pad"], item["worldflow.step_is_pad"])
            assert item["observation.point_cloud"].shape[-1] == 6
            checks.append(dict(episode=e, frame=f, maximum_error=err))
            if f == n // 2:
                examples.append(item)
    align_merged_rgb_features(config, dataset_to_policy_features(raw.meta.features))
    # strict=True also checks tied tensors / lazy materialization via the native loader.
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=config, strict=True)
    pre, _ = make_pre_post_processors(
        config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "features": {**config.input_features, **config.output_features},
                "stats": raw.meta.stats,
                "norm_map": config.normalization_mapping,
            },
        },
    )
    validate_smolvla_worldflow_preprocessor(pre)
    policy.train()
    losses = []
    # Pair unlike sources, including the 50k/3072 variable-length boundary.
    for start in range(0, len(examples), 2):
        policy.zero_grad(set_to_none=True)
        batch = pre(collate(copy.deepcopy(examples[start : start + 2])))
        loss, output = policy.forward(batch)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        entry = dict(
            episodes=episodes[start : start + 2], loss=float(loss.detach()), gradient_tensors=len(gradients)
        )
        losses.append(entry)
        print("FORWARD BACKWARD PASSED", entry, flush=True)
    report = dict(
        status="passed",
        strict_checkpoint_load=True,
        online_pointseg_labels=args.online_labels,
        dataset=str(args.dataset_root),
        checkpoint=str(args.policy_path),
        sources=5,
        episodes=episodes,
        pose_checks=checks,
        forward_backward=losses,
        optimizer_steps=0,
        peak_cuda_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("SMOKE COMPLETE", args.report, flush=True)


if __name__ == "__main__":
    main()
