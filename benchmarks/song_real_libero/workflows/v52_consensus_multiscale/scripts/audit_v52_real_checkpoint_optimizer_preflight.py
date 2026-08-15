#!/usr/bin/env python3
"""Load the real V32 source and verify V52 optimizer/gradient roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import draccus

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


EXPECTED_GROUP_NAMES = [
    "pretrained_ego_shared_nonpoint",
    "point_input_adaptation_path",
    "new_world_bidirectional",
    "world_physical_residual_head",
]
EXPECTED_LRS = [5e-9, 5e-8, 5e-9, 5e-9]
REQUIRED_POINT_PREFIXES = (
    "model.point_action_fusion.",
    "model.worldflow_branch.scene_encoder.",
    "model.worldflow_branch.scene_context_proj.",
    "model.worldflow_branch.point_action_adapter.",
    "model.ego_scene_to_expert.",
    "model.world_scene_to_expert.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repo-head", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if raw.pop("type") != "smolvla":
        raise RuntimeError("V52 source is not a SmolVLA checkpoint")
    config = draccus.decode(SmolVLAConfig, raw)
    config.device = "cpu"
    config.camera_views = "agentview,robot0_eye_in_hand"
    config.rgb_camera_views = "agentview"
    config.camera_view_fusion = "consensus_multiscale_novelty_union"
    config.camera_view_coarse_novelty_scale = 4.0
    config.multiview_input_pretrained_lr_multiplier = 0.005
    config.multiview_input_point_lr_multiplier = 0.05
    config.multiview_input_symmetric_point_path_adaptation = True
    config.worldflow_pretrained_lr_multiplier = 0.005
    config.worldflow_new_lr_multiplier = 0.005
    config.worldflow_residual_lr_multiplier = 0.005
    config.optimizer_lr = 1e-6

    policy = SmolVLAPolicy.from_pretrained(source, config=config)
    groups = policy.get_optim_params()
    trainable = {
        id(parameter): name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    group_names = [str(group["group_name"]) for group in groups]
    group_lrs = [float(group["lr"]) for group in groups]
    if group_names != EXPECTED_GROUP_NAMES or group_lrs != EXPECTED_LRS:
        raise RuntimeError(f"Unexpected V52 optimizer groups/LRs: {group_names}, {group_lrs}")

    seen: list[int] = []
    rows = []
    point_prefix_counts = None
    point_group_ids: set[int] = set()
    for group in groups:
        parameters = list(group["params"])
        ids = [id(parameter) for parameter in parameters]
        seen.extend(ids)
        names = [trainable[parameter_id] for parameter_id in ids]
        if group["group_name"] == "point_input_adaptation_path":
            point_group_ids = set(ids)
            point_prefix_counts = {
                prefix: sum(name.startswith(prefix) for name in names)
                for prefix in REQUIRED_POINT_PREFIXES
            }
            if not all(count > 0 for count in point_prefix_counts.values()):
                raise RuntimeError(f"A required Ego/World point path is absent: {point_prefix_counts}")
        rows.append(
            {
                "group_name": group["group_name"],
                "lr": float(group["lr"]),
                "tensor_count": len(parameters),
                "element_count": sum(parameter.numel() for parameter in parameters),
            }
        )

    exact_membership = len(seen) == len(set(seen)) == len(trainable) and set(seen) == set(trainable)
    if not exact_membership:
        raise RuntimeError("V52 optimizer grouping overlaps or omits real checkpoint parameters")
    world_only_point_role_ids = policy.get_worldflow_ego_tangent_world_only_parameter_ids()
    if not world_only_point_role_ids:
        raise RuntimeError("V52 resolved no World-only point-path gradient roles")
    if not world_only_point_role_ids < point_group_ids:
        raise RuntimeError("V52 World-only point roles must be inside the point-input group")
    world_group_names = {"new_world_bidirectional", "world_physical_residual_head"}
    protected_ids = {
        id(parameter)
        for group in groups
        if group["group_name"] not in world_group_names
        for parameter in group["params"]
    }
    protected_ids.difference_update(world_only_point_role_ids)
    ego_point_role_ids = point_group_ids - world_only_point_role_ids
    if protected_ids & world_only_point_role_ids:
        raise RuntimeError("World-only point paths remain Ego-tangent protected")
    if not ego_point_role_ids or not ego_point_role_ids <= protected_ids:
        raise RuntimeError("Ego point paths must remain Ego-tangent protected")

    payload = {
        "schema_version": 3,
        "source": str(source),
        "source_model_sha256": sha256(source / "model.safetensors"),
        "expected_repo_head": args.expected_repo_head,
        "camera_view_fusion": "consensus_multiscale_novelty_union",
        "camera_view_voxel_size_m": 0.01,
        "camera_view_coarse_novelty_scale": 4.0,
        "symmetric_point_path_adaptation": True,
        "group_rows": rows,
        "required_point_prefix_tensor_counts": point_prefix_counts,
        "trainable_tensor_count": len(trainable),
        "exact_no_overlap_no_omission": exact_membership,
        "world_only_point_role_tensor_count": len(world_only_point_role_ids),
        "ego_point_role_tensor_count": len(ego_point_role_ids),
        "ego_tangent_protected_tensor_count": len(protected_ids),
        "world_only_point_roles_excluded_from_ego_tangent_protection": True,
        "ego_point_roles_remain_ego_tangent_protected": True,
        "passes_v52_real_checkpoint_optimizer_preflight": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
