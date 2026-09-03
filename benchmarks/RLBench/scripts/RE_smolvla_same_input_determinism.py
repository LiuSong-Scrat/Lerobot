#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from smolvla_model_inference import SmolVLA_ModelInference


def digest(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    sha = hashlib.sha256()
    sha.update(str(array.dtype).encode())
    sha.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    sha.update(array.tobytes())
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument(
        "--base-world-translation",
        default="0.2679615616798401,0.0063918232917785645,-0.7500323696136475",
        help="Comma-separated translation of T_base_world; RLBench base rotation is identity.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    saved = np.load(args.input)
    rgb = saved["model_front_rgb"]
    pose7 = np.asarray(saved["reset_eef_pose7_world"], dtype=np.float64)
    qx, qy, qz, qw = pose7[3:7]
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    base_world_translation = np.asarray(
        [float(value) for value in args.base_world_translation.split(",")],
        dtype=np.float64,
    )
    worldflow_pose9 = np.concatenate(
        [pose7[:3] + base_world_translation, rotation[:, 0], rotation[:, 1]]
    ).astype(np.float32)
    observation = {
        "front": rgb,
        "agentview": rgb,
        "observation.images.agentview": rgb,
        "point_cloud": saved["model_point_cloud"],
        "state": saved["model_state"],
        "worldflow.current_ee_pose": worldflow_pose9,
    }
    inference = SmolVLA_ModelInference(policy_path=args.policy, device="cuda")
    outputs = []
    for index in range(args.repeats):
        action = inference.predict_action_chunk_obs(
            observation,
            task="water plant",
            postprocess=True,
            state_pose_mode="identity",
            noise_seed=args.noise_seed,
        ).numpy()[0]
        outputs.append(action)
        np.save(args.output.with_name(f"{args.output.stem}_action_{index}.npy"), action)

    comparisons = []
    for index in range(1, len(outputs)):
        delta = np.abs(outputs[index] - outputs[0])
        comparisons.append(
            {
                "against_repeat_0": index,
                "exact_equal": bool(np.array_equal(outputs[index], outputs[0])),
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
            }
        )
    report = {
        "policy": str(args.policy.resolve()),
        "input": str(args.input.resolve()),
        "input_hashes": {
            "rgb": digest(rgb),
            "point_cloud": digest(observation["point_cloud"]),
            "state": digest(observation["state"]),
            "worldflow.current_ee_pose": digest(worldflow_pose9),
        },
        "noise_seed": args.noise_seed,
        "action_hashes": [digest(output) for output in outputs],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
