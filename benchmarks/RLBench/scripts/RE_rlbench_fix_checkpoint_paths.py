#!/usr/bin/env python3
"""Rewrite local VLM paths stored in RLBench policy checkpoints."""

import argparse
import json
from pathlib import Path


def _rewrite_paths(value, model_path: str, weights_path: str) -> bool:
    changed = False
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "vlm_model_name" and item != model_path:
                value[key] = model_path
                changed = True
            elif key == "vlm_weights_path" and item != weights_path:
                value[key] = weights_path
                changed = True
            elif key == "tokenizer_name" and item != model_path:
                value[key] = model_path
                changed = True
            else:
                changed = _rewrite_paths(item, model_path, weights_path) or changed
    elif isinstance(value, list):
        for item in value:
            changed = _rewrite_paths(item, model_path, weights_path) or changed
    return changed


def _checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    if checkpoint_root.name == "pretrained_model":
        return [checkpoint_root]
    return sorted(path for path in checkpoint_root.glob("*/pretrained_model") if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/outputs/"
            "wep_vla_v041_rlbench_0805/checkpoints"
        ),
    )
    parser.add_argument(
        "--vlm-model-path",
        type=Path,
        default=Path("/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct"),
    )
    parser.add_argument(
        "--vlm-weights-path",
        type=Path,
        default=Path("/home/liusong/hf_models/smolvla_base"),
    )
    args = parser.parse_args()

    model_path = args.vlm_model_path.expanduser().resolve()
    weights_path = args.vlm_weights_path.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()

    required_files = [
        model_path / "config.json",
        weights_path / "config.json",
        weights_path / "model.safetensors",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local model files:\n" + "\n".join(missing))

    checkpoint_dirs = _checkpoint_dirs(checkpoint_root)
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No pretrained_model directories found under {checkpoint_root}")

    seen_dirs: set[Path] = set()
    changed_files = 0
    for checkpoint_dir in checkpoint_dirs:
        resolved_dir = checkpoint_dir.resolve()
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        for filename in ("config.json", "train_config.json", "policy_preprocessor.json"):
            path = checkpoint_dir / filename
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            if not _rewrite_paths(document, str(model_path), str(weights_path)):
                continue
            with path.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            changed_files += 1
            print(f"[updated] {path}")

    print(f"[done] checkpoint_root={checkpoint_root}")
    print(f"[done] model_path={model_path}")
    print(f"[done] weights_path={weights_path}")
    print(f"[done] changed_files={changed_files}")


if __name__ == "__main__":
    main()
