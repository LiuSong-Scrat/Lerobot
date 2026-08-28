#!/usr/bin/env python
"""Release clean checkpoint pages from the Linux file cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--readvise-seconds", type=float, default=60.0)
    return parser.parse_args()


def ready_checkpoints(train_root: Path) -> list[Path]:
    checkpoints = []
    for variant_dir in train_root.iterdir() if train_root.is_dir() else ():
        checkpoint_root = variant_dir / "checkpoints"
        for checkpoint in checkpoint_root.iterdir() if checkpoint_root.is_dir() else ():
            if not checkpoint.name.isdigit() or not checkpoint.is_dir():
                continue
            model_dir = checkpoint / "pretrained_model"
            if all((model_dir / name).is_file() for name in REQUIRED_MODEL_FILES):
                checkpoints.append(checkpoint)
    return sorted(checkpoints)


def release_checkpoint_cache(checkpoint: Path) -> tuple[int, int]:
    files = 0
    bytes_advised = 0
    for path in checkpoint.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            files += 1
            bytes_advised += path.stat().st_size
        except (FileNotFoundError, PermissionError):
            continue
    return files, bytes_advised


def sweep(train_root: Path) -> dict[str, int | float]:
    checkpoints = ready_checkpoints(train_root)
    files = 0
    bytes_advised = 0
    for checkpoint in checkpoints:
        checkpoint_files, checkpoint_bytes = release_checkpoint_cache(checkpoint)
        files += checkpoint_files
        bytes_advised += checkpoint_bytes
    return {
        "timestamp": time.time(),
        "checkpoints": len(checkpoints),
        "files": files,
        "bytes_advised": bytes_advised,
    }


def sweep_due(
    train_root: Path,
    last_advised: dict[Path, float],
    *,
    now: float,
    readvise_seconds: float,
) -> dict[str, int | float]:
    checkpoints = ready_checkpoints(train_root)
    ready = set(checkpoints)
    for checkpoint in set(last_advised) - ready:
        last_advised.pop(checkpoint, None)
    due = [
        checkpoint
        for checkpoint in checkpoints
        if now - last_advised.get(checkpoint, float("-inf")) >= readvise_seconds
    ]
    files = 0
    bytes_advised = 0
    for checkpoint in due:
        checkpoint_files, checkpoint_bytes = release_checkpoint_cache(checkpoint)
        files += checkpoint_files
        bytes_advised += checkpoint_bytes
        last_advised[checkpoint] = now
    return {
        "timestamp": now,
        "checkpoints": len(checkpoints),
        "checkpoints_advised": len(due),
        "files": files,
        "bytes_advised": bytes_advised,
    }


def main() -> int:
    args = parse_args()
    last_advised: dict[Path, float] = {}
    while True:
        payload = sweep_due(
            args.train_root,
            last_advised,
            now=time.time(),
            readvise_seconds=max(1.0, args.readvise_seconds),
        )
        if payload["checkpoints_advised"]:
            print(json.dumps(payload, sort_keys=True), flush=True)
        if not args.watch:
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
