#!/usr/bin/env python3
"""Append normalized checkpoint IDs to historical RLBench evaluation roots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"/checkpoints/(\d+)(?:/|$)")
TEXT_SUFFIXES = {".json", ".txt", ".log"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-base", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_eval_root(path: Path) -> bool:
    summary = read_json(path / "summary.json")
    return summary is not None and bool({"task_summaries", "task_output_dirs"} & summary.keys())


def checkpoint_for_root(path: Path) -> str | None:
    values: set[str] = set()
    for config in path.glob("*/config.json"):
        data = read_json(config)
        if data is None:
            continue
        match = CHECKPOINT_RE.search(str(data.get("policy_path", "")))
        if match:
            values.add(str(int(match.group(1))))
    if not values:
        for candidate in (path / "eval_config.json", path / "summary.json"):
            data = read_json(candidate)
            if data is None:
                continue
            match = CHECKPOINT_RE.search(json.dumps(data))
            if match:
                values.add(str(int(match.group(1))))
    if len(values) != 1:
        return None
    return values.pop()


def target_path(path: Path, checkpoint: str) -> Path:
    padded_suffix = re.compile(r"_0*" + re.escape(checkpoint) + r"$")
    name = padded_suffix.sub("_" + checkpoint, path.name)
    if name == path.name and not name.endswith("_" + checkpoint):
        name += "_" + checkpoint
    return path.with_name(name)


def replace_paths(root: Path, replacements: list[tuple[str, str]]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.eval_base.expanduser().resolve()
    roots = sorted((path.parent for path in base.rglob("summary.json") if is_eval_root(path.parent)), key=lambda item: len(item.parts), reverse=True)
    mapping: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for root in roots:
        checkpoint = checkpoint_for_root(root)
        if checkpoint is None:
            skipped.append(str(root))
            continue
        target = target_path(root, checkpoint)
        if target != root:
            mapping.append((root, target))

    old_paths = {old for old, _ in mapping}
    conflicts = [str(new) for old, new in mapping if new.exists() and new not in old_paths]
    if conflicts:
        raise RuntimeError("Destination already exists:\n" + "\n".join(conflicts))

    print(json.dumps({
        "eval_roots": len(roots),
        "rename_count": len(mapping),
        "skipped_without_unique_checkpoint": len(skipped),
        "apply": bool(args.apply),
    }, ensure_ascii=True))
    for old, new in mapping:
        print(f"{old} -> {new}")
    if skipped:
        print("[skipped]")
        print("\n".join(skipped))
    if not args.apply:
        return

    for old, new in mapping:
        old.rename(new)
    replacements = sorted(
        ((str(old), str(new)) for old, new in mapping),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for _, new in mapping:
        replace_paths(new, replacements)


if __name__ == "__main__":
    main()
