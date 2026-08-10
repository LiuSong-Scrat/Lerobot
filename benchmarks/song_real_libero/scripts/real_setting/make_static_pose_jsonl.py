#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write one exact identity camera pose for every RGB-D frame."
    )
    parser.add_argument("frames_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    if not args.frames_jsonl.is_file():
        raise FileNotFoundError(args.frames_jsonl)

    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    output_records: list[dict] = []
    for fallback_index, line in enumerate(
        args.frames_jsonl.read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        frame = json.loads(line)
        output_records.append(
            {
                "record_index": int(frame.get("index", fallback_index)),
                "timestamp_ms": frame.get("timestamp_ms"),
                "camera_to_tracking": identity,
                "tracking_source": "static_identity_camera",
                "valid": True,
                "transform_direction": "camera_to_tracking",
            }
        )

    if not output_records:
        raise RuntimeError(f"No frames found in {args.frames_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records),
        encoding="utf-8",
    )
    print(f"Wrote {len(output_records)} static poses: {args.output_jsonl}")


if __name__ == "__main__":
    main()
