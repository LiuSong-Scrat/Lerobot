#!/usr/bin/env python3
"""Gate World/Ego interaction on comparable unweighted physical errors.

This reads the per-update physical endpoint diagnostics already emitted by
``train_song_benchmark.py``. Weighted optimization losses are deliberately not
used: changing a coefficient must never make a centimetre World trajectory
look equivalent to a millimetre Ego trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


METRICS = (
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
)
VALUE = re.compile(r"(" + "|".join(METRICS) + r"):([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--ego-trans-max-m", type=float, default=0.012)
    parser.add_argument("--world-trans-max-m", type=float, default=0.012)
    parser.add_argument("--world-to-ego-trans-ratio-max", type=float, default=1.5)
    parser.add_argument("--world-rot-max-deg", type=float, default=2.0)
    parser.add_argument("--world-to-ego-rot-ratio-max", type=float, default=1.5)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    records: list[dict[str, float]] = []
    for line in args.train_log.read_text(errors="replace").splitlines():
        record = {name: float(value) for name, value in VALUE.findall(line)}
        if set(record) == set(METRICS):
            records.append(record)
    if len(records) < args.window:
        raise RuntimeError(
            f"Need at least {args.window} complete physical-error records, found {len(records)}"
        )
    selected = records[-args.window :]
    summaries = {name: summarize([record[name] for record in selected]) for name in METRICS}
    ego_trans = summaries["action_endpoint_trans_err"]["mean"]
    ego_rot = summaries["action_endpoint_rot_err_deg"]["mean"]
    world_trans = summaries["worldflow_trans_err"]["mean"]
    world_rot = summaries["worldflow_rot_err_deg"]["mean"]
    ratios = {
        "world_to_ego_trans": world_trans / max(ego_trans, 1e-12),
        "world_to_ego_rot": world_rot / max(ego_rot, 1e-12),
    }
    checks = {
        "ego_translation_is_millimetre_scale": ego_trans <= args.ego_trans_max_m,
        "world_translation_is_millimetre_scale": world_trans <= args.world_trans_max_m,
        "translation_errors_are_comparable": (
            ratios["world_to_ego_trans"] <= args.world_to_ego_trans_ratio_max
        ),
        "world_rotation_is_precise": world_rot <= args.world_rot_max_deg,
        "rotation_errors_are_comparable": (
            ratios["world_to_ego_rot"] <= args.world_to_ego_rot_ratio_max
        ),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract": "unweighted physical endpoint errors; loss weights cannot satisfy this gate",
        "train_log": str(args.train_log.resolve()),
        "records_total": len(records),
        "window": args.window,
        "summaries": summaries,
        "ratios": ratios,
        "checks": checks,
        "thresholds": {
            "ego_trans_max_m": args.ego_trans_max_m,
            "world_trans_max_m": args.world_trans_max_m,
            "world_to_ego_trans_ratio_max": args.world_to_ego_trans_ratio_max,
            "world_rot_max_deg": args.world_rot_max_deg,
            "world_to_ego_rot_ratio_max": args.world_to_ego_rot_ratio_max,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
