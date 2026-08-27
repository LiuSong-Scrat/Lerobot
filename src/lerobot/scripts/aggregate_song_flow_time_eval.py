#!/usr/bin/env python

"""Aggregate comparable ``eval_song --flow_time_sweep`` JSON reports.

This utility is intentionally pure-JSON: it does not import Torch, construct a
policy, or touch a GPU.  It is designed for checkpoint comparisons where every
report was produced from the same strict fixed-anchor contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EULER_TIME_CODES = tuple(range(0, 1000, 100))
SWEEP_METRICS = (
    "loss_action",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
    "loss_worldflow_flow",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
)
ENDPOINT_METRICS = (
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
)
SAMPLED_ACTION_METRICS = (
    "sample_action_mse",
    "sample_action_translation_l2_m",
    "sample_action_rot6d_mse",
    "sample_action_gripper_mae_m",
)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a JSON number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _required_metric(metrics: dict[str, Any], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"eval_metrics.json is missing required metric {name!r}.")
    return _finite_float(metrics[name], name=f"metrics.{name}")


def _load_report(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "eval_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise TypeError(f"Expected an eval_song report with a metrics object: {path}")
    return path, report


def _validate_read_only_fixed_anchor(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    fixed_anchor = report.get("fixed_anchor")
    if not isinstance(fixed_anchor, dict) or fixed_anchor.get("enabled") is not True:
        raise ValueError(f"Flow-time comparison requires strict fixed-anchor output: {path}")
    if fixed_anchor.get("strict") is not True:
        raise ValueError(f"Fixed-anchor strict mode was not enabled: {path}")
    read_only_gates = {
        "gradients_created": False,
        "model_parameters_unchanged": True,
        "model_buffers_unchanged": True,
        "all_modules_eval_mode": True,
    }
    for name, expected in read_only_gates.items():
        if report.get(name) is not expected:
            raise ValueError(
                f"Read-only evaluation gate {name}={expected!r} is not proven in {path}."
            )
    comparison_contract = fixed_anchor.get("comparison_contract")
    if not isinstance(comparison_contract, dict):
        raise TypeError(f"Missing fixed_anchor.comparison_contract in {path}.")
    return fixed_anchor


def _comparison_signature(report: dict[str, Any], fixed_anchor: dict[str, Any]) -> dict[str, Any]:
    semantics = report.get("metric_semantics")
    if not isinstance(semantics, dict):
        semantics = {}
    return {
        "comparison_contract_sha256": fixed_anchor.get("comparison_contract_sha256"),
        "manifest_sha256": (fixed_anchor.get("manifest") or {}).get("sha256"),
        "raw_input_sha256": fixed_anchor.get("raw_input_sha256"),
        "preprocessed_input_sha256": fixed_anchor.get("preprocessed_input_sha256"),
        "sweep_rng_phase": (fixed_anchor.get("rng_phases") or {}).get("sweep"),
        "sample_rng_phase": (fixed_anchor.get("rng_phases") or {}).get("sample"),
        "evaluated_samples": report.get("evaluated_samples"),
        "flow_time_sweep_semantics": semantics.get("flow_time_sweep"),
        "sample_action_semantics": semantics.get("sample_action_mse"),
    }


def _sweep_values(metrics: dict[str, Any], metric_name: str) -> list[float]:
    return [
        _required_metric(metrics, f"flow_t{time_code:03d}_{metric_name}")
        for time_code in EULER_TIME_CODES
    ]


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _endpoint_euler_proxies(metrics: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Convert one-step-clean endpoint errors to Euler local-step proxies.

    At grid point ``t_i``, the reported one-step clean-endpoint error scales as
    ``(1-t_i) * velocity_error`` (exact for Euclidean translation/gripper and
    approximate after pose9 rotation projection).  A production Euler update
    uses ``dt * velocity_error``.  Therefore ``dt / (1-t_i)`` converts each
    endpoint diagnostic to its local Euler-step contribution.  Summing norms
    is a conservative terminal-error proxy, not an actual rollout endpoint.
    """

    dt = 1.0 / len(EULER_TIME_CODES)
    weights = [dt / (1.0 - time_code / 1000.0) for time_code in EULER_TIME_CODES]
    weight_sum = math.fsum(weights)
    local_sum: dict[str, float] = {}
    normalized_mean: dict[str, float] = {}
    for metric_name in ENDPOINT_METRICS:
        values = _sweep_values(metrics, metric_name)
        weighted_sum = math.fsum(weight * value for weight, value in zip(weights, values, strict=True))
        local_sum[metric_name] = weighted_sum
        normalized_mean[metric_name] = weighted_sum / weight_sum
    return local_sum, normalized_mean


def _sampled_action_metrics(metrics: dict[str, Any]) -> dict[str, float] | None:
    present = [name in metrics for name in SAMPLED_ACTION_METRICS]
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, exists in zip(SAMPLED_ACTION_METRICS, present, strict=True) if not exists]
        raise KeyError(f"Incomplete sampled-action metric group; missing {missing}.")
    return {name: _required_metric(metrics, name) for name in SAMPLED_ACTION_METRICS}


def aggregate_flow_time_reports(path_values: list[str | Path]) -> dict[str, Any]:
    """Load, validate, and aggregate comparable eval_song JSON reports."""

    if not path_values:
        raise ValueError("At least one eval_metrics.json path is required.")
    loaded = [_load_report(path) for path in path_values]
    validated = [
        (path, report, _validate_read_only_fixed_anchor(path, report))
        for path, report in loaded
    ]
    reference_signature = _comparison_signature(validated[0][1], validated[0][2])
    if any(value is None for value in reference_signature.values()):
        raise ValueError(
            "The reference report lacks one or more JSON comparability gates: "
            f"{reference_signature}."
        )
    for path, report, fixed_anchor in validated[1:]:
        signature = _comparison_signature(report, fixed_anchor)
        if signature != reference_signature:
            differing = {
                name: {"reference": reference_signature[name], "actual": signature[name]}
                for name in reference_signature
                if signature[name] != reference_signature[name]
            }
            raise ValueError(f"Evaluation reports are not comparable ({path}): {differing}")

    loss_contract = validated[0][2].get("loss_contract")
    if not isinstance(loss_contract, dict):
        raise TypeError("Reference report lacks fixed_anchor.loss_contract.")
    worldflow_weight = _finite_float(
        loss_contract.get("worldflow_loss_weight"),
        name="fixed_anchor.loss_contract.worldflow_loss_weight",
    )

    rows: list[dict[str, Any]] = []
    sampled_presence: list[bool] = []
    for path, report, _fixed_anchor in validated:
        metrics = report["metrics"]
        sweep = {name: _sweep_values(metrics, name) for name in SWEEP_METRICS}
        t0 = {name: values[0] for name, values in sweep.items()}
        grid_mean = {name: _mean(values) for name, values in sweep.items()}
        t0["joint_velocity_objective"] = (
            t0["loss_action"] + worldflow_weight * t0["loss_worldflow_flow"]
        )
        grid_mean["joint_velocity_objective"] = (
            grid_mean["loss_action"]
            + worldflow_weight * grid_mean["loss_worldflow_flow"]
        )
        endpoint_sum, endpoint_weighted_mean = _endpoint_euler_proxies(metrics)
        sampled = _sampled_action_metrics(metrics)
        sampled_presence.append(sampled is not None)
        rows.append(
            {
                "name": path.parent.name,
                "report": str(path),
                "checkpoint": report.get("checkpoint"),
                "t0": t0,
                "euler_grid_mean_t000_to_t900": grid_mean,
                "euler_endpoint_local_error_sum_proxy": endpoint_sum,
                "euler_endpoint_weighted_mean": endpoint_weighted_mean,
                "sampled_action": sampled,
            }
        )
    if any(sampled_presence) and not all(sampled_presence):
        raise ValueError(
            "sample_action metrics are present in only some reports; rerun every arm with "
            "the same --sample_action_mse and --sample_action_noise_mode settings."
        )

    return {
        "schema_version": 1,
        "protocol": {
            "euler_times": [time_code / 1000.0 for time_code in EULER_TIME_CODES],
            "excluded_boundary_diagnostic_time": 0.999,
            "t0_definition": "Direct flow_t000 metrics.",
            "grid_mean_definition": (
                "Arithmetic mean over the ten production Euler query times t=0.0,...,0.9; "
                "flow_t999 is excluded."
            ),
            "joint_velocity_objective_definition": (
                "loss_action + worldflow_loss_weight * loss_worldflow_flow; PointSeg is "
                "time-independent and excluded."
            ),
            "euler_endpoint_local_error_sum_proxy_definition": (
                "sum_i [dt/(1-t_i)] * one_step_clean_endpoint_error(t_i). This is a "
                "conservative local-error accumulation proxy, not an actual Euler rollout."
            ),
            "euler_endpoint_weighted_mean_definition": (
                "The same dt/(1-t_i)-weighted endpoint proxy normalized by the sum of weights."
            ),
        },
        "comparability": {
            "json_gates_passed": True,
            "signature": reference_signature,
            "limitations": [
                "JSON cannot prove that every checkpoint used the same action/world noise-prior "
                "implementation; verify that config/code invariant separately.",
                "sample_action_mse is a full denoising-path metric but is only one fixed noise draw "
                "unless evaluation is repeated over multiple sample seeds.",
                "Ordinary native loss/loss_action/loss_worldflow_flow are intentionally not "
                "aggregated: checkpoints with different flow_time_sampling draw different times.",
                "Translation metres, rotation degrees, gripper metres, and MSE are separate units "
                "and must not be added to one scalar.",
            ],
        },
        "loss_contract": loss_contract,
        "rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        help="eval_metrics.json files, or directories containing that file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON. Without this option the result is printed only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate = aggregate_flow_time_reports(args.reports)
    rendered = json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
