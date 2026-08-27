import json
import math

import pytest

from lerobot.scripts.aggregate_song_flow_time_eval import aggregate_flow_time_reports

SWEEP_METRICS = (
    "loss_action",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
    "loss_worldflow_flow",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
)


def _write_report(path, *, contract_hash="same", with_sample=True):
    metrics = {}
    for step, time_code in enumerate(range(0, 1000, 100), start=1):
        for metric_name in SWEEP_METRICS:
            metrics[f"flow_t{time_code:03d}_{metric_name}"] = float(step)
        # Must exist in input but must not influence the ten-step grid aggregates.
        for metric_name in SWEEP_METRICS:
            metrics[f"flow_t999_{metric_name}"] = 1_000_000.0
    if with_sample:
        metrics.update(
            {
                "sample_action_mse": 0.1,
                "sample_action_translation_l2_m": 0.2,
                "sample_action_rot6d_mse": 0.3,
                "sample_action_gripper_mae_m": 0.4,
            }
        )
    report = {
        "checkpoint": "/checkpoint",
        "evaluated_samples": 8,
        "metrics": metrics,
        "metric_semantics": {
            "flow_time_sweep": "same sweep",
            "sample_action_mse": "same sample mode",
        },
        "fixed_anchor": {
            "enabled": True,
            "strict": True,
            "comparison_contract_sha256": contract_hash,
            "manifest": {"sha256": "manifest"},
            "raw_input_sha256": "raw",
            "preprocessed_input_sha256": "processed",
            "rng_phases": {"sweep": 89, "sample": 71},
            "comparison_contract": {"schema_version": 1},
            "loss_contract": {"worldflow_loss_weight": 2.0},
        },
        "gradients_created": False,
        "model_parameters_unchanged": True,
        "model_buffers_unchanged": True,
        "all_modules_eval_mode": True,
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_aggregate_uses_production_grid_and_euler_endpoint_weights(tmp_path):
    report_path = tmp_path / "arm" / "eval_metrics.json"
    report_path.parent.mkdir()
    _write_report(report_path)

    aggregate = aggregate_flow_time_reports([report_path])
    row = aggregate["rows"][0]

    assert row["t0"]["loss_action"] == 1.0
    assert row["t0"]["joint_velocity_objective"] == 3.0
    assert row["euler_grid_mean_t000_to_t900"]["loss_action"] == 5.5
    assert row["euler_grid_mean_t000_to_t900"]["joint_velocity_objective"] == 16.5
    weights = [0.1 / (1.0 - step / 10.0) for step in range(10)]
    expected_sum = math.fsum(weight * value for weight, value in zip(weights, range(1, 11), strict=True))
    assert row["euler_endpoint_local_error_sum_proxy"]["action_endpoint_trans_err"] == pytest.approx(
        expected_sum
    )
    assert row["euler_endpoint_weighted_mean"]["action_endpoint_trans_err"] == pytest.approx(
        expected_sum / math.fsum(weights)
    )
    assert row["sampled_action"]["sample_action_mse"] == 0.1


def test_aggregate_rejects_different_fixed_anchor_contracts(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first, contract_hash="first")
    _write_report(second, contract_hash="second")

    with pytest.raises(ValueError, match="not comparable"):
        aggregate_flow_time_reports([first, second])


def test_aggregate_requires_sample_metrics_on_all_or_no_arms(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first, with_sample=True)
    _write_report(second, with_sample=False)

    with pytest.raises(ValueError, match="present in only some reports"):
        aggregate_flow_time_reports([first, second])
