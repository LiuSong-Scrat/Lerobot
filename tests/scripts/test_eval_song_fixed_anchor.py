import hashlib
import json
import random

import numpy as np
import pytest
import torch

from lerobot.scripts import build_song_fixed_anchor_manifest as manifest_builder
from lerobot.scripts.eval_song import (
    _anchor_total_from_metrics,
    _changed_tensor_fingerprints,
    _fixed_anchor_phase_seed,
    _fixed_anchor_pointseg_aux_loss,
    _fixed_anchor_rng,
    _load_fixed_anchor_manifest,
    _tensor_state_fingerprints,
    _update_batch_fingerprint,
    _validate_fixed_anchor_loss_contract,
)


def test_fixed_anchor_rng_repeats_and_restores_all_cpu_rngs() -> None:
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    expected_python_state = random.getstate()
    expected_numpy_state = np.random.get_state()
    expected_torch_state = torch.random.get_rng_state().clone()

    draws = []
    for _ in range(2):
        with _fixed_anchor_rng(123456):
            draws.append((random.random(), np.random.rand(3), torch.rand(3)))

    assert draws[0][0] == draws[1][0]
    np.testing.assert_array_equal(draws[0][1], draws[1][1])
    torch.testing.assert_close(draws[0][2], draws[1][2], rtol=0.0, atol=0.0)
    assert random.getstate() == expected_python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == expected_numpy_state[0]
    np.testing.assert_array_equal(restored_numpy_state[1], expected_numpy_state[1])
    assert restored_numpy_state[2:] == expected_numpy_state[2:]
    torch.testing.assert_close(
        torch.random.get_rng_state(), expected_torch_state, rtol=0.0, atol=0.0
    )


def test_phase_seeds_are_stable_and_phase_separated() -> None:
    seed = _fixed_anchor_phase_seed(20260827, "forward", 3, 0)
    assert seed == _fixed_anchor_phase_seed(20260827, "forward", 3, 0)
    assert seed != _fixed_anchor_phase_seed(20260827, "preprocess", 3, 0)
    assert seed != _fixed_anchor_phase_seed(20260827, "forward", 4, 0)
    assert seed != _fixed_anchor_phase_seed(20260827, "forward", 3, 1)


def test_fixed_anchor_pointseg_aux_flag_is_scoped_without_changing_module_modes() -> None:
    class Conditioner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._force_pointseg_aux_loss = False
            self.batch_norm = torch.nn.BatchNorm1d(2)

    policy = torch.nn.Sequential(torch.nn.Linear(2, 2), Conditioner())
    policy.eval()
    conditioner = policy[1]
    modes_before = [module.training for module in policy.modules()]

    with _fixed_anchor_pointseg_aux_loss(policy, True):
        assert conditioner._force_pointseg_aux_loss is True
        assert [module.training for module in policy.modules()] == modes_before

    assert conditioner._force_pointseg_aux_loss is False
    assert [module.training for module in policy.modules()] == modes_before

    conditioner._force_pointseg_aux_loss = True
    with _fixed_anchor_pointseg_aux_loss(policy, False):
        assert conditioner._force_pointseg_aux_loss is True


def test_manifest_strict_validation_and_canonical_hash(tmp_path) -> None:
    path = tmp_path / "anchor.json"
    payload = {
        "schema_version": 1,
        "dataset_repo_id": "/dataset",
        "dataset_length": 10,
        "indices": [7, 1, 9],
        "loss_contract": {
            "pointseg_aux_loss_weight": 0.0005,
            "worldflow_loss_weight": 1.0,
            "worldflow_geo_loss_weight": 0.0,
            "worldflow_bridge_loss_weight": 0.0,
            "worldflow_equiv_loss_weight": 0.0,
        },
        "note": "ignored by the canonical comparison identity",
    }
    path.write_text(json.dumps(payload))

    manifest = _load_fixed_anchor_manifest(
        path,
        dataset_repo_id="/dataset",
        dataset_length=10,
        strict=True,
    )
    canonical = {
        "schema_version": 1,
        "dataset_repo_id": "/dataset",
        "dataset_length": 10,
        "indices": [7, 1, 9],
        "loss_contract": payload["loss_contract"],
    }
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["indices"] == [7, 1, 9]
    assert manifest["sha256"] == expected

    payload["indices"] = [7, 7]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        _load_fixed_anchor_manifest(
            path,
            dataset_repo_id="/dataset",
            dataset_length=10,
            strict=True,
        )


def test_batch_fingerprint_supports_bfloat16_and_detects_change() -> None:
    first = {
        "index": torch.tensor([1, 2]),
        "feature": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "task": ["one", "two"],
    }
    second = {
        **first,
        "feature": torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16),
    }
    hashes = []
    for batch in (first, first, second):
        hasher = hashlib.sha256()
        _update_batch_fingerprint(hasher, 1, batch)
        hashes.append(hasher.hexdigest())
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_anchor_total_formula_and_contract() -> None:
    total, gap = _anchor_total_from_metrics(
        {
            "loss": 0.00115,
            "loss_action": 0.0005,
            "loss_worldflow_flow": 0.0004,
            "loss_pointseg_aux": 0.5,
        },
        pointseg_weight=0.0005,
    )
    assert total == pytest.approx(0.0005 + 0.0004 + 0.0005 * 0.5)
    assert gap == pytest.approx(0.0, abs=1e-12)

    class Config:
        pointseg_aux_loss_weight = 0.0005
        worldflow_loss_weight = 1.0
        worldflow_geo_loss_weight = 0.0
        worldflow_bridge_loss_weight = 0.0
        worldflow_equiv_loss_weight = 0.0

    expected = {
        "pointseg_aux_loss_weight": 0.0005,
        "worldflow_loss_weight": 1.0,
        "worldflow_geo_loss_weight": 0.0,
        "worldflow_bridge_loss_weight": 0.0,
        "worldflow_equiv_loss_weight": 0.0,
    }
    assert _validate_fixed_anchor_loss_contract(
        Config(), expected, cli_expected_pointseg_weight=0.0005
    )["worldflow_loss_weight"] == 1.0
    Config.worldflow_geo_loss_weight = 0.1
    with pytest.raises(ValueError, match="contract drifted"):
        _validate_fixed_anchor_loss_contract(Config(), expected)


def test_tensor_state_fingerprint_detects_buffer_but_not_parameter_change() -> None:
    module = torch.nn.BatchNorm1d(3)
    buffers_before = _tensor_state_fingerprints(module, buffers=True)
    parameters_before = _tensor_state_fingerprints(module, buffers=False)

    with torch.no_grad():
        module.running_mean.add_(1.0)
    buffers_after = _tensor_state_fingerprints(module, buffers=True)
    assert _changed_tensor_fingerprints(buffers_before, buffers_after) == ["running_mean"]
    assert not _changed_tensor_fingerprints(
        parameters_before, _tensor_state_fingerprints(module, buffers=False)
    )


def test_manifest_builder_is_deterministic_and_records_loss_contract(monkeypatch) -> None:
    class Metadata:
        total_frames = 9
        episodes = [
            {"dataset_from_index": 0, "dataset_to_index": 4},
            {"dataset_from_index": 4, "dataset_to_index": 9},
        ]

    monkeypatch.setattr(manifest_builder, "LeRobotDatasetMetadata", lambda *_args, **_kwargs: Metadata())
    kwargs = {
        "count": 4,
        "seed": 17,
        "pointseg_aux_loss_weight": 0.0005,
        "drop_n_last_frames": 1,
    }
    first = manifest_builder.build_manifest("dataset", **kwargs)
    second = manifest_builder.build_manifest("dataset", **kwargs)
    assert first == second
    assert len(first["indices"]) == len(set(first["indices"])) == 4
    assert first["loss_contract"]["pointseg_aux_loss_weight"] == 0.0005
    assert first["selection"]["valid_frame_count"] == 7
