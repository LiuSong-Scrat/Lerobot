from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmarks.song_real_libero.scripts.smolvla_model_inference import SmolVLA_ModelInference


class _NoiseModel:
    def __init__(self, *, se3_enable: bool, pose9_enable: bool) -> None:
        self.config = SimpleNamespace(
            chunk_size=3,
            max_action_dim=10,
            se3_enable=se3_enable,
            pose9_action_noise_enable=pose9_enable,
        )
        self.calls: list[str] = []

    def sample_noise(self, shape, device):
        self.calls.append("legacy")
        return torch.full(shape, 1.0, device=device)

    def sample_pose9_action_noise(self, shape, device):
        self.calls.append("pose9")
        return torch.full(shape, 2.0, device=device)

    def sample_se3_action_noise(self, actions):
        self.calls.append("se3")
        return None, None, torch.full_like(actions, 3.0)


def _inference_stub(model: _NoiseModel) -> SmolVLA_ModelInference:
    inference = object.__new__(SmolVLA_ModelInference)
    inference.policy = SimpleNamespace(
        model=model,
        prepare_state=lambda _batch: torch.zeros(1, 10),
    )
    return inference


def test_seeded_noise_uses_pose9_sampler_when_pose9_prior_is_enabled():
    model = _NoiseModel(se3_enable=False, pose9_enable=True)
    noise = _inference_stub(model)._make_seeded_action_noise({}, 7)

    assert model.calls == ["pose9"]
    assert torch.equal(noise, torch.full((1, 3, 10), 2.0))


def test_seeded_noise_keeps_se3_and_legacy_dispatch():
    se3_model = _NoiseModel(se3_enable=True, pose9_enable=True)
    se3_noise = _inference_stub(se3_model)._make_seeded_action_noise({}, 7)
    assert se3_model.calls == ["se3"]
    assert torch.equal(se3_noise, torch.full((1, 3, 10), 3.0))

    legacy_model = _NoiseModel(se3_enable=False, pose9_enable=False)
    legacy_noise = _inference_stub(legacy_model)._make_seeded_action_noise({}, 7)
    assert legacy_model.calls == ["legacy"]
    assert torch.equal(legacy_noise, torch.full((1, 3, 10), 1.0))


def test_zero_noise_is_valid_identity_pose9_for_pose_checkpoints():
    model = _NoiseModel(se3_enable=False, pose9_enable=True)
    noise = _inference_stub(model)._make_zero_action_noise({})

    expected = torch.zeros(1, 3, 10)
    expected[..., 3] = 1.0
    expected[..., 7] = 1.0
    assert torch.equal(noise, expected)
