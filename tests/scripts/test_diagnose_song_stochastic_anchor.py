import pytest
import torch

from lerobot.scripts.diagnose_song_stochastic_anchor import (
    derive_forward_seeds,
    native_objective_terms,
    summarize_scalar_samples,
)


def test_derive_forward_seeds_is_reproducible_and_distinct() -> None:
    first = derive_forward_seeds(20260828, 8)
    second = derive_forward_seeds(20260828, 8)
    assert first == second
    assert len(first) == len(set(first)) == 8
    assert derive_forward_seeds(20260829, 8) != first


def test_summarize_scalar_samples_uses_population_std_and_quantiles() -> None:
    summary = summarize_scalar_samples([1.0, 2.0, 3.0, 4.0])
    assert summary == pytest.approx(
        {
            "mean": 2.5,
            "std": 5**0.5 / 2,
            "median": 2.5,
            "p10": 1.3,
            "p90": 3.7,
            "min": 1.0,
            "max": 4.0,
        }
    )
    assert summarize_scalar_samples([7.0])["std"] == 0.0


class _FakeWorldModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_pointseg_aux_loss = None
        self.last_worldflow_aux = None

    def compute_worldflow_aux_loss(self):
        return self.last_worldflow_aux


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeWorldModel()

    def forward(self, _batch):
        self.model.last_pointseg_aux_loss = torch.tensor(4.0)
        self.model.last_worldflow_aux = {"per_sample_loss": torch.tensor([0.2, 0.4])}
        return torch.tensor(1.5), {"loss_action": 1.0}


def test_native_objective_terms_uses_weighted_additive_components() -> None:
    terms = native_objective_terms(_FakePolicy(), {}, pointseg_weight=0.05)
    assert terms == pytest.approx(
        {
            "total": 1.5,
            "action": 1.0,
            "weighted_pointseg": 0.2,
            "worldflow": 0.3,
        }
    )
