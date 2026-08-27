import pytest
import torch

from lerobot.scripts.diagnose_song_gradient_conflict import (
    cosine_similarity,
    decompose_objective_gradients,
    parse_layer_specs,
    summarize_gradient_moments,
)


def test_diagnose_validates_before_reading_pretrained_path() -> None:
    """Keep --policy.path resolution ahead of the pretrained-path guard."""

    import inspect

    from lerobot.scripts import diagnose_song_gradient_conflict as module

    source = inspect.getsource(module.diagnose)
    assert source.index("cfg.validate()") < source.index(
        "if cfg.policy is None or cfg.policy.pretrained_path"
    )


def test_parse_layer_specs() -> None:
    assert parse_layer_specs("early=model.a.*, head=model.b.weight") == {
        "early": "model.a.*",
        "head": "model.b.weight",
    }
    with pytest.raises(ValueError, match="label=parameter_glob"):
        parse_layer_specs("model.a.*")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_layer_specs("same=model.a.*,same=model.b.*")


def test_gradient_moments_noise_to_signal() -> None:
    # Equally weighted samples [1, 0] and [3, 0]: mean norm=2, RMS noise=1.
    summary = summarize_gradient_moments(
        torch.tensor([4.0, 0.0]),
        weighted_squared_norm=10.0,
        total_weight=2.0,
    )
    assert summary["mean_gradient_norm"] == pytest.approx(2.0)
    assert summary["microbatch_noise_rms"] == pytest.approx(1.0)
    assert summary["noise_to_signal"] == pytest.approx(0.5)


def test_cosine_similarity_handles_conflict_and_zero() -> None:
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])) == pytest.approx(2**-0.5)
    assert cosine_similarity(torch.zeros(2), torch.ones(2)) is None


def test_nested_objective_gradient_decomposition() -> None:
    action = torch.tensor([1.0, 2.0])
    point = torch.tensor([-0.5, 3.0])
    world = torch.tensor([4.0, -1.0])
    actual = decompose_objective_gradients(action, action + point, action + point + world)
    for result, expected in zip(actual, (action, point, world), strict=True):
        torch.testing.assert_close(result, expected)
