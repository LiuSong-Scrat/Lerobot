import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.umi_processor import UMIProcessor


def _pose10(x: float, gripper: float = 0.08) -> list[float]:
    return [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, gripper]


def test_umi_processor_uses_observed_state_as_origin_for_absolute_targets():
    state = torch.tensor([[_pose10(0.1)]], dtype=torch.float32)
    action = torch.tensor(
        [[_pose10(0.3), _pose10(0.4)]],
        dtype=torch.float32,
    )

    output = UMIProcessor()(
        {
            TransitionKey.OBSERVATION: {"observation.state": state},
            TransitionKey.ACTION: action,
        }
    )

    output_state = output[TransitionKey.OBSERVATION]["observation.state"]
    output_action = output[TransitionKey.ACTION]
    assert torch.allclose(output_state[0, 0, :3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        output_action[0, :, :3],
        torch.tensor([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(output_action[..., 9], action[..., 9])


def test_umi_processor_supports_observation_only_inference():
    state = torch.tensor([_pose10(0.1)], dtype=torch.float32)

    output = UMIProcessor()(
        {TransitionKey.OBSERVATION: {"observation.state": state}}
    )

    output_state = output[TransitionKey.OBSERVATION]["observation.state"]
    assert output_state.shape == state.shape
    assert torch.allclose(output_state[0, :3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        output_state[0, 3:9],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        atol=1e-6,
    )
