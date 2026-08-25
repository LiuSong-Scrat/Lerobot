from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla import molmo2_with_expert as molmo_core
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def _tiny_spec(hidden_size: int, intermediate_size: int, *, num_layers: int):
    return molmo_core.Molmo2TextSpec(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=31,
        additional_vocab_size=1,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        max_position_embeddings=1024,
        qkv_bias=False,
        use_qk_norm=True,
        qk_norm_type="qwen3",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=0.02,
    )


def _make_backend(*, gradient_checkpointing: bool):
    num_layers = 4
    prefix_spec = _tiny_spec(8, 16, num_layers=num_layers)
    expert_spec = _tiny_spec(6, 12, num_layers=num_layers)
    backend = molmo_core.Molmo2WithExpertModel.__new__(molmo_core.Molmo2WithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = molmo_core.Molmo2TextBackbone(
        prefix_spec,
        num_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.lm_expert = molmo_core.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        num_layers,
        self_attn_every_n_layers=2,
        share_cross_attention_kv_projection=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    molmo_core._initialize_module(backend, 0.02)
    backend.num_vlm_layers = num_layers
    backend.num_expert_layers = num_layers
    backend.self_attn_every_n_layers = 2
    backend.attention_mode = "cross_attn"
    backend.train_expert_only = True
    backend.text_spec = prefix_spec
    backend.expert_spec = expert_spec
    backend.inference_only_vlm = False
    backend.gradient_checkpointing = gradient_checkpointing
    backend.gradient_checkpointing_layers_per_segment = 2
    backend.set_requires_grad()
    backend.train()
    return backend


def _joint_layout():
    # The two samples intentionally have different native lengths and therefore
    # different physical Ego-scene slots inside the same padded prefix.
    prefix_valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True, True],
        ]
    )
    prefix_boundaries = torch.tensor(
        [
            [True, True, False, True, True, True, False, False],
            [True, True, False, True, True, True, True, False],
        ]
    )
    scene_indices = torch.tensor([[5, 6], [6, 7]], dtype=torch.long)
    action_valid = torch.ones(2, 4, dtype=torch.bool)

    prefix_attention = make_att_2d_masks(prefix_valid, prefix_boundaries)
    prefix_to_action = torch.zeros(2, 8, 4, dtype=torch.bool)
    action_to_prefix = action_valid[:, :, None] & prefix_valid[:, None, :]
    action_attention = action_valid[:, :, None] & action_valid[:, None, :]
    attention = torch.cat(
        [
            torch.cat([prefix_attention, prefix_to_action], dim=2),
            torch.cat([action_to_prefix, action_attention], dim=2),
        ],
        dim=1,
    )
    prefix_positions = torch.cumsum(prefix_valid, dim=1, dtype=torch.long) - 1
    offsets = prefix_positions.masked_fill(~prefix_valid, -1).amax(dim=1, keepdim=True) + 1
    relative_action_positions = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
    positions = torch.cat([prefix_positions, offsets + relative_action_positions], dim=1)
    return attention, positions, scene_indices


def _joint_layout_with_world_scene():
    # Ego FG/BG occupy per-sample positions after each valid native sequence;
    # World FG/BG share fixed tail slots after the padded native/Ego layout.
    prefix_valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True, True, True, True],
        ]
    )
    prefix_boundaries = torch.tensor(
        [
            [True, True, False, True, True, True, False, False, True, False],
            [True, True, False, True, True, True, True, False, True, False],
        ]
    )
    scene_indices = torch.tensor([[5, 6, 8, 9], [6, 7, 8, 9]], dtype=torch.long)
    action_valid = torch.ones(2, 4, dtype=torch.bool)

    prefix_attention = make_att_2d_masks(prefix_valid, prefix_boundaries)
    prefix_to_action = torch.zeros(2, 10, 4, dtype=torch.bool)
    action_to_prefix = action_valid[:, :, None] & prefix_valid[:, None, :]
    action_attention = action_valid[:, :, None] & action_valid[:, None, :]
    attention = torch.cat(
        [
            torch.cat([prefix_attention, prefix_to_action], dim=2),
            torch.cat([action_to_prefix, action_attention], dim=2),
        ],
        dim=1,
    )
    prefix_positions = torch.cumsum(prefix_valid, dim=1, dtype=torch.long) - 1
    offsets = prefix_positions.masked_fill(~prefix_valid, -1).amax(dim=1, keepdim=True) + 1
    relative_action_positions = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
    positions = torch.cat([prefix_positions, offsets + relative_action_positions], dim=1)
    return attention, positions, scene_indices


def _scatter_scene(native: torch.Tensor, scene: torch.Tensor, indices: torch.Tensor):
    expanded = indices[..., None].expand(-1, -1, native.shape[-1])
    return native.scatter(1, expanded, scene)


def _expert_grads(backend):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in backend.lm_expert.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
@pytest.mark.parametrize("include_world_scene", [False, True])
def test_native_scene_split_matches_legacy_forward_and_trainable_gradients(
    gradient_checkpointing: bool,
    include_world_scene: bool,
) -> None:
    torch.manual_seed(37)
    legacy = _make_backend(gradient_checkpointing=gradient_checkpointing)
    split = copy.deepcopy(legacy)
    attention, positions, scene_indices = (
        _joint_layout_with_world_scene() if include_world_scene else _joint_layout()
    )
    native_values = torch.randn(2, attention.shape[1] - 4, 8)
    scene_values = torch.randn(2, scene_indices.shape[1], 8)
    expert_values = torch.randn(2, 4, 6)

    legacy_native = native_values.clone().requires_grad_()
    legacy_scene = scene_values.clone().requires_grad_()
    legacy_expert = expert_values.clone().requires_grad_()
    legacy_prefix = _scatter_scene(legacy_native, legacy_scene, scene_indices)
    legacy_outputs, _ = legacy(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[legacy_prefix, legacy_expert],
        use_cache=False,
        fill_kv_cache=False,
    )
    legacy_loss = legacy_outputs[1].float().square().mean()
    legacy_loss.backward()

    split_native = native_values.clone().requires_grad_()
    split_scene = scene_values.clone().requires_grad_()
    split_expert = expert_values.clone().requires_grad_()
    split_prefix = _scatter_scene(split_native, split_scene, scene_indices)
    split_outputs, _ = split(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[split_prefix, split_expert],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_indices=scene_indices,
    )
    split_loss = split_outputs[1].float().square().mean()
    split_loss.backward()

    torch.testing.assert_close(split_outputs[0], legacy_outputs[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(split_outputs[1], legacy_outputs[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(split_scene.grad, legacy_scene.grad, atol=1e-8, rtol=1e-5)
    torch.testing.assert_close(split_expert.grad, legacy_expert.grad, atol=1e-8, rtol=1e-5)
    legacy_gradients = _expert_grads(legacy)
    split_gradients = _expert_grads(split)
    assert split_gradients.keys() == legacy_gradients.keys()
    for name in legacy_gradients:
        torch.testing.assert_close(
            split_gradients[name],
            legacy_gradients[name],
            atol=1e-8,
            rtol=1e-5,
        )

    # Native inputs receive an action-loss adjoint in the legacy graph even
    # though their real producers are frozen. The split deliberately removes
    # only this dead-end gradient, while scene/action/Expert gradients match.
    assert legacy_native.grad is not None
    assert split_native.grad is None or torch.count_nonzero(split_native.grad) == 0
    assert all(parameter.grad is None for parameter in split.vlm.parameters())


def test_trainable_vlm_parameter_forces_legacy_fallback(monkeypatch) -> None:
    backend = _make_backend(gradient_checkpointing=False)
    attention, positions, scene_indices = _joint_layout()
    trainable_vlm_parameter = backend.vlm.blocks[0].self_attn.att_proj.weight
    trainable_vlm_parameter.requires_grad_(True)

    def fail_if_split(*args, **kwargs):
        raise AssertionError("native/scene split must not run while Molmo has trainable parameters")

    monkeypatch.setattr(backend, "_forward_native_scene_split", fail_if_split)
    prefix = torch.randn(2, 8, 8, requires_grad=True)
    expert = torch.randn(2, 4, 6, requires_grad=True)
    outputs, _ = backend(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix, expert],
        use_cache=False,
        fill_kv_cache=False,
        prefix_scene_indices=scene_indices,
    )
    outputs[1].float().square().mean().backward()
    assert trainable_vlm_parameter.grad is not None


def test_native_scene_split_adds_no_checkpoint_state() -> None:
    backend = _make_backend(gradient_checkpointing=False)
    state = copy.deepcopy(backend.state_dict())
    parameter_order = tuple(name for name, _ in backend.named_parameters())
    reloaded = _make_backend(gradient_checkpointing=False)
    incompatible = reloaded.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(name for name, _ in reloaded.named_parameters()) == parameter_order
