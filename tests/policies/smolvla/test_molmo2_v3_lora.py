#!/usr/bin/env python

"""CPU-only contract tests for the Molmo2-ER V3 attention LoRA path."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    classify_v3_lora_checkpoint_keys,
    is_full_molmo2er_lora_policy_parameter_name,
)
from lerobot.policies.smolvla.molmo2_full_with_expert import Molmo2FullWithExpertModel
from lerobot.policies.smolvla.molmo2_with_expert import (
    MOLMO_V3_LORA_TARGET_MODULES,
    Molmo2ExpertBackbone,
    Molmo2TextBackbone,
    Molmo2TextSpec,
    MolmoV3LoRALinear,
    _initialize_module,
    expected_molmo_v3_lora_parameters,
    inject_molmo_v3_attention_lora,
    is_molmo_v3_lora_parameter_name,
)


def _tiny_spec(hidden_size: int, intermediate_size: int) -> Molmo2TextSpec:
    return Molmo2TextSpec(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=36,
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


def _make_tiny_backend(*, num_layers: int = 3) -> Molmo2FullWithExpertModel:
    prefix_spec = _tiny_spec(8, 16)
    expert_spec = _tiny_spec(6, 12)
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = Molmo2TextBackbone(
        prefix_spec,
        num_layers,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.lm_expert = Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        num_layers,
        self_attn_every_n_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.vision_backbone = nn.Linear(1, 1, bias=False)
    _initialize_module(backend, 0.02)
    backend.num_vlm_layers = num_layers
    backend.num_expert_layers = num_layers
    backend.self_attn_every_n_layers = 2
    backend.attention_mode = "cross_attn"
    backend.train_expert_only = True
    backend.freeze_vision_encoder = True
    backend.text_spec = prefix_spec
    backend.expert_spec = expert_spec
    backend.inference_only_vlm = False
    backend.gradient_checkpointing = False
    backend.gradient_checkpointing_layers_per_segment = 2
    backend.molmo_lora_enable = False
    backend.molmo_lora_rank = 0
    backend.molmo_lora_alpha = 0.0
    backend.molmo_lora_dropout = 0.0
    backend.molmo_lora_target_modules = MOLMO_V3_LORA_TARGET_MODULES
    backend.molmo_lora_module_names = ()
    backend.molmo_lora_parameter_count = 0
    backend.set_requires_grad()
    backend.train()
    return backend


def _inject_lora(
    backend: Molmo2FullWithExpertModel,
    *,
    rank: int = 2,
    alpha: float = 2.0,
    dropout: float = 0.0,
) -> tuple[str, ...]:
    names = inject_molmo_v3_attention_lora(
        backend.vlm,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    backend.molmo_lora_enable = True
    backend.molmo_lora_rank = rank
    backend.molmo_lora_alpha = alpha
    backend.molmo_lora_dropout = dropout
    backend.molmo_lora_module_names = names
    backend.molmo_lora_parameter_count = sum(
        parameter.numel()
        for name, parameter in backend.vlm.named_parameters()
        if is_molmo_v3_lora_parameter_name(name)
    )
    backend.set_requires_grad()
    backend.train()
    return names


def _joint_attention_and_positions(
    prefix_valid: torch.Tensor,
    action_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_attention = prefix_valid[:, :, None] & prefix_valid[:, None, :]
    prefix_to_action = torch.zeros(
        prefix_valid.shape[0],
        prefix_valid.shape[1],
        action_valid.shape[1],
        dtype=torch.bool,
    )
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
    offset = prefix_positions.masked_fill(~prefix_valid, -1).amax(dim=1, keepdim=True) + 1
    action_steps = action_valid.shape[1] // 2
    relative = torch.arange(action_steps)[None].repeat(prefix_valid.shape[0], 2)
    return attention, torch.cat([prefix_positions, offset + relative], dim=1)


def _forward_action(
    backend: Molmo2FullWithExpertModel,
    prefix: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    attention, positions = _joint_attention_and_positions(
        torch.ones(prefix.shape[:2], dtype=torch.bool),
        torch.ones(action.shape[:2], dtype=torch.bool),
    )
    (_, action_output), cache = backend(
        attention_mask=attention,
        position_ids=positions,
        inputs_embeds=[prefix, action],
        use_cache=False,
        fill_kv_cache=False,
    )
    assert cache is None
    assert action_output is not None
    return action_output


def test_injection_targets_only_every_vlm_attention_projection() -> None:
    backend = _make_tiny_backend(num_layers=3)
    original_base_ids = {
        f"{layer_index}.{target_name}": id(
            getattr(block.self_attn, target_name).weight
        )
        for layer_index, block in enumerate(backend.vlm.blocks)
        for target_name in MOLMO_V3_LORA_TARGET_MODULES
    }

    names = _inject_lora(backend, rank=2)

    assert names == (
        "blocks.0.self_attn.att_proj",
        "blocks.0.self_attn.attn_out",
        "blocks.1.self_attn.att_proj",
        "blocks.1.self_attn.attn_out",
        "blocks.2.self_attn.att_proj",
    )
    for layer_index, block in enumerate(backend.vlm.blocks):
        for target_name in MOLMO_V3_LORA_TARGET_MODULES:
            module = getattr(block.self_attn, target_name)
            should_be_adapted = target_name == "att_proj" or layer_index < 2
            assert isinstance(module, MolmoV3LoRALinear) is should_be_adapted
            assert id(module.weight) == original_base_ids[f"{layer_index}.{target_name}"]
            if should_be_adapted:
                assert module.lora_A.dtype == module.lora_B.dtype == torch.float32
                assert torch.count_nonzero(module.lora_B) == 0
        assert isinstance(block.mlp.ff_proj, nn.Linear)
        assert not isinstance(block.mlp.ff_proj, MolmoV3LoRALinear)
        assert isinstance(block.mlp.ff_out, nn.Linear)
        assert not isinstance(block.mlp.ff_out, MolmoV3LoRALinear)

    with pytest.raises(RuntimeError, match="already injected"):
        inject_molmo_v3_attention_lora(backend.vlm, rank=2, alpha=2.0, dropout=0.0)

    bf16_base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    bf16_adapter = MolmoV3LoRALinear.from_linear(
        bf16_base,
        rank=2,
        alpha=2.0,
        dropout=0.0,
    )
    assert bf16_adapter.weight.dtype == torch.bfloat16
    assert bf16_adapter.lora_A.dtype == bf16_adapter.lora_B.dtype == torch.float32
    bf16_input = torch.randn(2, 8, dtype=torch.bfloat16)
    assert torch.equal(bf16_adapter(bf16_input), bf16_base(bf16_input))


def test_zero_initialized_lora_is_function_preserving_for_full_v3_forward() -> None:
    torch.manual_seed(17)
    base = _make_tiny_backend(num_layers=3)
    lora = copy.deepcopy(base)
    _inject_lora(lora, rank=3, alpha=6.0)
    base.eval()
    lora.eval()
    prefix = torch.randn(2, 5, 8)
    action = torch.randn(2, 4, 6)

    with torch.no_grad():
        base_output = _forward_action(base, prefix, action)
        lora_output = _forward_action(lora, prefix, action)

    assert torch.equal(base_output, lora_output)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_only_lora_and_existing_v3_trainables_receive_parameter_gradients(
    gradient_checkpointing: bool,
) -> None:
    torch.manual_seed(29)
    backend = _make_tiny_backend(num_layers=3)
    _inject_lora(backend, rank=2)
    backend.gradient_checkpointing = gradient_checkpointing
    prefix = torch.randn(2, 5, 8, requires_grad=True)
    action = torch.randn(2, 4, 6, requires_grad=True)

    _forward_action(backend, prefix, action).square().mean().backward()

    lora_a = {}
    lora_b = {}
    frozen_base = []
    for name, parameter in backend.vlm.named_parameters():
        if name.endswith(".lora_A"):
            lora_a[name] = parameter
        elif name.endswith(".lora_B"):
            lora_b[name] = parameter
        else:
            frozen_base.append(parameter)
    assert lora_a and lora_b and frozen_base
    assert all(parameter.requires_grad for parameter in (*lora_a.values(), *lora_b.values()))

    # A zero-initialized B makes dL/dA exactly zero on the first update, while
    # every injected B must already receive a signal.  The unreachable final
    # VLM attn_out is intentionally not injected, which is required for DDP
    # with find_unused_parameters=False.
    assert all(parameter.grad is not None for parameter in (*lora_a.values(), *lora_b.values()))
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in lora_a.values())
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in lora_b.values())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in frozen_base)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in backend.vision_backbone.parameters()
    )
    assert all(parameter.requires_grad for parameter in backend.lm_expert.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in backend.lm_expert.parameters()
    )
    assert prefix.grad is not None and torch.count_nonzero(prefix.grad) > 0
    assert action.grad is not None and torch.count_nonzero(action.grad) > 0


def test_lora_parameter_count_matches_tiny_and_production_contracts() -> None:
    rank = 3
    layers = 4
    backend = _make_tiny_backend(num_layers=layers)
    _inject_lora(backend, rank=rank)
    lora_parameters = [
        parameter
        for name, parameter in backend.vlm.named_parameters()
        if is_molmo_v3_lora_parameter_name(name)
    ]

    # Tiny H=8: fused QKV is 8->16 in all L layers, while the 8->8
    # attention output is adapted only in the first L-1 layers.
    tiny_expected = rank * (24 * layers + 16 * (layers - 1))
    assert len(lora_parameters) == 2 * (2 * layers - 1)
    assert sum(parameter.numel() for parameter in lora_parameters) == tiny_expected
    assert backend.molmo_lora_parameter_count == tiny_expected
    assert expected_molmo_v3_lora_parameters(rank=1, num_layers=36) == 546_304
    assert expected_molmo_v3_lora_parameters(rank=8, num_layers=36) == 4_370_432


def test_lora_state_dict_strict_roundtrip_preserves_exact_output() -> None:
    torch.manual_seed(37)
    source = _make_tiny_backend(num_layers=3)
    _inject_lora(source, rank=2)
    target = _make_tiny_backend(num_layers=3)
    _inject_lora(target, rank=2)
    with torch.no_grad():
        for name, parameter in source.vlm.named_parameters():
            if name.endswith(".lora_B"):
                parameter.normal_(mean=0.0, std=0.03)

    incompatible = target.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    source.eval()
    target.eval()
    prefix = torch.randn(1, 5, 8)
    action = torch.randn(1, 4, 6)
    with torch.no_grad():
        source_output = _forward_action(source, prefix, action)
        target_output = _forward_action(target, prefix, action)
    assert torch.equal(source_output, target_output)


def test_lora_parameter_name_matchers_accept_only_their_intended_scope() -> None:
    relative_names = (
        "blocks.0.self_attn.att_proj.lora_A",
        "blocks.35.self_attn.attn_out.lora_B",
    )
    assert all(is_molmo_v3_lora_parameter_name(name) for name in relative_names)
    assert not is_molmo_v3_lora_parameter_name("blocks.0.self_attn.att_proj.weight")
    assert not is_molmo_v3_lora_parameter_name("blocks.0.self_attn.att_proj.lora_A_extra")

    assert is_full_molmo2er_lora_policy_parameter_name(
        "model.vlm_with_expert.vlm.blocks.0.self_attn.att_proj.lora_A"
    )
    assert is_full_molmo2er_lora_policy_parameter_name(
        "model.vlm_with_expert.vlm.blocks.34.self_attn.attn_out.lora_B"
    )
    invalid_policy_names = (
        "model.vlm_with_expert.vlm.blocks.36.self_attn.att_proj.lora_A",
        "model.vlm_with_expert.vlm.blocks.35.self_attn.attn_out.lora_B",
        "model.vlm_with_expert.vlm.blocks.0.mlp.ff_proj.lora_A",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.att_proj.lora_A",
        "model.vlm_with_expert.vlm.blocks.0.self_attn.att_proj.weight",
        "vlm.blocks.0.self_attn.att_proj.lora_A",
    )
    assert not any(
        is_full_molmo2er_lora_policy_parameter_name(name) for name in invalid_policy_names
    )


def test_checkpoint_key_classifier_restores_strict_load_semantics() -> None:
    expected = {
        "model.vlm_with_expert.vlm.blocks.0.self_attn.att_proj.lora_A",
        "model.vlm_with_expert.vlm.blocks.0.self_attn.att_proj.lora_B",
    }
    assert classify_v3_lora_checkpoint_keys(
        expected_lora_keys=expected,
        missing_keys=set(),
        unexpected_keys=set(),
    ) == "strict_v3_lora"
    assert classify_v3_lora_checkpoint_keys(
        expected_lora_keys=expected,
        missing_keys=set(expected),
        unexpected_keys=set(),
    ) == "base_v3_zero_lora_upgrade"

    with pytest.raises(RuntimeError, match="checkpoint contract failed"):
        classify_v3_lora_checkpoint_keys(
            expected_lora_keys=expected,
            missing_keys={next(iter(expected))},
            unexpected_keys=set(),
        )
    with pytest.raises(RuntimeError, match="checkpoint contract failed"):
        classify_v3_lora_checkpoint_keys(
            expected_lora_keys=expected,
            missing_keys=set(),
            unexpected_keys={"model.unexpected.weight"},
        )
    with pytest.raises(RuntimeError, match="no registered adapter"):
        classify_v3_lora_checkpoint_keys(
            expected_lora_keys=set(),
            missing_keys=set(),
            unexpected_keys=set(),
        )


def test_policy_optimizer_keeps_v3_trainables_and_uses_lower_lora_lr() -> None:
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=False,
        vlm_backend="molmo2_full",
        molmo_lora_enable=True,
        molmo_lora_lr_multiplier=0.1,
        optimizer_lr=1e-4,
    )
    policy.model = nn.Module()
    policy.model.vlm_with_expert = nn.Module()
    policy.model.vlm_with_expert.vlm = nn.Module()
    policy.model.vlm_with_expert.vlm.blocks = nn.ModuleList([nn.Module()])
    attention = nn.Module()
    attention.att_proj = MolmoV3LoRALinear.from_linear(
        nn.Linear(4, 6, bias=False), rank=2, alpha=2.0, dropout=0.0
    )
    attention.att_proj.weight.requires_grad_(False)
    policy.model.vlm_with_expert.vlm.blocks[0].self_attn = attention
    policy.model.vlm_with_expert.lm_expert = nn.Linear(4, 4)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "v3_original_trainable",
        "molmo_v3_lora",
    ]
    assert [group["lr"] for group in groups] == [1e-4, 1e-5]
    lora_ids = {
        id(attention.att_proj.lora_A),
        id(attention.att_proj.lora_B),
    }
    assert {id(parameter) for parameter in groups[1]["params"]} == lora_ids
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in policy.model.vlm_with_expert.lm_expert.parameters()
    }
