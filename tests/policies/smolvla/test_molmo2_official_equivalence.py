"""Direct equivalence checks against the vendored Molmo2-ER implementation.

These tests deliberately import the local remote-code model instead of using
another copy of LeRobot's custom runtime as the reference.  They cover the
native path before any trainable FG/BG scene token is inserted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from lerobot.policies.smolvla import molmo2_with_expert as custom_molmo
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.molmo2_full_with_expert import Molmo2FullWithExpertModel
from lerobot.policies.smolvla.molmo2_processing import (
    MOLMO2_NATIVE_OUTPUT_KEYS,
    load_local_molmo2_processor,
    prepare_molmo2_multimodal_batch,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MOLMO2_DIRECTORY = _REPOSITORY_ROOT / "Molmo2-ER"


@pytest.fixture(scope="module")
def official_molmo_modules() -> tuple[object, ModuleType, ModuleType]:
    if not _MOLMO2_DIRECTORY.is_dir():
        pytest.skip(f"Local Molmo2-ER directory is unavailable: {_MOLMO2_DIRECTORY}")

    config = AutoConfig.from_pretrained(
        _MOLMO2_DIRECTORY,
        trust_remote_code=True,
        local_files_only=True,
    )
    generation_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForImageTextToText"],
        str(_MOLMO2_DIRECTORY),
        local_files_only=True,
    )
    modeling_module = sys.modules[generation_class.__module__]
    configuration_module = sys.modules[type(config).__module__]
    return config, modeling_module, configuration_module


def _tiny_text_config(configuration_module: ModuleType):
    return configuration_module.Molmo2TextConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=31,
        additional_vocab_size=1,
        qkv_bias=False,
        num_hidden_layers=1,
        intermediate_size=16,
        hidden_act="silu",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        max_position_embeddings=1024,
        rope_theta=5_000_000.0,
        rope_scaling=None,
        rope_scaling_layers=None,
        use_qk_norm=True,
        qk_norm_type="qwen3",
        layer_norm_eps=1e-6,
        norm_after=False,
        initializer_range=0.02,
        attn_implementation="sdpa",
    )


def _tiny_text_spec(
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_hidden_layers: int = 1,
) -> custom_molmo.Molmo2TextSpec:
    return custom_molmo.Molmo2TextSpec(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
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


def _native_hybrid_mask(valid: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
    sequence_length = valid.shape[1]
    positions = torch.arange(sequence_length, device=valid.device)
    causal = positions[None, :] <= positions[:, None]
    perception = image_tokens[:, :, None] & image_tokens[:, None, :]
    return valid[:, :, None] & valid[:, None, :] & (causal[None] | perception)


def test_local_official_config_stays_inside_custom_runtime_contract(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
) -> None:
    config, _, _ = official_molmo_modules
    text = config.text_config
    assert text.norm_after is False
    assert text.rope_scaling is None
    assert text.rope_scaling_layers is None
    assert text.layer_norm_eps == 1e-6
    assert text.max_position_embeddings == 16384
    assert text.embedding_dropout == 0.0
    assert text.attention_dropout == 0.0
    assert text.residual_dropout == 0.0
    assert text._attn_implementation == "sdpa"

    vision = config.vit_config
    adapter = config.adapter_config
    assert vision.layer_norm_eps == 1e-6
    assert vision.attention_dropout == 0.0
    assert vision.residual_dropout == 0.0
    assert vision._attn_implementation == "sdpa"
    assert adapter.attention_dropout == 0.0
    assert adapter.residual_dropout == 0.0
    assert adapter.image_feature_dropout == 0.0
    assert adapter._attn_implementation == "sdpa"


def test_custom_native_mask_matches_official_token_type_rule(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
) -> None:
    _, official, _ = official_molmo_modules
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, False, False],
        ]
    )
    image_tokens = torch.tensor(
        [
            [False, True, True, True, True, False, False, False, False],
            [False, True, True, True, True, False, False, False, False],
        ]
    )

    # Reproduce the boundary representation used by make_att_2d_masks: every
    # text token starts a causal block, while one contiguous image block starts
    # only at its first token.
    starts_image_block = image_tokens & ~torch.nn.functional.pad(
        image_tokens[:, :-1], (1, 0), value=False
    )
    boundaries = ((~image_tokens) | starts_image_block) & valid
    custom_mask = make_att_2d_masks(valid, boundaries)

    official_extra_rule = official.token_type_ids_mask_function(image_tokens)
    assert official_extra_rule is not None
    official_mask = torch.zeros_like(custom_mask)
    for batch_index in range(valid.shape[0]):
        for query_index in range(valid.shape[1]):
            for key_index in range(valid.shape[1]):
                extra = official_extra_rule(
                    torch.tensor(batch_index),
                    torch.tensor(0),
                    torch.tensor(query_index),
                    torch.tensor(key_index),
                )
                official_mask[batch_index, query_index, key_index] = (
                    valid[batch_index, query_index]
                    and valid[batch_index, key_index]
                    and (key_index <= query_index or bool(extra))
                )

    assert torch.equal(custom_mask, official_mask)
    sequential_positions = torch.cumsum(valid, dim=1, dtype=torch.long) - 1
    for batch_index in range(valid.shape[0]):
        count = int(valid[batch_index].sum())
        assert torch.equal(sequential_positions[batch_index, :count], torch.arange(count))


def test_official_and_custom_tiny_decoder_match_every_native_stage_and_cache(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
) -> None:
    _, official, configuration = official_molmo_modules
    torch.manual_seed(101)
    official_config = _tiny_text_config(configuration)
    official_layer = official.Molmo2DecoderLayer(official_config, layer_idx=0).eval()
    custom_layer = custom_molmo.Molmo2DecoderLayer(
        _tiny_text_spec(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    custom_layer.load_state_dict(official_layer.state_dict(), strict=True)

    hidden = torch.randn(2, 9, 8)
    valid = torch.tensor(
        [
            [True] * 9,
            [True, True, True, True, True, True, True, False, False],
        ]
    )
    image_tokens = torch.zeros_like(valid)
    image_tokens[:, 1:5] = True
    mask = _native_hybrid_mask(valid, image_tokens)
    position_ids = torch.cumsum(valid, dim=1, dtype=torch.long) - 1

    official_normalized = official_layer.attn_norm(hidden)
    custom_normalized = custom_layer.attn_norm(hidden)
    assert torch.equal(official_normalized, custom_normalized)

    official_qkv = official_layer.self_attn.att_proj(official_normalized)
    official_query, official_key, official_value = official_qkv.split(
        official_layer.self_attn.fused_dims, dim=-1
    )
    official_query = official_query.view(2, 9, 2, 4)
    official_key = official_key.view(2, 9, 1, 4)
    official_value = official_value.view(2, 9, 1, 4)
    official_query = official_layer.self_attn.q_norm(official_query).transpose(1, 2)
    official_key = official_layer.self_attn.k_norm(official_key).transpose(1, 2)
    official_value_heads = official_value.transpose(1, 2)
    rotary = official.Molmo2RotaryEmbedding(official_config)
    official_rope = rotary(hidden, position_ids)
    official_query, official_key = official.apply_rotary_pos_emb(
        official_query,
        official_key,
        *official_rope,
    )

    custom_query, custom_key, custom_value = custom_layer.self_attn.project(
        custom_normalized,
        position_ids,
    )
    assert torch.equal(custom_query, official_query.transpose(1, 2))
    assert torch.equal(custom_key, official_key.transpose(1, 2))
    assert torch.equal(custom_value, official_value)

    cache = DynamicCache(config=official_config)
    official_projected_attention, _ = official_layer.self_attn(
        hidden_states=official_normalized,
        position_embeddings=official_rope,
        attention_mask=mask[:, None],
        past_key_values=cache,
        cache_position=torch.arange(hidden.shape[1]),
    )
    custom_attention = custom_molmo._scaled_dot_product_attention(
        custom_query,
        custom_key,
        custom_value,
        mask,
    )
    custom_projected_attention = custom_layer.self_attn.attn_out(custom_attention)
    assert torch.equal(custom_projected_attention, official_projected_attention)

    official_cached_key, official_cached_value = cache[0]
    assert torch.equal(custom_key, official_cached_key.transpose(1, 2))
    assert torch.equal(custom_value, official_cached_value.transpose(1, 2))
    assert torch.equal(official_value_heads, official_cached_value)

    official_after_attention = hidden + official_layer.dropout(official_projected_attention)
    official_output = official_after_attention + official_layer.dropout(
        official_layer.mlp(official_layer.ff_norm(official_after_attention))
    )
    custom_output = custom_layer.finish_attention(hidden, custom_attention)
    assert torch.equal(custom_output, official_output)

    official_input = hidden.detach().clone().requires_grad_()
    custom_input = hidden.detach().clone().requires_grad_()
    official_position_embeddings = rotary(official_input, position_ids)
    official_forward = official_layer(
        official_input,
        position_embeddings=official_position_embeddings,
        attention_mask=mask[:, None],
        position_ids=position_ids,
    )[0]
    custom_normalized = custom_layer.attn_norm(custom_input)
    custom_query, custom_key, custom_value = custom_layer.self_attn.project(
        custom_normalized,
        position_ids,
    )
    custom_forward = custom_layer.finish_attention(
        custom_input,
        custom_molmo._scaled_dot_product_attention(
            custom_query,
            custom_key,
            custom_value,
            mask,
        ),
    )
    official_gradient = torch.autograd.grad(official_forward.square().mean(), official_input)[0]
    custom_gradient = torch.autograd.grad(custom_forward.square().mean(), custom_input)[0]
    torch.testing.assert_close(custom_forward, official_forward, rtol=0.0, atol=0.0)
    torch.testing.assert_close(custom_gradient, official_gradient, rtol=0.0, atol=0.0)


def test_masked_action_suffix_does_not_change_official_native_prefix_path(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
) -> None:
    """WEP joint attention may read the VLM, but the VLM must not read actions."""

    _, official, configuration = official_molmo_modules
    torch.manual_seed(173)
    official_config = _tiny_text_config(configuration)
    official_layers = nn.ModuleList(
        [official.Molmo2DecoderLayer(official_config, layer_idx=index) for index in range(2)]
    ).eval()
    official_final_norm = official.Molmo2RMSNorm(8, eps=1e-6).eval()

    prefix_spec = _tiny_text_spec(num_hidden_layers=2)
    expert_spec = _tiny_text_spec(
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=2,
    )
    backend = custom_molmo.Molmo2WithExpertModel.__new__(custom_molmo.Molmo2WithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = custom_molmo.Molmo2TextBackbone(
        prefix_spec,
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.lm_expert = custom_molmo.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        2,
        self_attn_every_n_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    custom_molmo._initialize_module(backend, 0.02)
    for custom_layer, official_layer in zip(
        backend.vlm.blocks, official_layers, strict=True
    ):
        custom_layer.load_state_dict(official_layer.state_dict(), strict=True)
    backend.vlm.ln_f.load_state_dict(official_final_norm.state_dict(), strict=True)
    backend.self_attn_every_n_layers = 2
    backend.inference_only_vlm = False
    backend.gradient_checkpointing = False
    backend.eval()

    prefix = torch.randn(2, 9, 8)
    actions = torch.randn(2, 4, 6)
    valid = torch.tensor(
        [
            [True] * 9,
            [True, True, True, True, True, True, True, False, False],
        ]
    )
    image_tokens = torch.zeros_like(valid)
    image_tokens[:, 1:5] = True
    prefix_mask = _native_hybrid_mask(valid, image_tokens)
    prefix_positions = torch.cumsum(valid, dim=1, dtype=torch.long) - 1

    action_valid = torch.ones(2, 4, dtype=torch.bool)
    prefix_to_action = torch.zeros(2, 9, 4, dtype=torch.bool)
    action_to_prefix = action_valid[:, :, None] & valid[:, None, :]
    action_to_action = action_valid[:, :, None] & action_valid[:, None, :]
    joint_mask = torch.cat(
        [
            torch.cat([prefix_mask, prefix_to_action], dim=2),
            torch.cat([action_to_prefix, action_to_action], dim=2),
        ],
        dim=1,
    )
    prefix_offset = prefix_positions.masked_fill(~valid, -1).amax(dim=1, keepdim=True) + 1
    action_positions = prefix_offset + torch.arange(4)[None]
    joint_positions = torch.cat([prefix_positions, action_positions], dim=1)

    with torch.no_grad():
        official_hidden = prefix
        rotary = official.Molmo2RotaryEmbedding(official_config)
        for official_layer in official_layers:
            official_hidden = official_layer(
                official_hidden,
                position_embeddings=rotary(official_hidden, prefix_positions),
                attention_mask=prefix_mask[:, None],
                position_ids=prefix_positions,
            )[0]
        official_hidden = official_final_norm(official_hidden)

        (custom_prefix_only, _), _ = backend(
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            inputs_embeds=[prefix, None],
            use_cache=False,
            fill_kv_cache=False,
        )
        (custom_joint_prefix, _), _ = backend(
            attention_mask=joint_mask,
            position_ids=joint_positions,
            inputs_embeds=[prefix, actions],
            use_cache=False,
            fill_kv_cache=False,
        )

    assert torch.equal(custom_prefix_only, official_hidden)
    # The even joint layer changes the SDPA tensor shape, so floating-point
    # reduction can move by a few ULPs. It must not create a semantic action
    # dependency in the prefix stream.
    torch.testing.assert_close(custom_joint_prefix, official_hidden, rtol=1e-6, atol=2e-7)


def test_official_and_split_custom_wte_are_bit_exact(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
) -> None:
    _, official, _ = official_molmo_modules
    torch.manual_seed(211)
    official_embedding = official.Molmo2Embedding(7, 3, 5)
    custom_embedding = custom_molmo.Molmo2Embedding(
        7,
        3,
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    custom_embedding.load_state_dict(official_embedding.state_dict(), strict=True)
    token_ids = torch.tensor([[0, 6, 7, 9], [8, 2, 7, 1]])
    assert torch.equal(custom_embedding(token_ids), official_embedding(token_ids))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_custom_rope_is_bit_exact_to_official_at_real_head_width_and_long_positions(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
    dtype: torch.dtype,
) -> None:
    config, official, _ = official_molmo_modules
    torch.manual_seed(263)
    position_ids = torch.tensor([[0, 1, 410, 414, 10_000, 16_383]])
    hidden = torch.randn(1, position_ids.shape[1], 2, 128, dtype=dtype)
    cos, sin = official.Molmo2RotaryEmbedding(config.text_config)(hidden, position_ids)
    official_query, _ = official.apply_rotary_pos_emb(
        hidden.transpose(1, 2),
        hidden.transpose(1, 2),
        cos,
        sin,
    )
    custom_query = custom_molmo.apply_molmo2_rope(
        hidden,
        position_ids,
        config.text_config.rope_theta,
    )
    assert torch.equal(custom_query, official_query.transpose(1, 2))


def test_actual_processor_fast_path_is_bit_exact_to_official_slow_path() -> None:
    if not _MOLMO2_DIRECTORY.is_dir():
        pytest.skip(f"Local Molmo2-ER directory is unavailable: {_MOLMO2_DIRECTORY}")
    processor = load_local_molmo2_processor(_MOLMO2_DIRECTORY)
    torch.manual_seed(307)
    images = torch.rand(2, 3, 256, 256, dtype=torch.float32)
    tasks = [
        "pick up the black bowl",
        "place the red mug on the left plate and then stop",
    ]
    fast = prepare_molmo2_multimodal_batch(
        processor,
        tasks,
        images,
        max_text_length=48,
        use_fast_image_path=True,
    )
    slow = prepare_molmo2_multimodal_batch(
        processor,
        tasks,
        images,
        max_text_length=48,
        use_fast_image_path=False,
    )
    assert tuple(fast) == MOLMO2_NATIVE_OUTPUT_KEYS
    assert tuple(slow) == MOLMO2_NATIVE_OUTPUT_KEYS
    for key in MOLMO2_NATIVE_OUTPUT_KEYS:
        assert torch.equal(fast[key], slow[key]), key


def _tiny_official_vision(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
    *,
    dtype: torch.dtype,
) -> nn.Module:
    _, official, configuration = official_molmo_modules
    vit_config = configuration.Molmo2VitConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_num_pos=4,
        attention_dropout=0.0,
        residual_dropout=0.0,
        float32_attention=True,
        attn_implementation="sdpa",
    )
    adapter_config = configuration.Molmo2AdapterConfig(
        vit_layers=(-1, -2),
        pooling_attention_mask=True,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        float32_attention=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        hidden_act="silu",
        intermediate_size=16,
        text_hidden_size=8,
        image_feature_dropout=0.0,
        attn_implementation="sdpa",
    )
    return official.Molmo2VisionBackbone(vit_config, adapter_config).to(dtype=dtype).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_official_vision_single_encode_reuse_matches_native_two_crop_path(
    official_molmo_modules: tuple[object, ModuleType, ModuleType],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(401)
    vision = _tiny_official_vision(official_molmo_modules, dtype=dtype)
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    nn.Module.__init__(backend)
    backend.vision_backbone = vision
    backend.exact_vision_reuse = True
    backend.vision_reuse_calls = 0
    backend.vision_fallback_calls = 0
    backend._current_visual_batch_uses_registered_fast_contract = False

    images = torch.randn(2, 2, 4, 14 * 14 * 3, dtype=dtype)
    images[:, 1] = images[:, 0]
    pooled_patches_idx = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        ]
    )
    with torch.no_grad():
        official_output = vision(images, pooled_patches_idx)
        reused_output = backend._encode_native_vision(images, pooled_patches_idx)

    if dtype == torch.float32:
        assert torch.equal(reused_output, official_output)
    else:
        # Changing the GEMM batch dimension from B*2 to B can change BF16
        # accumulation at the last bits even though both crops are identical.
        torch.testing.assert_close(reused_output, official_output, rtol=2e-2, atol=2e-3)
    assert backend.last_vision_encode_report["path"] == "exact_single_encode_reuse"
