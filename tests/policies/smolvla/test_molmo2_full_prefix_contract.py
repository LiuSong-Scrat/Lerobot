#!/usr/bin/env python

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.smolvla import molmo2_with_expert as molmo_core
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks
from lerobot.policies.smolvla.molmo2_full_with_expert import Molmo2FullWithExpertModel

TEXT_ROLE = 0
IMAGE_ROLE = 1
SCENE_ROLE = 2
PAD_ROLE = 3
NATIVE_IMAGE_POSITIONS = 410


def _make_prefix_shell(*, hidden_size: int = 8, chunk_size: int = 3) -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.last_prefix_metrics = {}
    model.last_molmo_scene_insert_positions = None
    model.last_molmo_token_roles = None
    model.last_prefix_token_layout = ()
    model.config = SimpleNamespace(
        chunk_size=chunk_size,
        min_period=4e-3,
        max_period=4.0,
    )
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=hidden_size)
    model.action_in_proj = nn.Linear(3, hidden_size)
    model.action_time_mlp_in = nn.Linear(hidden_size * 2, hidden_size)
    model.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)
    return model


def _right_padded_native_batch(
    text_lengths: tuple[int, ...],
    *,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BOS + 410 IMAGE + variable causal TEXT + right padding."""

    batch_size = len(text_lengths)
    native_length = 1 + NATIVE_IMAGE_POSITIONS + max(text_lengths)
    embeddings = torch.zeros(batch_size, native_length, hidden_size)
    valid = torch.zeros(batch_size, native_length, dtype=torch.bool)
    image_tokens = torch.zeros_like(valid)
    for batch_index, text_length in enumerate(text_lengths):
        valid_length = 1 + NATIVE_IMAGE_POSITIONS + text_length
        valid[batch_index, :valid_length] = True
        image_tokens[batch_index, 1 : 1 + NATIVE_IMAGE_POSITIONS] = True
        position_markers = torch.arange(valid_length, dtype=embeddings.dtype)
        embeddings[batch_index, :valid_length, 0] = position_markers + 1000 * batch_index
    return embeddings, valid, image_tokens


def test_full_prefix_inserts_scene_before_variable_text_and_keeps_right_padding() -> None:
    model = _make_prefix_shell(hidden_size=8)
    native, native_valid, native_image = _right_padded_native_batch((2, 5), hidden_size=8)
    scene = torch.zeros(2, 2, 8)
    scene[:, 0, 0] = torch.tensor([-10.0, -11.0])
    scene[:, 1, 0] = torch.tensor([-20.0, -21.0])
    scene_valid = torch.ones(2, 2, dtype=torch.bool)

    prefix, valid, boundaries = model._build_full_molmo_prefix(
        native,
        native_valid,
        native_image,
        scene,
        scene_valid,
        ablate_language=False,
    )

    # BOS(1) + IMAGE(410) determines the insertion point independently of text
    # length.  The shorter sample is repacked and remains strictly right padded.
    assert torch.equal(model.last_molmo_scene_insert_positions, torch.tensor([411, 411]))
    assert prefix.shape == (2, 418, 8)
    assert torch.equal(valid.sum(dim=1), torch.tensor([415, 418]))
    assert valid[0, :415].all() and not valid[0, 415:].any()
    assert valid[1].all()

    # Native IMAGE structure is untouched, FG/BG are inserted after its final
    # position, and each sample's complete text follows the two scene tokens.
    assert torch.equal(prefix[:, :411], native[:, :411])
    assert torch.equal(prefix[:, 411:413], scene)
    assert torch.equal(prefix[0, 413:415], native[0, 411:413])
    assert torch.equal(prefix[1, 413:418], native[1, 411:416])
    assert not prefix[0, 415:].any()

    roles = model.last_molmo_token_roles
    assert roles is not None
    assert torch.equal(roles[:, 0], torch.full((2,), TEXT_ROLE, dtype=torch.int8))
    assert (roles[:, 1:411] == IMAGE_ROLE).all()
    assert (roles[:, 411:413] == SCENE_ROLE).all()
    assert (roles[0, 413:415] == TEXT_ROLE).all()
    assert (roles[0, 415:] == PAD_ROLE).all()
    assert (roles[1, 413:] == TEXT_ROLE).all()

    # BOS starts its own causal block; IMAGE+SCENE share one perception block;
    # every valid text token starts a new causal block; padding starts none.
    assert boundaries[:, 0].all()
    assert boundaries[:, 1].all()
    assert not boundaries[:, 2:413].any()
    assert boundaries[0, 413:415].all() and not boundaries[0, 415:].any()
    assert boundaries[1, 413:].all()


def test_full_prefix_uses_each_samples_actual_last_image_position() -> None:
    model = _make_prefix_shell(hidden_size=4)
    native = torch.randn(2, 414, 4)
    native_valid = torch.ones(2, 414, dtype=torch.bool)
    native_image = torch.zeros_like(native_valid)
    native_image[0, 1:411] = True
    # A deliberately shifted template proves insertion is derived from each
    # token_type_ids row rather than from a hard-coded position.
    native_image[1, 2:412] = True
    scene = torch.randn(2, 2, 4)

    prefix, valid, _ = model._build_full_molmo_prefix(
        native,
        native_valid,
        native_image,
        scene,
        torch.ones(2, 2, dtype=torch.bool),
        ablate_language=False,
    )

    assert torch.equal(model.last_molmo_scene_insert_positions, torch.tensor([411, 412]))
    assert torch.equal(prefix[0, 411:413], scene[0])
    assert torch.equal(prefix[1, 412:414], scene[1])
    assert valid.all()
    roles = model.last_molmo_token_roles
    assert roles is not None
    assert (roles[0, 411:413] == SCENE_ROLE).all()
    assert (roles[1, 412:414] == SCENE_ROLE).all()


def test_full_hybrid_mask_truth_table_with_variable_text_and_action_padding() -> None:
    model = _make_prefix_shell(hidden_size=8, chunk_size=3)
    native, native_valid, native_image = _right_padded_native_batch((2, 5), hidden_size=8)
    prefix, prefix_valid, prefix_boundaries = model._build_full_molmo_prefix(
        native,
        native_valid,
        native_image,
        torch.randn(2, 2, 8),
        torch.ones(2, 2, dtype=torch.bool),
        ablate_language=False,
    )
    _, action_valid, action_boundaries = model.embed_suffix(
        torch.randn(2, 3, 3),
        torch.tensor([0.25, 0.75]),
        actions_is_pad=torch.tensor([[False, False, True], [False, False, False]]),
    )
    full_valid = torch.cat([prefix_valid, action_valid], dim=1)
    full_boundaries = torch.cat([prefix_boundaries, action_boundaries.bool()], dim=1)
    allowed = make_att_2d_masks(full_valid, full_boundaries)

    prefix_length = prefix.shape[1]
    bos = 0
    perception = torch.arange(1, 413)  # 410 IMAGE + FG + BG
    actions = torch.arange(prefix_length, prefix_length + 3)

    for batch_index, text_length in enumerate((2, 5)):
        text = torch.arange(413, 413 + text_length)
        valid_actions = actions[: 2 if batch_index == 0 else 3]

        # BOS is causal and cannot read the later perceptual block.
        assert allowed[batch_index, bos, bos]
        assert not allowed[batch_index, bos, perception].any()
        assert not allowed[batch_index, bos, text].any()
        assert not allowed[batch_index, bos, valid_actions].any()

        # IMAGE and SCENE form one bidirectional perception block.  They can
        # read BOS but never the following instruction or action block.
        assert allowed[batch_index, perception[:, None], perception].all()
        assert allowed[batch_index, perception, bos].all()
        assert not allowed[batch_index, perception[:, None], text].any()
        assert not allowed[batch_index, perception[:, None], valid_actions].any()

        # TEXT reads BOS and all perception, remains strictly causal within
        # TEXT, and cannot read ACTION.
        assert allowed[batch_index, text, bos].all()
        assert allowed[batch_index, text[:, None], perception].all()
        text_to_text = allowed[batch_index, text[:, None], text]
        assert torch.equal(text_to_text, torch.ones(text_length, text_length, dtype=torch.bool).tril())
        assert not allowed[batch_index, text[:, None], valid_actions].any()

        # Every valid ACTION reads the complete valid prefix and the complete
        # valid action chunk; no prefix query can read ACTION.
        valid_prefix = torch.nonzero(prefix_valid[batch_index], as_tuple=False).flatten()
        assert allowed[batch_index, valid_actions[:, None], valid_prefix].all()
        assert allowed[batch_index, valid_actions[:, None], valid_actions].all()
        assert not allowed[batch_index, valid_prefix[:, None], valid_actions].any()

        # Padding is invalid as both a query and key, including padding between
        # the short prefix and the globally aligned action suffix.
        padding = torch.nonzero(~full_valid[batch_index], as_tuple=False).flatten()
        if padding.numel():
            assert not allowed[batch_index, padding].any()
            assert not allowed[batch_index, :, padding].any()


def _tiny_spec(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int = 36,
) -> molmo_core.Molmo2TextSpec:
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


def _make_tiny_frozen_36_layer_backend() -> Molmo2FullWithExpertModel:
    prefix_spec = _tiny_spec(hidden_size=8, intermediate_size=16)
    expert_spec = _tiny_spec(hidden_size=6, intermediate_size=12)
    backend = Molmo2FullWithExpertModel.__new__(Molmo2FullWithExpertModel)
    nn.Module.__init__(backend)
    backend.vlm = molmo_core.Molmo2TextBackbone(
        prefix_spec,
        num_layers=36,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.lm_expert = molmo_core.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        num_layers=36,
        self_attn_every_n_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.vision_backbone = nn.Linear(1, 1, bias=False)
    molmo_core._initialize_module(backend, 0.02)
    backend.num_vlm_layers = 36
    backend.num_expert_layers = 36
    backend.self_attn_every_n_layers = 2
    backend.attention_mode = "cross_attn"
    backend.train_expert_only = True
    backend.freeze_vision_encoder = True
    backend.text_spec = prefix_spec
    backend.expert_spec = expert_spec
    backend.set_requires_grad()
    backend.train()
    return backend


def test_scene_gradient_crosses_all_36_frozen_text_layers_without_unfreezing_molmo() -> None:
    torch.manual_seed(23)
    prefix_builder = _make_prefix_shell(hidden_size=8, chunk_size=2)
    backend = _make_tiny_frozen_36_layer_backend()
    native, native_valid, native_image = _right_padded_native_batch((2,), hidden_size=8)
    native = native + 0.01 * torch.randn_like(native)
    scene_projection = nn.Linear(4, 8)
    scene_embeddings = scene_projection(torch.randn(1, 2, 4))
    scene_embeddings.retain_grad()

    prefix, prefix_valid, prefix_boundaries = prefix_builder._build_full_molmo_prefix(
        native,
        native_valid,
        native_image,
        scene_embeddings,
        torch.ones(1, 2, dtype=torch.bool),
        ablate_language=False,
    )
    prefix.retain_grad()
    expert = torch.randn(1, 2, 6)
    expert_valid = torch.ones(1, 2, dtype=torch.bool)
    expert_boundaries = torch.tensor([[True, False]])
    full_valid = torch.cat([prefix_valid, expert_valid], dim=1)
    full_boundaries = torch.cat([prefix_boundaries, expert_boundaries], dim=1)
    position_ids = torch.cumsum(full_valid, dim=1) - 1
    (_, expert_output), _ = backend(
        attention_mask=make_att_2d_masks(full_valid, full_boundaries),
        position_ids=position_ids,
        inputs_embeds=[prefix, expert],
        use_cache=False,
        fill_kv_cache=False,
    )
    assert expert_output is not None
    loss = (expert_output * torch.randn_like(expert_output)).sum()
    loss.backward()

    assert len(backend.vlm.blocks) == 36
    assert len(backend.lm_expert.layers) == 36
    assert not backend.vlm.training and not backend.vision_backbone.training
    assert all(not parameter.requires_grad for parameter in backend.vlm.parameters())
    assert all(not parameter.requires_grad for parameter in backend.vision_backbone.parameters())
    assert all(parameter.grad is None for parameter in backend.vlm.parameters())
    assert all(parameter.grad is None for parameter in backend.vision_backbone.parameters())

    # The action loss reaches both inserted scene positions and their trainable
    # projection despite traversing every frozen Molmo decoder operation.
    assert scene_embeddings.grad is not None
    assert torch.isfinite(scene_embeddings.grad).all()
    assert scene_embeddings.grad.abs().sum() > 0
    assert prefix.grad is not None
    assert prefix.grad[:, 411:413].abs().sum() > 0
    assert scene_projection.weight.grad is not None
    assert torch.isfinite(scene_projection.weight.grad).all()
    assert scene_projection.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in backend.lm_expert.parameters()
    )
