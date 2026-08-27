#!/usr/bin/env python

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.song_real_libero.scripts.check_molmo2er_3b_contract import (
    EXPECTED_EXPERT_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    EXPECTED_VLM_PARAMETERS,
    audit_backend,
    expected_parameter_budget,
    validate_embedding_identity,
    validate_finite_forward_backward,
    validate_point_prefix,
    validate_token_ids,
)
from lerobot.policies.smolvla import molmo2_with_expert as molmo
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.processor_smolvla import (
    Molmo2BOSTokenizerProcessorStep,
    tokenize_molmo2_text,
)
from lerobot.processor import ProcessorStepRegistry


def _spec(
    *,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    vocab_size: int = 31,
    additional_vocab_size: int = 1,
) -> molmo.Molmo2TextSpec:
    return molmo.Molmo2TextSpec(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        additional_vocab_size=additional_vocab_size,
        hidden_act="silu",
        layer_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        max_position_embeddings=16_384,
        qkv_bias=False,
        use_qk_norm=True,
        qk_norm_type="qwen3",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        initializer_range=0.02,
    )


def _write_real_shape_config(directory: Path) -> None:
    text_config = {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "additional_vocab_size": 128,
        "hidden_act": "silu",
        "layer_norm_eps": 1e-6,
        "rope_theta": 5_000_000.0,
        "max_position_embeddings": 16_384,
        "qkv_bias": False,
        "use_qk_norm": True,
        "qk_norm_type": "qwen3",
        "embedding_dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "initializer_range": 0.02,
    }
    (directory / "config.json").write_text(json.dumps({"text_config": text_config}), encoding="utf-8")


def _make_meta_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> nn.Module:
    _write_real_shape_config(tmp_path)
    tokenizer = SimpleNamespace(
        fake_image_token_id=None,
        global_image_token_id=None,
        bos_token_id=1,
        eos_token_id=1,
    )
    monkeypatch.setattr(molmo.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    assert hasattr(molmo, "Molmo2TextHalfWithExpertModel"), (
        "modeling_smolvla imports Molmo2TextHalfWithExpertModel, so the backend must expose that "
        "explicit first-half alias."
    )
    return molmo.Molmo2TextHalfWithExpertModel(
        model_id=str(tmp_path),
        load_vlm_weights=False,
        train_expert_only=True,
        attention_mode="cross_attn",
        num_expert_layers=18,
        num_vlm_layers=18,
        self_attn_every_n_layers=2,
        expert_width_multiplier=0.75,
        device="meta",
        torch_dtype=torch.bfloat16,
    )


def _make_tiny_joint_model(dtype: torch.dtype = torch.float32) -> molmo.Molmo2WithExpertModel:
    prefix_spec = _spec()
    expert_spec = _spec(hidden_size=12, intermediate_size=24)
    model = molmo.Molmo2WithExpertModel.__new__(molmo.Molmo2WithExpertModel)
    nn.Module.__init__(model)
    model.vlm = molmo.Molmo2TextBackbone(
        prefix_spec,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    model.lm_expert = molmo.Molmo2ExpertBackbone(
        expert_spec,
        prefix_spec,
        num_layers=2,
        self_attn_every_n_layers=2,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    molmo._initialize_module(model, 0.02)
    model.num_vlm_layers = 2
    model.num_expert_layers = 2
    model.self_attn_every_n_layers = 2
    model.attention_mode = "cross_attn"
    model.train_expert_only = True
    model.expert_hidden_size = expert_spec.hidden_size
    model.num_attention_heads = expert_spec.num_attention_heads
    model.num_key_value_heads = expert_spec.num_key_value_heads
    model.head_dim = expert_spec.head_dim
    model.text_spec = prefix_spec
    model.expert_spec = expert_spec
    model.set_requires_grad()
    return model


class _PointOnlyVLM(nn.Module):
    scale_input_embeddings = False
    bos_token_id = 1

    def __init__(self, hidden_size: int):
        super().__init__()
        self.language_embedding = nn.Embedding(32, hidden_size)
        self.language_embedding.requires_grad_(False)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.language_embedding(tokens)


class _TwoTokenPointConditioner(nn.Module):
    def forward(self, payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        point_cloud = payload["point_cloud"]
        feature = point_cloud.mean(dim=1)[..., :4]
        batch_size, num_points = point_cloud.shape[:2]
        zeros = torch.zeros(batch_size, num_points, device=point_cloud.device)
        return {
            "object_feat": feature,
            "background_feat": -feature,
            "operation_prob": zeros,
            "pointseg_selection_scores": zeros,
            "pointseg_aux_metrics": {},
        }


def _make_point_prefix_shell() -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vlm_backend="molmo2_text",
        vla_adapter_enable=False,
        encode_robot_state=False,
    )
    model.vlm_with_expert = _PointOnlyVLM(hidden_size=8)
    model.pointseg_conditioner = _TwoTokenPointConditioner()
    model.pointseg_object_proj = nn.Linear(4, 8)
    model.pointseg_background_proj = nn.Linear(4, 8)
    model.extractor = None
    model.pointcloud_proj = None
    model.point_action_fusion = None
    model.worldflow_branch = None
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.inference_ablation_modalities = frozenset()
    model.capture_pointseg_visualization = False
    return model


def _controlled_config(**overrides) -> SmolVLAConfig:
    values = {
        "vlm_backend": "molmo2_text",
        "vlm_model_name": "/does/not/load/during/config-test",
        "vlm_weights_path": "/does/not/load/during/config-test",
        "load_vlm_weights": True,
        "train_expert_only": True,
        "freeze_vision_encoder": True,
        "num_vlm_layers": 18,
        "num_expert_layers": 18,
        "expert_width_multiplier": 0.75,
        "pointseg_enable": True,
        "pointseg_foreground_ratio": 0.025,
        "pointseg_background_ratio": 0.025,
        "pointseg_min_foreground_points": 2500,
        "pointseg_min_background_points": 0,
        "pointseg_aux_loss_weight": 0.0005,
        "pointseg_use_pseudo_selection": False,
        "scheduler_decay_steps": 30_000,
    }
    values.update(overrides)
    return SmolVLAConfig(**values)


def test_molmo2_er_config_contract_reads_exact_4b_source_shape(tmp_path: Path):
    _write_real_shape_config(tmp_path)
    spec = molmo.Molmo2TextSpec.from_model_directory(tmp_path)
    molmo.validate_molmo2_er_text_contract(spec)

    assert spec.num_hidden_layers == 36
    assert spec.hidden_size == 2560
    assert spec.intermediate_size == 9728
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.head_dim == 128
    assert spec.rope_theta == 5_000_000.0
    assert spec.total_vocab_size == 152_064


def test_registered_policy_config_locks_point_only_v043_control_and_survives_json_lists():
    config = _controlled_config(optimizer_betas=[0.9, 0.95])
    assert config.selected_camera_views == ("agentview",)
    assert config.selected_rgb_camera_views == ()

    with pytest.raises(ValueError, match="locked v0.4.3 control"):
        _controlled_config(pointseg_enable=False)
    with pytest.raises(ValueError, match="locked v0.4.3 control"):
        _controlled_config(encode_robot_state=True)
    with pytest.raises(ValueError, match="locked v0.4.3 control"):
        _controlled_config(prefix_length=64)


def test_registered_policy_config_accepts_only_cold_start_and_continuation_lr_floors():
    assert _controlled_config(scheduler_decay_lr=2.5e-6).scheduler_decay_lr == 2.5e-6
    assert _controlled_config(scheduler_decay_lr=3e-5).scheduler_decay_lr == 3e-5

    with pytest.raises(ValueError, match="scheduler_decay_lr must be either"):
        _controlled_config(scheduler_decay_lr=1e-5)

    with pytest.raises(ValueError, match="locked v0.4.3 control"):
        _controlled_config(worldflow_bootstrap_from_ego=True)


def test_parameter_budget_is_derived_from_registered_widths_and_depths():
    vlm_hidden, vlm_intermediate = 2560, 9728
    expert_hidden, expert_intermediate = 1920, 5120
    query_dim, kv_dim, head_dim = 4096, 1024, 128

    vlm_attention = vlm_hidden * (query_dim + 2 * kv_dim) + 2 * head_dim + query_dim * vlm_hidden
    vlm_mlp_and_norms = vlm_hidden * (2 * vlm_intermediate) + vlm_intermediate * vlm_hidden + 2 * vlm_hidden
    derived_vlm = 152_064 * vlm_hidden + 18 * (vlm_attention + vlm_mlp_and_norms) + vlm_hidden

    expert_common = (
        expert_hidden * (2 * expert_intermediate) + expert_intermediate * expert_hidden + 2 * expert_hidden
    )
    expert_self_attention = (
        expert_hidden * (query_dim + 2 * kv_dim) + 2 * head_dim + query_dim * expert_hidden
    )
    expert_cross_attention = (
        expert_hidden * query_dim + 2 * kv_dim * kv_dim + 2 * head_dim + query_dim * expert_hidden
    )
    derived_expert = (
        9 * (expert_self_attention + expert_common)
        + 9 * (expert_cross_attention + expert_common)
        + expert_hidden
    )

    assert derived_vlm == EXPECTED_VLM_PARAMETERS
    assert derived_expert == EXPECTED_EXPERT_PARAMETERS
    budget = expected_parameter_budget()
    assert budget["total"] == EXPECTED_TOTAL_PARAMETERS == 3_137_049_278
    assert budget["trainable"] == EXPECTED_TRAINABLE_PARAMETERS == 930_980_030


def test_meta_model_has_only_first_half_molmo_and_alternating_expert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = _make_meta_backend(tmp_path, monkeypatch)
    report = audit_backend(backend)

    assert report["vlm_parameters"] == 2_206_041_088
    assert report["expert_parameters"] == 868_296_576
    assert [layer.is_cross_attention for layer in backend.lm_expert.layers] == [
        index % 2 == 1 for index in range(18)
    ]
    assert not any("blocks.18." in name for name, _ in backend.named_parameters())
    assert not hasattr(backend, "vision_backbone")
    assert not hasattr(backend, "lm_head")
    with pytest.raises(RuntimeError, match="point-cloud-only"):
        backend.embed_image(torch.empty(1, 3, 4, 4, device="meta"))


def test_point_prefix_is_exactly_foreground_background_then_language_without_sqrt_scale():
    torch.manual_seed(9)
    model = _make_point_prefix_shell()
    point_cloud = torch.randn(2, 7, 6)
    point_mask = torch.tensor([True, True])
    language = torch.tensor([[1, 2, 3], [1, 5, 0]])
    language_mask = torch.tensor([[True, True, True], [True, True, False]])

    conditioned = model.pointseg_conditioner({"point_cloud": point_cloud})
    expected_foreground = model.pointseg_object_proj(conditioned["object_feat"])
    expected_background = model.pointseg_background_proj(conditioned["background_feat"])
    expected_language = model.vlm_with_expert.embed_language_tokens(language)
    prefix, pad_mask, block_mask = model.embed_prefix(
        [point_cloud],
        [point_mask],
        language,
        language_mask,
    )

    validate_point_prefix(
        prefix,
        pad_mask,
        language_mask,
        token_layout=model.last_prefix_token_layout,
    )
    torch.testing.assert_close(prefix[:, 0], expected_foreground)
    torch.testing.assert_close(prefix[:, 1], expected_background)
    torch.testing.assert_close(prefix[:, 2:], expected_language)
    assert not block_mask.any()
    assert torch.equal(pad_mask[:, :2], torch.ones(2, 2, dtype=torch.bool))
    assert torch.isfinite(model.last_prefix_metrics["point_language_rms_ratio"])


def test_point_only_prefix_rejects_every_2d_image_input():
    model = _make_point_prefix_shell()
    with pytest.raises(ValueError, match="must never receive 2-D RGB"):
        model.embed_prefix(
            [torch.randn(1, 4, 6)],
            [torch.ones(1, dtype=torch.bool)],
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            images=[torch.randn(1, 3, 8, 8)],
            image_masks=[torch.ones(1, dtype=torch.bool)],
        )


def test_point_only_prefix_rejects_language_without_molmo_bos():
    model = _make_point_prefix_shell()
    with pytest.raises(ValueError, match="native BOS"):
        model.embed_prefix(
            [torch.randn(1, 4, 6)],
            [torch.ones(1, dtype=torch.bool)],
            torch.tensor([[2, 3]], dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
        )


def test_real_molmo_tokenizer_inserts_one_bos_within_48_token_cap_and_roundtrips():
    model_dir = Path("/raid5/rongshengwang/Lerobot/Molmo2-ER")
    if not model_dir.is_dir():
        pytest.skip(f"Local Molmo tokenizer is unavailable: {model_dir}")

    step = Molmo2BOSTokenizerProcessorStep(
        tokenizer_name=str(model_dir),
        max_length=48,
        padding="longest",
        padding_side="right",
    )
    raw = step.input_tokenizer("Pick up the object.\n", return_tensors="pt")
    bos_token_id = int(step.input_tokenizer.bos_token_id)
    assert int(raw["input_ids"][0, 0]) != bos_token_id

    tokenized = step._tokenize_text(["Pick up the object.\n", "token " * 200])
    assert tokenized["input_ids"].shape == tokenized["attention_mask"].shape
    assert tokenized["input_ids"].shape[1] == 48
    assert torch.equal(tokenized["input_ids"][:, 0], torch.full((2,), bos_token_id))
    assert tokenized["attention_mask"].dtype == torch.bool
    assert tokenized["attention_mask"][:, 0].all()
    assert (tokenized["input_ids"] == bos_token_id).sum(dim=1).tolist() == [1, 1]

    already_prefixed = tokenize_molmo2_text(
        step.input_tokenizer,
        f"{step.input_tokenizer.bos_token}Pick up the object.\n",
        max_length=48,
        padding="longest",
    )
    assert int((already_prefixed["input_ids"] == bos_token_id).sum()) == 1

    restored_cls = ProcessorStepRegistry.get("molmo2_bos_tokenizer_processor")
    restored = restored_cls(**step.get_config())
    restored_tokens = restored._tokenize_text("Pick up the object.\n")
    assert torch.equal(
        restored_tokens["input_ids"],
        tokenize_molmo2_text(
            step.input_tokenizer,
            "Pick up the object.\n",
            max_length=48,
            padding="longest",
        )["input_ids"],
    )


def test_molmo_embedding_is_unscaled_and_checks_full_checkpoint_vocab():
    torch.manual_seed(4)
    model = _make_tiny_joint_model()
    token_ids = torch.tensor([[0, 4, 31]])
    validate_token_ids(token_ids, total_vocab_size=32)
    validate_embedding_identity(model, token_ids)

    native = model.vlm.wte(token_ids)
    actual = model.embed_language_tokens(token_ids)
    assert torch.equal(actual, native)
    assert not torch.equal(actual, native * native.shape[-1] ** 0.5)
    with pytest.raises(AssertionError, match="must stay"):
        validate_token_ids(torch.tensor([[152_064]]))
    with pytest.raises(IndexError, match="outside"):
        model.embed_language_tokens(torch.tensor([[32]]))


def test_tiny_joint_forward_backward_is_finite_and_keeps_vlm_frozen():
    torch.manual_seed(8)
    model = _make_tiny_joint_model()
    diagnostics = validate_finite_forward_backward(model, torch.tensor([[1, 2, 3]]))

    assert diagnostics["loss"] >= 0.0
    assert diagnostics["point_prefix_grad_norm"] > 0.0
    assert diagnostics["action_suffix_grad_norm"] > 0.0
    assert all(parameter.grad is None for parameter in model.vlm.parameters())
    assert all(parameter.grad is not None for parameter in model.lm_expert.parameters())


def test_bfloat16_backend_accepts_float32_policy_embeddings_without_amp_and_cache_matches_joint():
    """The controlled train run has FP32 WEP projections and BF16 transformers."""

    torch.manual_seed(18)
    model = _make_tiny_joint_model(dtype=torch.bfloat16).eval()
    prefix = torch.randn(1, 4, model.text_spec.hidden_size, dtype=torch.float32, requires_grad=True)
    suffix = torch.randn(1, 3, model.expert_spec.hidden_size, dtype=torch.float32, requires_grad=True)
    total_length = prefix.shape[1] + suffix.shape[1]
    joint_mask = torch.ones(1, total_length, total_length, dtype=torch.bool)
    joint_positions = torch.arange(total_length).unsqueeze(0)

    joint, _ = model(
        attention_mask=joint_mask,
        position_ids=joint_positions,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        fill_kv_cache=False,
    )
    assert joint[0] is not None and joint[1] is not None
    assert joint[0].dtype == torch.bfloat16
    assert joint[1].dtype == torch.bfloat16
    joint[1].float().square().mean().backward()
    assert prefix.grad is not None and torch.isfinite(prefix.grad).all() and prefix.grad.abs().sum() > 0
    assert suffix.grad is not None and torch.isfinite(suffix.grad).all() and suffix.grad.abs().sum() > 0

    with torch.no_grad():
        _, cache = model(
            attention_mask=torch.ones(1, prefix.shape[1], prefix.shape[1], dtype=torch.bool),
            position_ids=torch.arange(prefix.shape[1]).unsqueeze(0),
            inputs_embeds=[prefix.detach(), None],
            use_cache=True,
            fill_kv_cache=True,
        )
        cached, _ = model(
            attention_mask=torch.ones(1, suffix.shape[1], total_length, dtype=torch.bool),
            position_ids=joint_positions[:, -suffix.shape[1] :],
            past_key_values=cache,
            inputs_embeds=[None, suffix.detach()],
            use_cache=True,
            fill_kv_cache=False,
        )
    assert cached[1] is not None
    torch.testing.assert_close(cached[1], joint[1].detach(), rtol=0.0, atol=0.0)


def test_tiny_backend_checkpoint_roundtrip_is_exact(tmp_path: Path):
    torch.manual_seed(11)
    source = _make_tiny_joint_model().eval()
    prefix = torch.randn(1, 4, source.text_spec.hidden_size)
    suffix = torch.randn(1, 3, source.expert_spec.hidden_size)
    total_length = prefix.shape[1] + suffix.shape[1]
    mask = torch.ones(1, total_length, total_length, dtype=torch.bool)
    position_ids = torch.arange(total_length).unsqueeze(0)
    expected, _ = source(
        attention_mask=mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
    )

    checkpoint = tmp_path / "tiny_molmo2_expert.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = _make_tiny_joint_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    actual, _ = restored(
        attention_mask=mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
    )

    assert expected[0] is not None and expected[1] is not None
    assert actual[0] is not None and actual[1] is not None
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    assert all(parameter.requires_grad is False for parameter in restored.vlm.parameters())
    assert all(parameter.requires_grad is True for parameter in restored.lm_expert.parameters())


@pytest.mark.skipif(
    os.environ.get("LEROBOT_RUN_MOLMO2_REAL_SMOKE") != "1",
    reason="Set LEROBOT_RUN_MOLMO2_REAL_SMOKE=1 to load the explicit local 3B checkpoint smoke.",
)
def test_optional_real_weights_forward_backward_smoke():
    model_dir = Path(os.environ.get("MOLMO2_ER_MODEL_DIR", "/raid5/rongshengwang/Lerobot/Molmo2-ER"))
    if not model_dir.is_dir():
        pytest.fail(f"MOLMO2_ER_MODEL_DIR does not exist: {model_dir}")
    if not torch.cuda.is_available():
        pytest.fail("The explicit real-weight smoke requires a CUDA GPU.")

    backend = molmo.Molmo2TextHalfWithExpertModel(
        model_id=str(model_dir),
        load_vlm_weights=True,
        train_expert_only=True,
        attention_mode="cross_attn",
        num_expert_layers=18,
        num_vlm_layers=18,
        self_attn_every_n_layers=2,
        expert_width_multiplier=0.75,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    audit_backend(backend)
    token_ids = tokenize_molmo2_text(
        backend.processor.tokenizer,
        "Pick up the object.\n",
        max_length=48,
        padding="longest",
    )["input_ids"]
    assert int(token_ids[0, 0]) == backend.bos_token_id == 151_645
    validate_token_ids(token_ids)
    validate_embedding_identity(backend, token_ids.to(device="cuda"))
    validate_finite_forward_backward(backend, token_ids)
