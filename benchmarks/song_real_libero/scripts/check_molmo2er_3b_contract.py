#!/usr/bin/env python

"""Fail-fast architecture checks for the point-only Molmo2-ER 3B control.

The normal invocation constructs the 3B shape on the meta device and therefore
does not read any safetensors or allocate parameter storage::

    python benchmarks/song_real_libero/scripts/check_molmo2er_3b_contract.py \
        --model-dir /raid5/rongshengwang/Lerobot/Molmo2-ER

Use ``--real-weights --device cuda --forward-backward`` only as an explicit
pre-training smoke test.  ``audit_policy`` is also intended to be called on the
fully constructed SmolVLA policy immediately before optimizer/DDP creation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

EXPECTED_VLM_PARAMETERS = 2_206_041_088
EXPECTED_EXPERT_PARAMETERS = 868_296_576
EXPECTED_WEP_EXTRA_PARAMETERS = 62_711_614
EXPECTED_WEP_TRAINABLE_PARAMETERS = 62_683_454
EXPECTED_TOTAL_PARAMETERS = 3_137_049_278
EXPECTED_TRAINABLE_PARAMETERS = 930_980_030

EXPECTED_ARCHITECTURE: dict[str, Any] = {
    "source_vlm_layers": 36,
    "retained_vlm_layers": 18,
    "retained_vlm_layer_indices": list(range(18)),
    "vlm_hidden_size": 2560,
    "vlm_intermediate_size": 9728,
    "expert_layers": 18,
    "expert_hidden_size": 1920,
    "expert_intermediate_size": 5120,
    "self_attention_layers": list(range(0, 18, 2)),
    "cross_attention_layers": list(range(1, 18, 2)),
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 5_000_000.0,
    "vision_module_present": False,
    "lm_head_present": False,
}

# This is deliberately narrow.  A newly trainable branch is an experimental
# change and must not silently enter the controlled v0.4.3 comparison.
POLICY_TRAINABLE_PREFIXES = (
    "vlm_with_expert.lm_expert.",
    "pointseg_conditioner.",
    "pointseg_object_proj.",
    "pointseg_background_proj.",
    "point_action_fusion.",
    "action_in_proj.",
    "action_out_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters on real or meta devices without materializing tensors."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def expected_parameter_budget() -> dict[str, int]:
    """Return the registered parameter budget and verify its own arithmetic."""

    budget = {
        "frozen_molmo_first_18": EXPECTED_VLM_PARAMETERS,
        "trainable_action_expert": EXPECTED_EXPERT_PARAMETERS,
        "wep_v043_extras": EXPECTED_WEP_EXTRA_PARAMETERS,
        "wep_v043_trainable_extras": EXPECTED_WEP_TRAINABLE_PARAMETERS,
        "total": EXPECTED_TOTAL_PARAMETERS,
        "trainable": EXPECTED_TRAINABLE_PARAMETERS,
    }
    if (
        budget["frozen_molmo_first_18"] + budget["trainable_action_expert"] + budget["wep_v043_extras"]
        != budget["total"]
    ):
        raise AssertionError(f"Invalid registered total parameter arithmetic: {budget}")
    if budget["trainable_action_expert"] + budget["wep_v043_trainable_extras"] != budget["trainable"]:
        raise AssertionError(f"Invalid registered trainable parameter arithmetic: {budget}")
    return budget


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def audit_backend(backend: nn.Module) -> dict[str, Any]:
    """Validate the retained Molmo text stack and alternating Action Expert."""

    contract = dict(backend.architecture_contract)
    for key, expected in EXPECTED_ARCHITECTURE.items():
        _assert_equal(contract.get(key), expected, f"architecture_contract[{key!r}]")

    _assert_equal(len(backend.vlm.blocks), 18, "number of instantiated VLM blocks")
    _assert_equal(len(backend.lm_expert.layers), 18, "number of instantiated Expert blocks")
    actual_cross = [
        index
        for index, layer in enumerate(backend.lm_expert.layers)
        if bool(getattr(layer, "is_cross_attention", False))
    ]
    _assert_equal(actual_cross, list(range(1, 18, 2)), "cross-attention Expert block indices")

    parameter_names = [name for name, _ in backend.named_parameters()]
    forbidden = [
        name
        for name in parameter_names
        if any(marker in name for marker in ("vision_backbone", "vision_model", "lm_head"))
    ]
    if forbidden:
        raise AssertionError(f"Excluded Molmo parameters were instantiated: {forbidden[:8]}")
    for excluded_attribute in ("vision_backbone", "vision_model", "lm_head"):
        if hasattr(backend, excluded_attribute) or hasattr(backend.vlm, excluded_attribute):
            raise AssertionError(f"Excluded attribute is present: {excluded_attribute}")

    vlm_parameters = count_parameters(backend.vlm)
    expert_parameters = count_parameters(backend.lm_expert)
    _assert_equal(vlm_parameters, EXPECTED_VLM_PARAMETERS, "retained VLM parameters")
    _assert_equal(expert_parameters, EXPECTED_EXPERT_PARAMETERS, "Action Expert parameters")
    _assert_equal(
        count_parameters(backend),
        EXPECTED_VLM_PARAMETERS + EXPECTED_EXPERT_PARAMETERS,
        "backend parameters",
    )
    _assert_equal(
        count_parameters(backend, trainable_only=True), EXPECTED_EXPERT_PARAMETERS, "backend trainable"
    )

    trainable_vlm = [name for name, parameter in backend.vlm.named_parameters() if parameter.requires_grad]
    frozen_expert = [
        name for name, parameter in backend.lm_expert.named_parameters() if not parameter.requires_grad
    ]
    if trainable_vlm:
        raise AssertionError(f"Frozen Molmo VLM has trainable parameters: {trainable_vlm[:8]}")
    if frozen_expert:
        raise AssertionError(f"Action Expert has unexpectedly frozen parameters: {frozen_expert[:8]}")

    # modeling_smolvla consults this switch before applying its legacy
    # sqrt(hidden_size) scale to point and language embeddings.
    _assert_equal(
        getattr(backend, "scale_input_embeddings", None),
        False,
        "Molmo input-embedding scale switch",
    )
    backend.train(True)
    if backend.vlm.training:
        raise AssertionError("Frozen Molmo VLM entered train mode (dropout would no longer be frozen).")

    report = {
        "architecture": contract,
        "vlm_parameters": vlm_parameters,
        "expert_parameters": expert_parameters,
        "backend_parameters": count_parameters(backend),
        "backend_trainable_parameters": count_parameters(backend, trainable_only=True),
    }
    load_report = getattr(backend, "load_report", None)
    if load_report is not None:
        _assert_equal(
            int(load_report["selected_parameters"]),
            EXPECTED_VLM_PARAMETERS,
            "selectively loaded VLM parameters",
        )
        _assert_equal(int(load_report["retained_blocks"]), 18, "selectively loaded blocks")
        if int(load_report.get("ignored_vision_tensors", 0)) <= 0:
            raise AssertionError("Selective loader did not report the excluded vision tensors.")
        if int(load_report.get("ignored_dropped_block_tensors", 0)) <= 0:
            raise AssertionError("Selective loader did not report excluded Molmo blocks 18--35.")
        if int(load_report.get("ignored_lm_head_tensors", 0)) <= 0:
            raise AssertionError("Selective loader did not report the excluded LM head.")
        report["load_report"] = dict(load_report)
    return report


def _normalize_policy_parameter_name(name: str) -> str:
    # SmolVLAPolicy owns VLAFlowMatching under ``model``; smoke shells can pass
    # VLAFlowMatching directly.  Both must be audited identically.
    return name.removeprefix("model.")


def audit_policy(policy: nn.Module) -> dict[str, Any]:
    """Validate exact whole-policy counts and the controlled trainable allowlist."""

    total = count_parameters(policy)
    trainable = count_parameters(policy, trainable_only=True)
    _assert_equal(total, EXPECTED_TOTAL_PARAMETERS, "whole-policy parameters")
    _assert_equal(trainable, EXPECTED_TRAINABLE_PARAMETERS, "whole-policy trainable parameters")

    unexpected_trainable = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
        and not _normalize_policy_parameter_name(name).startswith(POLICY_TRAINABLE_PREFIXES)
    ]
    if unexpected_trainable:
        raise AssertionError(
            f"Parameters outside the registered v0.4.3 allowlist are trainable: {unexpected_trainable[:12]}"
        )

    state = policy.model if hasattr(policy, "model") else policy
    backend_report = audit_backend(state.vlm_with_expert)
    config = state.config
    required_flags = {
        "vlm_backend": "molmo2_text",
        "vla_adapter_enable": False,
        "encode_robot_state": False,
        "pointseg_enable": True,
        "point_action_fusion_enable": True,
        "worldflow_enable": False,
        "se3_enable": False,
    }
    for key, expected in required_flags.items():
        _assert_equal(getattr(config, key), expected, f"controlled config {key}")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "backend": backend_report,
        "budget": expected_parameter_budget(),
    }


def validate_token_ids(token_ids: Tensor, *, total_vocab_size: int = 152_064) -> None:
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise AssertionError(f"Token IDs must be integral, got {token_ids.dtype}.")
    if token_ids.numel() == 0:
        raise AssertionError("Token sequence is empty.")
    minimum = int(token_ids.min())
    maximum = int(token_ids.max())
    if minimum < 0 or maximum >= total_vocab_size:
        raise AssertionError(
            f"Molmo token IDs must stay in [0, {total_vocab_size}); got min={minimum}, max={maximum}."
        )


def validate_point_prefix(
    prefix: Tensor,
    pad_mask: Tensor,
    language_mask: Tensor,
    *,
    token_layout: tuple[str, ...],
) -> None:
    """Check exactly one foreground and one background token before language."""

    expected_length = 2 + language_mask.shape[1]
    _assert_equal(prefix.shape[1], expected_length, "point-only prefix length")
    _assert_equal(pad_mask.shape, prefix.shape[:2], "point-only prefix pad-mask shape")
    _assert_equal(token_layout, ("foreground", "background", "language"), "prefix token layout")
    if not torch.equal(pad_mask[:, 2:].to(device="cpu"), language_mask.to(device="cpu", dtype=torch.bool)):
        raise AssertionError("Language mask is not aligned immediately after the two point tokens.")


def validate_embedding_identity(backend: nn.Module, token_ids: Tensor) -> None:
    """Ensure the wrapper returns native Molmo WTE values without sqrt(H)."""

    validate_token_ids(token_ids, total_vocab_size=int(backend.text_spec.total_vocab_size))
    actual = backend.embed_language_tokens(token_ids)
    expected = backend.vlm.emb_drop(backend.vlm.wte(token_ids))
    if not torch.equal(actual, expected):
        max_error = float((actual.float() - expected.float()).abs().max())
        raise AssertionError(f"Molmo embedding wrapper changed native WTE values (max_abs={max_error}).")


def validate_finite_forward_backward(backend: nn.Module, token_ids: Tensor) -> dict[str, float]:
    """Run one explicit real-weight joint pass and audit the gradient partition."""

    if next(backend.parameters()).is_meta:
        raise ValueError("Forward/backward smoke requires materialized parameters, not device=meta.")
    device = next(backend.parameters()).device
    token_ids = token_ids.to(device=device)
    language = backend.embed_language_tokens(token_ids)
    foreground_background = torch.randn(
        language.shape[0],
        2,
        backend.text_spec.hidden_size,
        device=device,
        dtype=language.dtype,
        requires_grad=True,
    )
    prefix = torch.cat([foreground_background, language], dim=1)
    suffix = torch.randn(
        language.shape[0],
        2,
        backend.expert_spec.hidden_size,
        device=device,
        dtype=language.dtype,
        requires_grad=True,
    )
    total_length = prefix.shape[1] + suffix.shape[1]
    attention_mask = torch.ones(prefix.shape[0], total_length, total_length, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(prefix.shape[0], -1)

    backend.zero_grad(set_to_none=True)
    outputs, _ = backend(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
    )
    if outputs[0] is None or outputs[1] is None:
        raise AssertionError("Joint Molmo/Expert forward did not return both streams.")
    loss = outputs[0].float().square().mean() + outputs[1].float().square().mean()
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite smoke loss: {float(loss)}")
    loss.backward()

    if foreground_background.grad is None or not torch.isfinite(foreground_background.grad).all():
        raise AssertionError("Point-prefix gradient is missing or non-finite.")
    if suffix.grad is None or not torch.isfinite(suffix.grad).all():
        raise AssertionError("Action-suffix gradient is missing or non-finite.")
    vlm_gradients = [name for name, parameter in backend.vlm.named_parameters() if parameter.grad is not None]
    if vlm_gradients:
        raise AssertionError(f"Frozen VLM accumulated gradients: {vlm_gradients[:8]}")
    missing_expert_gradients = [
        name
        for name, parameter in backend.lm_expert.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_expert_gradients:
        raise AssertionError(f"Expert gradient route is missing: {missing_expert_gradients[:8]}")
    nonfinite_expert_gradients = [
        name
        for name, parameter in backend.lm_expert.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_expert_gradients:
        raise AssertionError(f"Expert gradients are non-finite: {nonfinite_expert_gradients[:8]}")
    return {
        "loss": float(loss.detach()),
        "point_prefix_grad_norm": float(foreground_background.grad.float().norm()),
        "action_suffix_grad_norm": float(suffix.grad.float().norm()),
    }


def _resolve_backend_class():
    from lerobot.policies.smolvla import molmo2_with_expert

    try:
        return molmo2_with_expert.Molmo2TextHalfWithExpertModel
    except AttributeError as exc:
        raise RuntimeError(
            "The policy wiring requires the explicit Molmo2TextHalfWithExpertModel alias; "
            "the backend module does not expose it."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--forward-backward", action="store_true")
    parser.add_argument("--device", default=None, help="Default: meta, or cuda with --real-weights")
    args = parser.parse_args()

    if args.forward_backward and not args.real_weights:
        parser.error("--forward-backward requires --real-weights")
    device = args.device or ("cuda" if args.real_weights else "meta")
    if args.real_weights and str(device) == "meta":
        parser.error("--real-weights cannot use device=meta")
    if not args.model_dir.is_dir():
        parser.error(f"Molmo2-ER directory does not exist: {args.model_dir}")

    backend_class = _resolve_backend_class()
    backend = backend_class(
        model_id=str(args.model_dir),
        load_vlm_weights=args.real_weights,
        train_expert_only=True,
        attention_mode="cross_attn",
        num_expert_layers=18,
        num_vlm_layers=18,
        self_attn_every_n_layers=2,
        expert_width_multiplier=0.75,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    report: dict[str, Any] = {
        "backend": audit_backend(backend),
        "registered_whole_policy_budget": expected_parameter_budget(),
    }

    from lerobot.policies.smolvla.processor_smolvla import tokenize_molmo2_text

    tokenizer = backend.processor.tokenizer
    encoded = tokenize_molmo2_text(
        tokenizer,
        "Pick up the object.\n",
        max_length=48,
        padding="longest",
    )["input_ids"]
    validate_token_ids(encoded)
    if int(encoded[0, 0]) != backend.bos_token_id:
        raise AssertionError(
            f"Molmo prompt must begin with BOS={backend.bos_token_id}; got {int(encoded[0, 0])}."
        )
    report["bos_token_id"] = backend.bos_token_id
    report["token_id_range"] = [int(encoded.min()), int(encoded.max())]
    if not next(backend.parameters()).is_meta:
        encoded = encoded.to(device=next(backend.parameters()).device)
        validate_embedding_identity(backend, encoded)
    if args.forward_backward:
        report["forward_backward"] = validate_finite_forward_backward(backend, encoded)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
