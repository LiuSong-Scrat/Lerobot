from __future__ import annotations

from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    aggregate_exact_global_batch_cuda_memory_records,
    audit_first_optimizer_step_trainable_gradients,
    build_exact_global_batch_cuda_memory_rank_record,
    exact_global_batch_cuda_memory_audit_path,
    exact_global_batch_cuda_memory_rank_dir,
    hash_full_molmo2er_frozen_parameters,
    make_song_training_ddp_kwargs,
    optimizer_state_tensor_summary,
    resolve_exact_global_batch_plan,
    resolve_logical_physical_cuda_mapping,
    update_policy,
    write_exact_global_batch_cuda_memory_rank_record,
    write_full_molmo2er_frozen_parameter_hash_audit,
)


def _exact_plan():
    plan = resolve_exact_global_batch_plan(
        global_batch_size=192,
        batch_size=1,
        gradient_accumulation_steps=24,
        world_size=8,
    )
    assert plan is not None
    return plan


def test_logical_cuda_index_resolves_fixed_physical_gpu_contract() -> None:
    mapping = resolve_logical_physical_cuda_mapping(
        logical_cuda_index=3,
        environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )
    assert mapping == {
        "logical_cuda_index": 3,
        "physical_cuda_device": "3",
        "cuda_visible_devices": ["0", "1", "2", "3", "4", "5", "6", "7"],
    }

    assert (
        resolve_logical_physical_cuda_mapping(
            logical_cuda_index=2,
            environ={},
        )["physical_cuda_device"]
        == "2"
    )
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        resolve_logical_physical_cuda_mapping(
            logical_cuda_index=2,
            environ={"CUDA_VISIBLE_DEVICES": "5,6"},
        )


def test_eight_rank_post_adam_memory_records_are_strictly_aggregated(tmp_path) -> None:
    plan = _exact_plan()
    for rank in range(8):
        allocated = 1_000 + rank
        record = build_exact_global_batch_cuda_memory_rank_record(
            plan=plan,
            completed_optimizer_step=1,
            global_rank=rank,
            local_rank=rank,
            logical_cuda_index=rank,
            accelerator_device=f"cuda:{rank}",
            memory_snapshot={
                "allocated_bytes": allocated,
                "reserved_bytes": 2_000 + rank,
                "peak_allocated_bytes": 3_000 + rank,
                "peak_reserved_bytes": 4_000 + rank,
            },
            optimizer_state_tensor_count=12,
            optimizer_state_total_numel=345,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            hostname="a800-host",
        )
        rank_path = write_exact_global_batch_cuda_memory_rank_record(tmp_path, record)
        assert rank_path == exact_global_batch_cuda_memory_rank_dir(tmp_path) / f"rank_{rank:03d}.json"

    audit_path = aggregate_exact_global_batch_cuda_memory_records(
        tmp_path,
        plan=plan,
        expected_completed_optimizer_step=1,
    )
    assert audit_path == exact_global_batch_cuda_memory_audit_path(tmp_path)
    payload = json.loads(audit_path.read_text())
    assert payload["completed_optimizer_step"] == 1
    assert payload["world_size"] == 8
    assert payload["rank_count"] == 8
    assert payload["global_batch_size"] == 192
    assert payload["max_allocated_bytes"] == 1_007
    assert payload["max_reserved_bytes"] == 2_007
    assert payload["max_peak_allocated_bytes"] == 3_007
    assert payload["max_peak_reserved_bytes"] == 4_007
    assert [item["physical_cuda_device"] for item in payload["logical_to_physical_cuda_mapping"]] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    assert all(record["optimizer_state_tensor_count"] == 12 for record in payload["ranks"])


def test_post_adam_record_rejects_missing_optimizer_state() -> None:
    with pytest.raises(RuntimeError, match="not materialized"):
        build_exact_global_batch_cuda_memory_rank_record(
            plan=_exact_plan(),
            completed_optimizer_step=1,
            global_rank=0,
            local_rank=0,
            logical_cuda_index=0,
            accelerator_device="cuda:0",
            memory_snapshot={
                "allocated_bytes": 1,
                "reserved_bytes": 2,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 2,
            },
            optimizer_state_tensor_count=0,
            optimizer_state_total_numel=0,
            environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
            hostname="a800-host",
        )


def test_optimizer_state_summary_observes_lazy_adam_state_after_step() -> None:
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(module.parameters())
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()

    tensor_count, total_numel = optimizer_state_tensor_summary(optimizer)
    assert tensor_count == 6
    assert total_numel == 18


def _tiny_frozen_full_policy():
    vlm = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2, bias=False))
    vision_backbone = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    for parameter in (*vlm.parameters(), *vision_backbone.parameters()):
        parameter.requires_grad_(False)
    backend = SimpleNamespace(vlm=vlm, vision_backbone=vision_backbone)
    return SimpleNamespace(model=SimpleNamespace(vlm_with_expert=backend))


def test_frozen_parameter_hash_is_chunk_bounded_stable_and_detects_mutation(tmp_path) -> None:
    policy = _tiny_frozen_full_policy()
    before = hash_full_molmo2er_frozen_parameters(policy, chunk_bytes=5)
    repeated = hash_full_molmo2er_frozen_parameters(policy, chunk_bytes=5)
    assert repeated == before
    assert before["parameter_count"] == 5
    assert before["chunk_bytes"] == 5

    pass_path = write_full_molmo2er_frozen_parameter_hash_audit(
        tmp_path,
        before=before,
        after=repeated,
    )
    assert json.loads(pass_path.read_text())["comparison_pass"] is True

    with torch.no_grad():
        next(policy.model.vlm_with_expert.vlm.parameters()).view(-1)[0].add_(1)
    after = hash_full_molmo2er_frozen_parameters(policy, chunk_bytes=5)
    assert after["sha256"] != before["sha256"]
    with pytest.raises(RuntimeError, match="changed during training"):
        write_full_molmo2er_frozen_parameter_hash_audit(
            tmp_path,
            before=before,
            after=after,
        )
    assert json.loads(pass_path.read_text())["comparison_pass"] is False


def test_full_ddp_disables_unused_scan_while_legacy_backends_keep_it() -> None:
    full = make_song_training_ddp_kwargs("molmo2_full")
    assert full.find_unused_parameters is False
    assert full.static_graph is False
    assert full.gradient_as_bucket_view is True

    legacy = make_song_training_ddp_kwargs("molmo2_text")
    assert legacy.find_unused_parameters is True
    assert legacy.static_graph is False
    assert legacy.gradient_as_bucket_view is False
