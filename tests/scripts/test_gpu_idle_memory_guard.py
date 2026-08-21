from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.song_real_libero.scripts.gpu_idle_memory_guard import (
    MIB,
    GuardConfig,
    aggregate_state,
    allocation_plan_bytes,
    atomic_write_json,
    build_worker_command,
    build_worker_environment,
    parse_nvidia_smi_snapshot,
)


def make_config(tmp_path: Path, gpu_indices: tuple[int, ...] = (0, 3)) -> GuardConfig:
    return GuardConfig(
        gpu_indices=gpu_indices,
        min_free_mib=31500,
        reserve_mib=30000,
        allocation_headroom_mib=512,
        chunk_mib=512,
        poll_seconds=30,
        heartbeat_seconds=600,
        query_timeout_seconds=10,
        status_dir=tmp_path,
    )


def test_guard_config_rejects_duplicate_gpus_and_unsafe_memory_boundary(tmp_path: Path) -> None:
    duplicate = make_config(tmp_path, (4, 4))
    with pytest.raises(ValueError, match="unique"):
        duplicate.validate()

    unsafe = GuardConfig(
        **{
            **make_config(tmp_path).__dict__,
            "min_free_mib": 30511,
        }
    )
    with pytest.raises(ValueError, match=r"reserve-mib \+ allocation-headroom-mib"):
        unsafe.validate()


def test_allocation_plan_exactly_matches_non_divisible_target() -> None:
    plan = allocation_plan_bytes(reserve_mib=1025, chunk_mib=512)
    assert plan == (512 * MIB, 512 * MIB, MIB)
    assert sum(plan) == 1025 * MIB


def test_nvidia_smi_parser_requires_the_requested_physical_index() -> None:
    snapshot = parse_nvidia_smi_snapshot(
        "4, GPU-01234567, 32607, 7, 32600, 0\n",
        expected_index=4,
    )
    assert snapshot.index == 4
    assert snapshot.free_mib == 32600
    assert snapshot.uuid == "GPU-01234567"

    with pytest.raises(ValueError, match="while GPU 4 was requested"):
        parse_nvidia_smi_snapshot("5, GPU-other, 32607, 7, 32600, 0\n", expected_index=4)


def test_atomic_json_replacement_leaves_no_partial_temp_file(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    atomic_write_json(status_path, {"state": "waiting", "generation": 1})
    atomic_write_json(status_path, {"state": "held", "generation": 2})

    assert json.loads(status_path.read_text()) == {"state": "held", "generation": 2}
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_worker_command_scopes_one_physical_gpu_and_preserves_safety_config(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    command = build_worker_command(Path("/repo/gpu_idle_memory_guard.py"), config, gpu_index=3)

    assert command[2:4] == ["--worker-gpu", "3"]
    assert "--gpus" not in command
    assert command[command.index("--min-free-mib") + 1] == "31500"
    assert command[command.index("--reserve-mib") + 1] == "30000"
    assert command[command.index("--heartbeat-seconds") + 1] == "600"


def test_worker_environment_uses_resolved_uuid_not_an_ambiguous_cuda_index() -> None:
    parent_environment = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "7,6"}
    environment = build_worker_environment(parent_environment, "GPU-01234567")

    assert environment["PATH"] == "/usr/bin"
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-01234567"
    assert parent_environment["CUDA_VISIBLE_DEVICES"] == "7,6"

    with pytest.raises(ValueError, match="invalid physical GPU UUID"):
        build_worker_environment(parent_environment, "4")


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["held", "held"], "held"),
        (["held", "waiting"], "partial"),
        (["fatal", "waiting"], "degraded"),
        (["allocating", "waiting"], "acquiring"),
        (["waiting", "waiting"], "waiting"),
        (["stopped", "stopped"], "stopped"),
    ],
)
def test_aggregate_state(states: list[str], expected: str) -> None:
    assert aggregate_state([{"state": state} for state in states]) == expected
