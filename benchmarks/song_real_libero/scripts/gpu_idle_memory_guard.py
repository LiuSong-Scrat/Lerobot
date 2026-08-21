#!/usr/bin/env python3

"""Reserve CUDA memory on selected GPUs only after each GPU becomes idle.

The coordinator starts one worker per physical GPU.  A worker uses read-only
``nvidia-smi`` queries while it waits; it imports torch and creates a CUDA
context only after that GPU meets the configured free-memory threshold.
Workers only receive signals from the coordinator that created them.  This
program never discovers, signals, or otherwise manages unrelated processes.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
RETRYABLE_WORKER_EXIT = 75
FATAL_WORKER_EXIT = 78


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one complete JSON object and atomically replace ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def log(message: str, **fields: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[gpu-idle-guard] {utc_now()} {message}{' ' if suffix else ''}{suffix}", flush=True)


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "total_mib": self.total_mib,
            "used_mib": self.used_mib,
            "free_mib": self.free_mib,
            "utilization_percent": self.utilization_percent,
        }


def parse_nvidia_smi_snapshot(output: str, expected_index: int) -> GpuSnapshot:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one nvidia-smi row for GPU {expected_index}, got {len(lines)}")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 6:
        raise ValueError(f"expected six nvidia-smi fields for GPU {expected_index}, got {len(fields)}")
    try:
        index, total_mib, used_mib, free_mib, utilization_percent = (
            int(fields[0]),
            int(fields[2]),
            int(fields[3]),
            int(fields[4]),
            int(fields[5]),
        )
    except ValueError as error:
        raise ValueError(f"non-integer nvidia-smi field for GPU {expected_index}: {lines[0]}") from error
    if index != expected_index:
        raise ValueError(f"nvidia-smi returned GPU {index} while GPU {expected_index} was requested")
    if min(total_mib, used_mib, free_mib, utilization_percent) < 0:
        raise ValueError(f"negative nvidia-smi field for GPU {expected_index}: {lines[0]}")
    return GpuSnapshot(
        index=index,
        uuid=fields[1],
        total_mib=total_mib,
        used_mib=used_mib,
        free_mib=free_mib,
        utilization_percent=utilization_percent,
    )


def query_gpu_snapshot(gpu_index: int, timeout_seconds: float) -> GpuSnapshot:
    command = [
        "nvidia-smi",
        "--id",
        str(gpu_index),
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return parse_nvidia_smi_snapshot(result.stdout, gpu_index)


def allocation_plan_bytes(reserve_mib: int, chunk_mib: int) -> tuple[int, ...]:
    target_bytes = reserve_mib * MIB
    chunk_bytes = chunk_mib * MIB
    full_chunks, remainder = divmod(target_bytes, chunk_bytes)
    plan = [chunk_bytes] * full_chunks
    if remainder:
        plan.append(remainder)
    return tuple(plan)


@dataclass(frozen=True)
class GuardConfig:
    gpu_indices: tuple[int, ...]
    min_free_mib: int
    reserve_mib: int
    allocation_headroom_mib: int
    chunk_mib: int
    poll_seconds: float
    heartbeat_seconds: float
    query_timeout_seconds: float
    status_dir: Path

    def validate(self) -> None:
        if not self.gpu_indices:
            raise ValueError("at least one physical GPU index is required")
        if any(index < 0 for index in self.gpu_indices):
            raise ValueError("physical GPU indices must be non-negative")
        if len(set(self.gpu_indices)) != len(self.gpu_indices):
            raise ValueError("physical GPU indices must be unique")
        for name, value in (
            ("min-free-mib", self.min_free_mib),
            ("reserve-mib", self.reserve_mib),
            ("allocation-headroom-mib", self.allocation_headroom_mib),
            ("chunk-mib", self.chunk_mib),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chunk_mib > self.reserve_mib:
            raise ValueError("chunk-mib cannot exceed reserve-mib")
        if self.min_free_mib < self.reserve_mib + self.allocation_headroom_mib:
            raise ValueError(
                "min-free-mib must be at least reserve-mib + allocation-headroom-mib "
                "so the guard cannot intentionally exhaust a GPU"
            )
        if self.poll_seconds < 1:
            raise ValueError("poll-seconds must be at least 1")
        if self.heartbeat_seconds < 10:
            raise ValueError("heartbeat-seconds must be at least 10")
        if self.query_timeout_seconds < 1:
            raise ValueError("query-timeout-seconds must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu_indices": list(self.gpu_indices),
            "min_free_mib": self.min_free_mib,
            "reserve_mib": self.reserve_mib,
            "allocation_headroom_mib": self.allocation_headroom_mib,
            "chunk_mib": self.chunk_mib,
            "poll_seconds": self.poll_seconds,
            "heartbeat_seconds": self.heartbeat_seconds,
            "query_timeout_seconds": self.query_timeout_seconds,
            "status_dir": str(self.status_dir),
        }


class StatusReporter:
    def __init__(self, path: Path, heartbeat_seconds: float, identity: dict[str, Any]) -> None:
        self.path = path
        self.heartbeat_seconds = heartbeat_seconds
        self.identity = identity
        self.started_at = utc_now()
        self._last_log_key: tuple[object, ...] | None = None
        self._next_log_monotonic = 0.0

    def report(self, state: str, message: str, *, force_log: bool = False, **fields: Any) -> None:
        payload = {
            "schema_version": 1,
            **self.identity,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "state": state,
            "message": message,
            **fields,
        }
        atomic_write_json(self.path, payload)
        now = time.monotonic()
        log_key = (state, message)
        if force_log or log_key != self._last_log_key or now >= self._next_log_monotonic:
            printable_fields = {
                key: value
                for key, value in fields.items()
                if key in {"gpu_index", "observed_free_mib", "reserved_mib", "child_pid", "exit_code"}
            }
            log(message, state=state, **printable_fields)
            self._last_log_key = log_key
            self._next_log_monotonic = now + self.heartbeat_seconds


def install_stop_handlers(stop_event: threading.Event) -> None:
    def request_stop(signum: int, _frame: object) -> None:
        log("stop requested", signal=signal.Signals(signum).name, pid=os.getpid())
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)


class StopRequestedError(Exception):
    pass


def reserve_cuda_memory(
    config: GuardConfig,
    gpu_index: int,
    stop_event: threading.Event,
) -> tuple[list[Any], dict[str, int]]:
    """Allocate live uint8 tensors on the sole visible CUDA device."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is unavailable in the guard Python environment") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker for physical GPU {gpu_index} expected exactly one visible CUDA device; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    required_bytes = (config.reserve_mib + config.allocation_headroom_mib) * MIB
    if free_bytes < required_bytes:
        raise MemoryError(
            f"runtime CUDA free memory dropped to {free_bytes // MIB} MiB; "
            f"need {required_bytes // MIB} MiB including headroom"
        )

    held_tensors: list[Any] = []
    try:
        for chunk_bytes in allocation_plan_bytes(config.reserve_mib, config.chunk_mib):
            if stop_event.is_set():
                raise StopRequestedError
            held_tensors.append(torch.empty((chunk_bytes,), dtype=torch.uint8, device="cuda:0"))
        torch.cuda.synchronize(0)
    except BaseException:
        held_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        raise

    post_free_bytes, _ = torch.cuda.mem_get_info(0)
    return held_tensors, {
        "cuda_total_mib": total_bytes // MIB,
        "cuda_free_before_mib": free_bytes // MIB,
        "cuda_free_after_mib": post_free_bytes // MIB,
        "torch_allocated_mib": torch.cuda.memory_allocated(0) // MIB,
        "torch_reserved_mib": torch.cuda.memory_reserved(0) // MIB,
    }


def release_cuda_memory(held_tensors: list[Any]) -> None:
    if not held_tensors:
        return
    import torch

    held_tensors.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)


def worker_status_path(status_dir: Path, gpu_index: int) -> Path:
    return status_dir / f"gpu_{gpu_index:03d}.json"


def run_worker(config: GuardConfig, gpu_index: int) -> int:
    stop_event = threading.Event()
    install_stop_handlers(stop_event)
    reporter = StatusReporter(
        worker_status_path(config.status_dir, gpu_index),
        config.heartbeat_seconds,
        {"role": "worker", "pid": os.getpid(), "gpu_index": gpu_index},
    )
    last_error: str | None = None
    qualifying_snapshot: GpuSnapshot | None = None

    while not stop_event.is_set():
        try:
            snapshot = query_gpu_snapshot(gpu_index, config.query_timeout_seconds)
            last_error = None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            reporter.report(
                "waiting",
                "GPU query failed; waiting without creating a CUDA context",
                gpu_index=gpu_index,
                observed_free_mib=None,
                last_error=last_error,
            )
            stop_event.wait(config.poll_seconds)
            continue

        if snapshot.free_mib < config.min_free_mib:
            reporter.report(
                "waiting",
                "GPU is busy; waiting without creating a CUDA context",
                gpu_index=gpu_index,
                observed_free_mib=snapshot.free_mib,
                threshold_free_mib=config.min_free_mib,
                snapshot=snapshot.as_dict(),
                last_error=last_error,
            )
            stop_event.wait(config.poll_seconds)
            continue
        qualifying_snapshot = snapshot
        break

    if stop_event.is_set():
        reporter.report(
            "stopped",
            "worker stopped before CUDA allocation",
            force_log=True,
            gpu_index=gpu_index,
            observed_free_mib=qualifying_snapshot.free_mib if qualifying_snapshot else None,
        )
        return 0

    assert qualifying_snapshot is not None
    reporter.report(
        "allocating",
        "free-memory threshold met; starting guarded allocation",
        force_log=True,
        gpu_index=gpu_index,
        observed_free_mib=qualifying_snapshot.free_mib,
        threshold_free_mib=config.min_free_mib,
        requested_reserve_mib=config.reserve_mib,
        snapshot=qualifying_snapshot.as_dict(),
    )

    held_tensors: list[Any] = []
    try:
        held_tensors, allocation_stats = reserve_cuda_memory(config, gpu_index, stop_event)
    except StopRequestedError:
        reporter.report(
            "stopped",
            "worker stopped during CUDA allocation and released partial allocations",
            force_log=True,
            gpu_index=gpu_index,
        )
        return 0
    except RuntimeError as error:
        error_message = f"{type(error).__name__}: {error}"
        fatal = "torch is unavailable" in str(error) or "expected exactly one visible CUDA device" in str(
            error
        )
        reporter.report(
            "fatal" if fatal else "allocation_failed",
            "CUDA allocation failed; process will exit so its CUDA context is released",
            force_log=True,
            gpu_index=gpu_index,
            observed_free_mib=qualifying_snapshot.free_mib,
            last_error=error_message,
        )
        return FATAL_WORKER_EXIT if fatal else RETRYABLE_WORKER_EXIT
    except (MemoryError, OSError) as error:
        reporter.report(
            "allocation_failed",
            "CUDA allocation race detected; process will exit so its CUDA context is released",
            force_log=True,
            gpu_index=gpu_index,
            observed_free_mib=qualifying_snapshot.free_mib,
            last_error=f"{type(error).__name__}: {error}",
        )
        return RETRYABLE_WORKER_EXIT

    try:
        while not stop_event.is_set():
            try:
                held_snapshot = query_gpu_snapshot(gpu_index, config.query_timeout_seconds)
                snapshot_payload: dict[str, Any] | None = held_snapshot.as_dict()
                held_free_mib: int | None = held_snapshot.free_mib
                query_error: str | None = None
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                snapshot_payload = None
                held_free_mib = None
                query_error = f"{type(error).__name__}: {error}"
            reporter.report(
                "held",
                "CUDA memory reservation is active",
                gpu_index=gpu_index,
                observed_free_mib=held_free_mib,
                reserved_mib=config.reserve_mib,
                allocation=allocation_stats,
                snapshot=snapshot_payload,
                last_error=query_error,
            )
            stop_event.wait(config.poll_seconds)
    finally:
        reporter.report(
            "stopping",
            "releasing CUDA memory reservation",
            force_log=True,
            gpu_index=gpu_index,
            reserved_mib=config.reserve_mib,
        )
        release_cuda_memory(held_tensors)
        reporter.report(
            "stopped",
            "CUDA memory reservation released",
            force_log=True,
            gpu_index=gpu_index,
            reserved_mib=0,
        )
    return 0


def build_worker_command(script_path: Path, config: GuardConfig, gpu_index: int) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "--worker-gpu",
        str(gpu_index),
        "--min-free-mib",
        str(config.min_free_mib),
        "--reserve-mib",
        str(config.reserve_mib),
        "--allocation-headroom-mib",
        str(config.allocation_headroom_mib),
        "--chunk-mib",
        str(config.chunk_mib),
        "--poll-seconds",
        str(config.poll_seconds),
        "--heartbeat-seconds",
        str(config.heartbeat_seconds),
        "--query-timeout-seconds",
        str(config.query_timeout_seconds),
        "--status-dir",
        str(config.status_dir),
    ]


def build_worker_environment(parent_environment: Mapping[str, str], gpu_uuid: str) -> dict[str, str]:
    """Scope a worker to an exact physical GPU without relying on index ordering."""

    if not gpu_uuid.startswith("GPU-"):
        raise ValueError(f"invalid physical GPU UUID from nvidia-smi: {gpu_uuid}")
    environment = dict(parent_environment)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return environment


@dataclass
class ChildSlot:
    gpu_index: int
    gpu_uuid: str | None = None
    process: subprocess.Popen[str] | None = None
    restart_at: float = 0.0
    exit_code: int | None = None
    fatal: bool = False
    launch_count: int = 0
    pid_history: list[int] = field(default_factory=list)


def aggregate_state(worker_payloads: Sequence[dict[str, Any] | None], stopping: bool = False) -> str:
    if stopping:
        return "stopping"
    states = [payload.get("state") if payload is not None else "starting" for payload in worker_payloads]
    if states and all(state == "held" for state in states):
        return "held"
    if "fatal" in states:
        return "degraded"
    if "held" in states:
        return "partial"
    if any(state in {"allocating", "allocation_failed"} for state in states):
        return "acquiring"
    if any(state == "waiting" for state in states):
        return "waiting"
    if states and all(state == "stopped" for state in states):
        return "stopped"
    return "starting"


def acquire_status_lock(status_dir: Path) -> int:
    status_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = status_dir / "guard.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise RuntimeError(f"another GPU guard already holds {lock_path}") from None
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.fsync(lock_fd)
    return lock_fd


def stop_owned_workers(slots: Sequence[ChildSlot], heartbeat_seconds: float) -> None:
    """Ask only coordinator-owned workers to exit, then wait for release."""

    waiting: list[ChildSlot] = []
    for slot in slots:
        if slot.process is not None and slot.process.poll() is None:
            slot.process.send_signal(signal.SIGTERM)
            waiting.append(slot)

    next_wait_log = time.monotonic() + heartbeat_seconds
    while waiting:
        for slot in list(waiting):
            assert slot.process is not None
            exit_code = slot.process.poll()
            if exit_code is not None:
                slot.exit_code = exit_code
                waiting.remove(slot)
        if waiting and time.monotonic() >= next_wait_log:
            log(
                "waiting for owned workers to release CUDA memory gracefully",
                child_pids=",".join(str(slot.process.pid) for slot in waiting if slot.process),
            )
            next_wait_log = time.monotonic() + heartbeat_seconds
        if waiting:
            time.sleep(0.2)


def run_coordinator(config: GuardConfig) -> int:
    lock_fd = acquire_status_lock(config.status_dir)
    stop_event = threading.Event()
    install_stop_handlers(stop_event)
    script_path = Path(__file__).resolve()
    reporter = StatusReporter(
        config.status_dir / "status.json",
        config.heartbeat_seconds,
        {"role": "coordinator", "pid": os.getpid(), "config": config.as_dict()},
    )
    slots = {gpu_index: ChildSlot(gpu_index=gpu_index) for gpu_index in config.gpu_indices}

    def launch_worker(slot: ChildSlot) -> None:
        if slot.gpu_uuid is None:
            raise RuntimeError(f"physical GPU {slot.gpu_index} has not been resolved to a UUID")
        environment = build_worker_environment(os.environ, slot.gpu_uuid)
        command = build_worker_command(script_path, config, slot.gpu_index)
        slot.process = subprocess.Popen(command, env=environment, text=True)
        slot.launch_count += 1
        slot.pid_history.append(slot.process.pid)
        slot.exit_code = None
        slot.restart_at = 0.0
        atomic_write_json(
            worker_status_path(config.status_dir, slot.gpu_index),
            {
                "schema_version": 1,
                "role": "worker",
                "pid": slot.process.pid,
                "gpu_index": slot.gpu_index,
                "gpu_uuid": slot.gpu_uuid,
                "state": "starting",
                "message": "worker launched; no CUDA context has been created",
                "started_at": utc_now(),
                "updated_at": utc_now(),
            },
        )
        log("worker launched", gpu_index=slot.gpu_index, child_pid=slot.process.pid)

    try:
        # Validate every requested physical index using read-only queries before
        # creating children.  No CUDA context is created by this validation.
        for gpu_index in config.gpu_indices:
            slots[gpu_index].gpu_uuid = query_gpu_snapshot(gpu_index, config.query_timeout_seconds).uuid
        for slot in slots.values():
            launch_worker(slot)

        last_payload_signature: str | None = None
        while not stop_event.is_set():
            now = time.monotonic()
            for slot in slots.values():
                if slot.process is not None:
                    exit_code = slot.process.poll()
                    if exit_code is not None:
                        slot.exit_code = exit_code
                        slot.process = None
                        if exit_code == FATAL_WORKER_EXIT:
                            slot.fatal = True
                            log("worker reported a fatal configuration error", gpu_index=slot.gpu_index)
                        else:
                            slot.restart_at = now + config.poll_seconds
                            log(
                                "worker exited; retry scheduled after poll interval",
                                gpu_index=slot.gpu_index,
                                exit_code=exit_code,
                            )
                elif not slot.fatal and now >= slot.restart_at:
                    launch_worker(slot)

            worker_payloads = [
                read_json_if_present(worker_status_path(config.status_dir, gpu_index))
                for gpu_index in config.gpu_indices
            ]
            child_details = {
                str(gpu_index): {
                    "pid": slot.process.pid if slot.process is not None else None,
                    "gpu_uuid": slot.gpu_uuid,
                    "exit_code": slot.exit_code,
                    "fatal": slot.fatal,
                    "launch_count": slot.launch_count,
                    "pid_history": slot.pid_history,
                    "status": worker_payloads[position],
                }
                for position, (gpu_index, slot) in enumerate(slots.items())
            }
            state = aggregate_state(worker_payloads)
            signature = json.dumps(
                [(payload or {}).get("state") for payload in worker_payloads], separators=(",", ":")
            )
            reporter.report(
                state,
                "GPU guard coordinator status",
                force_log=signature != last_payload_signature,
                children=child_details,
            )
            last_payload_signature = signature
            stop_event.wait(1.0)

        reporter.report(
            "stopping",
            "coordinator is signaling only its own workers to release reservations",
            force_log=True,
            children={
                str(index): {"pid": slot.process.pid if slot.process is not None else None}
                for index, slot in slots.items()
            },
        )
        stop_owned_workers(list(slots.values()), config.heartbeat_seconds)

        final_payloads = [
            read_json_if_present(worker_status_path(config.status_dir, gpu_index))
            for gpu_index in config.gpu_indices
        ]
        reporter.report(
            "stopped",
            "all owned workers exited and all CUDA reservations were released",
            force_log=True,
            children={
                str(gpu_index): {
                    "exit_code": slots[gpu_index].exit_code,
                    "status": final_payloads[position],
                }
                for position, gpu_index in enumerate(config.gpu_indices)
            },
        )
        return 0
    finally:
        # This also covers unexpected coordinator errors.  Never leave an
        # owned reservation orphaned, and never signal a PID not stored in a
        # ChildSlot created by this coordinator.
        stop_owned_workers(list(slots.values()), config.heartbeat_seconds)
        os.close(lock_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for selected physical GPUs to become idle, then hold CUDA memory without touching "
            "unrelated processes. Stop with SIGTERM or Ctrl-C to release all reservations."
        )
    )
    parser.add_argument("--gpus", nargs="+", type=int, help="physical GPU indices managed by the coordinator")
    parser.add_argument("--worker-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--min-free-mib", type=int, default=31500)
    parser.add_argument("--reserve-mib", type=int, default=30000)
    parser.add_argument("--allocation-headroom-mib", type=int, default=512)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--heartbeat-seconds", type=float, default=600)
    parser.add_argument("--query-timeout-seconds", type=float, default=10)
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path(os.environ.get("GPU_GUARD_STATUS_DIR", f"/tmp/lerobot_gpu_guard_{os.getuid()}")),
    )
    return parser


def config_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[GuardConfig, int | None]:
    if args.worker_gpu is None:
        if not args.gpus:
            parser.error("--gpus is required in coordinator mode")
        gpu_indices = tuple(args.gpus)
    else:
        if args.gpus:
            parser.error("--gpus and the internal --worker-gpu option are mutually exclusive")
        gpu_indices = (args.worker_gpu,)
    config = GuardConfig(
        gpu_indices=gpu_indices,
        min_free_mib=args.min_free_mib,
        reserve_mib=args.reserve_mib,
        allocation_headroom_mib=args.allocation_headroom_mib,
        chunk_mib=args.chunk_mib,
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        query_timeout_seconds=args.query_timeout_seconds,
        status_dir=args.status_dir.expanduser().resolve(),
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    return config, args.worker_gpu


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, worker_gpu = config_from_args(args, parser)
    try:
        if worker_gpu is not None:
            return run_worker(config, worker_gpu)
        return run_coordinator(config)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        log("guard failed", error=f"{type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
