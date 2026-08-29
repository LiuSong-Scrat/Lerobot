#!/usr/bin/env python
"""Resource admission and audit guard for the four-way WEPVLA ablation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
NFS_DATA_MOUNT = "/opt/data/private"


@dataclass(frozen=True)
class Limits:
    memory_soft_gib: float = 50.0
    memory_hard_gib: float = 58.0
    cpu_soft_cores: float = 36.0
    cpu_hard_cores: float = 42.0
    io_soft_mib_s: float = 400.0
    io_hard_mib_s: float = 768.0
    gpu_soft_used_gib: float = 2.0
    gpu_hard_used_gib: float = 23.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--memory-soft-gib", type=float, default=50.0)
    parser.add_argument("--memory-hard-gib", type=float, default=58.0)
    parser.add_argument("--cpu-soft-cores", type=float, default=36.0)
    parser.add_argument("--cpu-hard-cores", type=float, default=42.0)
    parser.add_argument("--io-soft-mib-s", type=float, default=400.0)
    parser.add_argument("--io-hard-mib-s", type=float, default=768.0)
    parser.add_argument("--gpu-soft-used-gib", type=float, default=2.0)
    parser.add_argument("--gpu-hard-used-gib", type=float, default=23.0)
    parser.add_argument("--terminate-eval-memory-gib", type=float)
    parser.add_argument(
        "--recover-hard-marker-after-soft",
        action="store_true",
        help=(
            "In --wait mode, archive prior hard-limit markers only after the "
            "requested consecutive soft-safe samples."
        ),
    )
    return parser.parse_args()


def _proc_stat(pid: int) -> tuple[int, float, int] | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        rest = raw[raw.rfind(")") + 2 :].split()
        ppid = int(rest[1])
        cpu_s = (int(rest[11]) + int(rest[12])) / CLK_TCK
        rss = (
            int((Path("/proc") / str(pid) / "statm").read_text().split()[1]) * PAGE_SIZE
        )
        return ppid, cpu_s, rss
    except (
        FileNotFoundError,
        IndexError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        return None


def storage_io_bytes_from_text(raw: str) -> int:
    """Return storage-accounted bytes, excluding cached logical I/O."""
    values = {}
    for line in raw.splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values.get("read_bytes", 0) + values.get("write_bytes", 0)


def _proc_io_bytes(pid: int) -> int:
    try:
        return storage_io_bytes_from_text((Path("/proc") / str(pid) / "io").read_text())
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return 0


def cgroup_block_io_bytes_from_text(raw: str) -> int:
    """Return actual block bytes without double-counting stacked devices."""
    device_totals = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 3 and ":" in fields[0] and fields[1] == "Total":
            device_totals.append(int(fields[2]))
    return max(device_totals, default=0)


def _cgroup_block_io_bytes() -> int:
    candidates = (
        Path("/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes_recursive"),
        Path("/sys/fs/cgroup/blkio/blkio.throttle.io_service_bytes"),
    )
    for path in candidates:
        try:
            return cgroup_block_io_bytes_from_text(path.read_text())
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return 0


def nfs_server_io_bytes_from_text(raw: str, mountpoint: str = NFS_DATA_MOUNT) -> int:
    """Return bytes transferred to the NFS server for one mounted filesystem."""
    selected = False
    for line in raw.splitlines():
        if line.startswith("device "):
            selected = f" mounted on {mountpoint} with fstype nfs " in f"{line} "
            continue
        if selected and line.strip().startswith("bytes:"):
            values = [int(value) for value in line.split(":", 1)[1].split()]
            if len(values) < 6:
                raise ValueError(f"Unexpected NFS bytes line: {line!r}")
            return values[4] + values[5]
    return 0


def _nfs_server_io_bytes() -> int:
    try:
        return nfs_server_io_bytes_from_text(Path("/proc/self/mountstats").read_text())
    except (FileNotFoundError, PermissionError, ValueError):
        return 0


def system_io_snapshot() -> tuple[int, int]:
    return _cgroup_block_io_bytes(), _nfs_server_io_bytes()


def _root_pids(pid_dir: Path) -> set[int]:
    result = set()
    for path in pid_dir.glob("*.pid") if pid_dir.is_dir() else ():
        try:
            pid = int(path.read_text().strip())
            os.kill(pid, 0)
            result.add(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def process_tree_snapshot(pid_dir: Path) -> dict[int, tuple[float, int, int]]:
    stats = {}
    children: dict[int, set[int]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        stat = _proc_stat(pid)
        if stat is None:
            continue
        ppid, cpu_s, rss = stat
        stats[pid] = (cpu_s, rss, _proc_io_bytes(pid))
        children.setdefault(ppid, set()).add(pid)

    selected = set()
    pending = list(_root_pids(pid_dir))
    while pending:
        pid = pending.pop()
        if pid in selected:
            continue
        selected.add(pid)
        pending.extend(children.get(pid, ()))
    return {pid: stats[pid] for pid in selected if pid in stats}


def descendant_pids(root_pids: set[int], parent_by_pid: dict[int, int]) -> set[int]:
    children: dict[int, set[int]] = {}
    for pid, ppid in parent_by_pid.items():
        children.setdefault(ppid, set()).add(pid)
    selected = set()
    pending = list(root_pids)
    while pending:
        pid = pending.pop()
        if pid in selected:
            continue
        selected.add(pid)
        pending.extend(children.get(pid, ()))
    return selected


def terminate_eval_processes(pid_dir: Path) -> list[int]:
    roots = set()
    for path in pid_dir.glob("eval_*.pid") if pid_dir.is_dir() else ():
        try:
            roots.add(int(path.read_text().strip()))
        except (FileNotFoundError, ValueError):
            continue
    parent_by_pid = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        stat = _proc_stat(int(path.name))
        if stat is not None:
            parent_by_pid[int(path.name)] = stat[0]
    selected = descendant_pids(roots, parent_by_pid)
    terminated = []
    for pid in sorted(selected, reverse=True):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except (PermissionError, ProcessLookupError):
            continue
    return terminated


def cgroup_memory_bytes() -> int:
    candidates = (
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        Path("/sys/fs/cgroup/memory.current"),
    )
    for path in candidates:
        try:
            return int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            continue
    raise RuntimeError("Cannot read cgroup memory usage.")


def inactive_file_bytes_from_text(raw: str) -> int:
    values = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = int(fields[1])
    return values.get("total_inactive_file", values.get("inactive_file", 0))


def cgroup_memory_snapshot() -> tuple[int, int, int]:
    usage = cgroup_memory_bytes()
    candidates = (
        Path("/sys/fs/cgroup/memory/memory.stat"),
        Path("/sys/fs/cgroup/memory.stat"),
    )
    for path in candidates:
        try:
            inactive_file = inactive_file_bytes_from_text(path.read_text())
            return usage, inactive_file, max(0, usage - inactive_file)
        except (FileNotFoundError, ValueError):
            continue
    raise RuntimeError("Cannot read cgroup inactive-file memory.")


def cgroup_cpu_usage_seconds() -> float:
    usage_candidates = (
        Path("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"),
        Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"),
    )
    for path in usage_candidates:
        try:
            return int(path.read_text().strip()) / 1_000_000_000.0
        except (FileNotFoundError, ValueError):
            continue
    cpu_stat = Path("/sys/fs/cgroup/cpu.stat")
    try:
        values = {
            fields[0]: int(fields[1])
            for line in cpu_stat.read_text().splitlines()
            if len(fields := line.split()) == 2
        }
        return values["usage_usec"] / 1_000_000.0
    except (FileNotFoundError, KeyError, ValueError):
        raise RuntimeError("Cannot read cgroup CPU usage.") from None


def gpu_samples(gpu: int | None) -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if gpu is not None:
        command.insert(1, f"--id={gpu}")
    result = []
    for line in subprocess.check_output(command, text=True).strip().splitlines():
        output = [value.strip() for value in line.split(",")]
        if len(output) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output: {output!r}")
        result.append(
            {
                "index": int(output[0]),
                "used_gib": float(output[1]) / 1024.0,
                "total_gib": float(output[2]) / 1024.0,
                "utilization_percent": float(output[3]),
            }
        )
    return result


def sample(root: Path, gpu: int | None, seconds: float, limits: Limits) -> dict:
    pid_dir = root / "pids"
    start = process_tree_snapshot(pid_dir)
    block_start, nfs_start = system_io_snapshot()
    cgroup_cpu_start = cgroup_cpu_usage_seconds()
    start_time = time.monotonic()
    time.sleep(max(0.1, seconds))
    end = process_tree_snapshot(pid_dir)
    block_end, nfs_end = system_io_snapshot()
    cgroup_cpu_end = cgroup_cpu_usage_seconds()
    elapsed = max(time.monotonic() - start_time, 1e-6)
    shared = start.keys() & end.keys()
    experiment_cpu_cores = (
        sum(max(0.0, end[pid][0] - start[pid][0]) for pid in shared) / elapsed
    )
    cpu_cores = max(0.0, cgroup_cpu_end - cgroup_cpu_start) / elapsed
    process_io_mib_s = (
        sum(max(0, end[pid][2] - start[pid][2]) for pid in shared) / elapsed / MIB
    )
    block_io_mib_s = max(0, block_end - block_start) / elapsed / MIB
    nfs_io_mib_s = max(0, nfs_end - nfs_start) / elapsed / MIB
    io_mib_s = max(block_io_mib_s, nfs_io_mib_s)
    experiment_rss_gib = sum(item[1] for item in end.values()) / GIB
    cgroup_usage_bytes, inactive_file_bytes, working_set_bytes = cgroup_memory_snapshot()
    memory_gib = working_set_bytes / GIB
    gpu_info = gpu_samples(gpu)
    soft_reasons = []
    hard_reasons = []
    comparisons = (
        ("memory", memory_gib, limits.memory_soft_gib, limits.memory_hard_gib),
        ("cpu", cpu_cores, limits.cpu_soft_cores, limits.cpu_hard_cores),
        ("io", io_mib_s, limits.io_soft_mib_s, limits.io_hard_mib_s),
    )
    for name, value, soft, hard in comparisons:
        if value >= soft:
            soft_reasons.append(f"{name}={value:.2f}>={soft:.2f}")
        if value >= hard:
            hard_reasons.append(f"{name}={value:.2f}>={hard:.2f}")
    for item in gpu_info:
        index = int(item["index"])
        used_gib = float(item["used_gib"])
        if used_gib >= limits.gpu_soft_used_gib:
            soft_reasons.append(
                f"gpu{index}_used={used_gib:.2f}>={limits.gpu_soft_used_gib:.2f}"
            )
        if used_gib >= limits.gpu_hard_used_gib:
            hard_reasons.append(
                f"gpu{index}_used={used_gib:.2f}>={limits.gpu_hard_used_gib:.2f}"
            )
    return {
        "timestamp": time.time(),
        "root": str(root),
        "pids": sorted(end),
        "memory_gib": memory_gib,
        "memory_metric": "cgroup_working_set_usage_minus_inactive_file",
        "cgroup_usage_gib": cgroup_usage_bytes / GIB,
        "inactive_file_gib": inactive_file_bytes / GIB,
        "experiment_rss_gib": experiment_rss_gib,
        "cpu_cores": cpu_cores,
        "cpu_metric": "cgroup_cpuacct_usage",
        "experiment_cpu_cores": experiment_cpu_cores,
        "io_mib_s": io_mib_s,
        "io_metric": "max_cgroup_block_and_nfs_server_bytes",
        "block_io_mib_s": block_io_mib_s,
        "nfs_io_mib_s": nfs_io_mib_s,
        "process_io_mib_s": process_io_mib_s,
        "gpus": gpu_info,
        "soft_ok": not soft_reasons,
        "hard_ok": not hard_reasons,
        "soft_reasons": soft_reasons,
        "hard_reasons": hard_reasons,
        "limits": asdict(limits),
    }


def append_audit(root: Path, payload: dict) -> None:
    audit_dir = root / "resource"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    marker = audit_dir / "HARD_LIMIT_EXCEEDED"
    if not payload["hard_ok"]:
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def failed_sample_payload(root: Path, exc: BaseException, limits: Limits) -> dict:
    error = f"{type(exc).__name__}: {exc}"
    return {
        "timestamp": time.time(),
        "root": str(root),
        "pids": [],
        "memory_gib": None,
        "memory_metric": "cgroup_working_set_usage_minus_inactive_file",
        "cgroup_usage_gib": None,
        "inactive_file_gib": None,
        "experiment_rss_gib": None,
        "cpu_cores": None,
        "cpu_metric": "cgroup_cpuacct_usage",
        "experiment_cpu_cores": None,
        "io_mib_s": None,
        "io_metric": "max_cgroup_block_and_nfs_server_bytes",
        "block_io_mib_s": None,
        "nfs_io_mib_s": None,
        "process_io_mib_s": None,
        "gpus": [],
        "soft_ok": False,
        "hard_ok": False,
        "soft_reasons": [f"resource_sample_error={error}"],
        "hard_reasons": [f"resource_sample_error={error}"],
        "sample_error": error,
        "limits": asdict(limits),
    }


def archive_resource_limit_incident(root: Path, recovery_sample: dict) -> Path | None:
    resource_dir = root / "resource"
    marker_names = ("HARD_LIMIT_EXCEEDED", "EVAL_TERMINATED_RESOURCE_LIMIT")
    active_markers = [
        resource_dir / name
        for name in marker_names
        if (resource_dir / name).is_file()
    ]
    if not active_markers:
        return None

    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    incidents_root = resource_dir / "incidents"
    incidents_root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        suffix_text = "" if suffix == 0 else f"_{suffix}"
        incident_dir = incidents_root / f"{timestamp}_automatic_soft_recovery{suffix_text}"
        try:
            incident_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1

    archived = []
    for marker in active_markers:
        destination = incident_dir / f"{marker.name}.json"
        try:
            marker.replace(destination)
        except FileNotFoundError:
            continue
        archived.append(str(destination))
    (incident_dir / "incident.json").write_text(
        json.dumps(
            {
                "reason": "automatic recovery after consecutive soft-safe samples",
                "recovery_sample": recovery_sample,
                "archived_markers": archived,
                "timestamp": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return incident_dir


def main() -> int:
    args = parse_args()
    limits = Limits(
        memory_soft_gib=args.memory_soft_gib,
        memory_hard_gib=args.memory_hard_gib,
        cpu_soft_cores=args.cpu_soft_cores,
        cpu_hard_cores=args.cpu_hard_cores,
        io_soft_mib_s=args.io_soft_mib_s,
        io_hard_mib_s=args.io_hard_mib_s,
        gpu_soft_used_gib=args.gpu_soft_used_gib,
        gpu_hard_used_gib=args.gpu_hard_used_gib,
    )
    consecutive_ok = 0
    while True:
        marker = args.root / "resource" / "HARD_LIMIT_EXCEEDED"
        if (
            args.wait
            and marker.is_file()
            and not args.recover_hard_marker_after_soft
        ):
            print(
                f"Refusing new work after a hard resource limit: {marker}", flush=True
            )
            return 2
        try:
            payload = sample(args.root, args.gpu, args.sample_seconds, limits)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            payload = failed_sample_payload(args.root, exc, limits)
        append_audit(args.root, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if (
            args.terminate_eval_memory_gib is not None
            and payload["memory_gib"] is not None
            and payload["memory_gib"] >= args.terminate_eval_memory_gib
        ):
            terminated = terminate_eval_processes(args.root / "pids")
            emergency = {
                "timestamp": time.time(),
                "memory_gib": payload["memory_gib"],
                "threshold_gib": args.terminate_eval_memory_gib,
                "terminated_pids": terminated,
            }
            marker = args.root / "resource" / "EVAL_TERMINATED_RESOURCE_LIMIT"
            marker.write_text(json.dumps(emergency, indent=2), encoding="utf-8")
            print(json.dumps(emergency, sort_keys=True), flush=True)
        consecutive_ok = consecutive_ok + 1 if payload["soft_ok"] else 0
        if args.wait and consecutive_ok >= max(1, args.consecutive):
            if args.recover_hard_marker_after_soft:
                incident_dir = archive_resource_limit_incident(args.root, payload)
                if incident_dir is not None:
                    print(
                        f"Archived recovered resource-limit incident: {incident_dir}",
                        flush=True,
                    )
            return 0
        if not args.wait and not args.watch:
            return 0 if payload["soft_ok"] else 1
        time.sleep(max(0.1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
