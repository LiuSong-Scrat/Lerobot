from benchmarks.song_real_libero.scripts.ablation_resource_guard import (
    descendant_pids,
    storage_io_bytes_from_text,
)


def test_descendant_pids_selects_only_requested_trees() -> None:
    parent_by_pid = {
        11: 1,
        12: 11,
        13: 12,
        21: 1,
        22: 21,
        30: 2,
    }

    assert descendant_pids({11, 21}, parent_by_pid) == {11, 12, 13, 21, 22}


def test_storage_io_excludes_cached_logical_io() -> None:
    raw = """rchar: 30000000000
wchar: 20000000000
syscr: 10
syscw: 20
read_bytes: 1048576
write_bytes: 2097152
cancelled_write_bytes: 0
"""

    assert storage_io_bytes_from_text(raw) == 3 * 1024 * 1024
