from benchmarks.song_real_libero.scripts.ablation_resource_guard import (
    cgroup_block_io_bytes_from_text,
    descendant_pids,
    nfs_server_io_bytes_from_text,
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


def test_cgroup_block_io_uses_busiest_stacked_device() -> None:
    raw = """8:0 Read 100
8:0 Write 300
8:0 Total 400
253:0 Read 100
253:0 Write 300
253:0 Total 400
Total 800
"""

    assert cgroup_block_io_bytes_from_text(raw) == 400


def test_nfs_server_io_selects_data_mount() -> None:
    raw = """device server:/other mounted on /other with fstype nfs statvers=1.1
\tbytes:\t1 2 3 4 500 600 7 8
device server:/private mounted on /opt/data/private with fstype nfs statvers=1.1
\tbytes:\t10 20 30 40 5000 6000 70 80
"""

    assert nfs_server_io_bytes_from_text(raw) == 11_000
