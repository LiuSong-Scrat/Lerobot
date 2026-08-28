from benchmarks.song_real_libero.scripts.ablation_resource_guard import descendant_pids


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
