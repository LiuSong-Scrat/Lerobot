from __future__ import annotations

import json
import sys

import pytest

from benchmarks.song_real_libero.scripts import summarize_v043_ablation


def _write_summary(
    root, variant: str, step: int, task6_successes: int, task8_successes: int
) -> None:
    results = []
    for task_id, successes in ((6, task6_successes), (8, task8_successes)):
        results.append(
            {
                "suite": "libero_10",
                "task_id": task_id,
                "episodes": [{"success": episode < successes} for episode in range(50)],
            }
        )
    output = root / "eval" / variant / f"step{step:06d}"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({"results": results}))


def test_summary_marks_all_variants_stable_from_three_complete_100_episode_runs(
    tmp_path, monkeypatch
):
    for variant in summarize_v043_ablation.VARIANTS:
        _write_summary(tmp_path, variant, 8000, 40, 42)
        _write_summary(tmp_path, variant, 9000, 41, 42)
        _write_summary(tmp_path, variant, 10000, 41, 43)
    monkeypatch.setattr(sys, "argv", ["summarize", "--root", str(tmp_path)])

    summarize_v043_ablation.main()

    stability = json.loads((tmp_path / "stability.json").read_text())
    assert stability["all_stable"]
    assert all(item["stable"] for item in stability["variants"].values())
    assert (tmp_path / "ablation_results.csv").is_file()
    assert (tmp_path / "ABLATION_RESULTS.md").is_file()


def test_summary_rejects_shortened_test_protocol(tmp_path, monkeypatch):
    _write_summary(tmp_path, summarize_v043_ablation.VARIANTS[0], 8000, 40, 42)
    summary_path = next(tmp_path.rglob("summary.json"))
    summary = json.loads(summary_path.read_text())
    summary["results"][0]["episodes"].pop()
    summary_path.write_text(json.dumps(summary))
    monkeypatch.setattr(sys, "argv", ["summarize", "--root", str(tmp_path)])

    with pytest.raises(RuntimeError, match="Expected 50 task 6 episodes"):
        summarize_v043_ablation.main()
