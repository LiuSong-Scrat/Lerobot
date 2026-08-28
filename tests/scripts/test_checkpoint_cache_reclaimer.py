from pathlib import Path

from benchmarks.song_real_libero.scripts import checkpoint_cache_reclaimer


def test_sweep_ignores_incomplete_and_last_symlink(tmp_path: Path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "variant" / "checkpoints"
    ready_model = checkpoint_root / "000100" / "pretrained_model"
    ready_model.mkdir(parents=True)
    for name in checkpoint_cache_reclaimer.REQUIRED_MODEL_FILES:
        (ready_model / name).write_bytes(b"checkpoint")

    incomplete_model = checkpoint_root / "000200" / "pretrained_model"
    incomplete_model.mkdir(parents=True)
    (incomplete_model / "model.safetensors").write_bytes(b"incomplete")
    (checkpoint_root / "last").symlink_to("000100", target_is_directory=True)

    advised = []
    monkeypatch.setattr(
        checkpoint_cache_reclaimer.os,
        "posix_fadvise",
        lambda fd, offset, length, advice: advised.append((offset, length, advice)),
    )

    payload = checkpoint_cache_reclaimer.sweep(tmp_path)

    assert payload["checkpoints"] == 1
    assert payload["files"] == len(checkpoint_cache_reclaimer.REQUIRED_MODEL_FILES)
    assert len(advised) == len(checkpoint_cache_reclaimer.REQUIRED_MODEL_FILES)


def test_sweep_due_only_readvises_after_interval(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "variant" / "checkpoints" / "000100"
    model = checkpoint / "pretrained_model"
    model.mkdir(parents=True)
    for name in checkpoint_cache_reclaimer.REQUIRED_MODEL_FILES:
        (model / name).write_bytes(b"checkpoint")
    calls = []
    monkeypatch.setattr(
        checkpoint_cache_reclaimer,
        "release_checkpoint_cache",
        lambda path: (calls.append(path) or (4, 40)),
    )
    last_advised = {}

    first = checkpoint_cache_reclaimer.sweep_due(
        tmp_path, last_advised, now=10.0, readvise_seconds=60.0
    )
    early = checkpoint_cache_reclaimer.sweep_due(
        tmp_path, last_advised, now=69.0, readvise_seconds=60.0
    )
    due = checkpoint_cache_reclaimer.sweep_due(
        tmp_path, last_advised, now=70.0, readvise_seconds=60.0
    )

    assert first["checkpoints_advised"] == 1
    assert early["checkpoints_advised"] == 0
    assert due["checkpoints_advised"] == 1
    assert calls == [checkpoint, checkpoint]
