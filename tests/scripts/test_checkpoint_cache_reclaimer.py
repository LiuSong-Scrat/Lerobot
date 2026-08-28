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
