"""Modal launcher for the existing WEP-VLA training entry point.

The regular server command remains unchanged and does not import Modal::

    python benchmarks/song_real_libero/scripts/train_song_benchmark.py ...

Install the separate client dependency from ``requirements-modal.txt`` before
using this launcher. Every argument after ``modal_runner.py`` is forwarded to
the original training script without changing its name or syntax::

    modal run modal_runner.py --policy.type=smolvla ...

Modal SDK 1.5+ deliberately supports a variadic local entrypoint, so dotted
Draccus flags such as ``--dataset.repo_id`` pass through unchanged.
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import modal


APP_NAME = "wep-vla-training"
PROJECT_ROOT = Path(__file__).resolve().parent
CONTAINER_PROJECT_ROOT = Path("/lerobot")
TRAIN_ENTRYPOINT = CONTAINER_PROJECT_ROOT / "benchmarks/song_real_libero/scripts/train_song_benchmark.py"

# Use a Modal-only ignore file with no negated rules. That lets the SDK prune
# unrelated multi-gigabyte benchmark trees before hashing the build context,
# without changing the project's existing Docker behavior.
MODAL_CONTEXT_IGNORE = modal.FilePatternMatcher.from_file(PROJECT_ROOT / "docker/modal.dockerignore")

def _stage_modal_context() -> tuple[tempfile.TemporaryDirectory, Path]:
    """Stage a small, automatically cleaned Docker build context.

    The shared server checkout contains hundreds of gigabytes of unrelated
    benchmark assets. Modal's Dockerfile COPY filter prevents uploading those
    files but still walks the entire tree. Copying the already-filtered source
    to a temporary directory avoids that scan, and Python removes it when the
    local Modal command exits.
    """

    temporary_dir = tempfile.TemporaryDirectory(prefix="wep-vla-modal-context-")
    context_root = Path(temporary_dir.name)
    resolved_project_root = PROJECT_ROOT.resolve()

    def ignored_names(directory: str, names: list[str]) -> set[str]:
        relative_dir = Path(directory).resolve().relative_to(resolved_project_root)
        return {
            name
            for name in names
            if MODAL_CONTEXT_IGNORE(relative_dir / name)
        }

    shutil.copytree(
        PROJECT_ROOT,
        context_root,
        dirs_exist_ok=True,
        ignore=ignored_names,
        symlinks=True,
    )
    return temporary_dir, context_root


_MODAL_CONTEXT_TEMP, MODAL_CONTEXT_ROOT = _stage_modal_context()
# The project uses CUDA extensions (FlashAttention, PointOps, PointROPE,
# SpConv, and Torch Scatter), so the Modal-specific Dockerfile uses a CUDA
# development image rather than the project's CPU-oriented user image.
TRAIN_IMAGE = modal.Image.from_dockerfile(
    PROJECT_ROOT / "docker/Dockerfile.modal",
    context_dir=PROJECT_ROOT,
    ignore=MODAL_CONTEXT_IGNORE,
)

# The home mount intentionally matches the existing Linux server paths. Data,
# model weights, PointSeg caches, checkpoints, outputs, and logs can therefore
# use the exact same CLI values on the server and on Modal.
PROJECT_VOLUME = modal.Volume.from_name("wep-vla-home", create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name("wep-vla-cache", create_if_missing=True)

# Static analysis found Hugging Face and Weights & Biases integrations. Put
# whichever keys a run needs (HF_TOKEN and/or WANDB_API_KEY) in this one Modal
# Secret. Neither value is stored in source code or baked into the image.
RUNTIME_SECRET = modal.Secret.from_name("wep-vla-secrets")

app = modal.App(APP_NAME)


def _has_option(args: tuple[str, ...], option: str) -> bool:
    """Return whether a Draccus option is present in either supported form."""

    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in args)


def _is_resume(args: tuple[str, ...]) -> bool:
    """Return whether the original CLI requests checkpoint resume."""

    for arg in args:
        if arg == "--resume":
            return True
        if arg.startswith("--resume="):
            return arg.partition("=")[2].strip().lower() in {"1", "true", "yes", "on"}
    return False


def _with_persistent_default_output(args: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the original output argument, or supply a persistent Modal default.

    The native training config normally creates a relative ``outputs/train``
    directory. That is correct on the server but ephemeral inside Modal. Only
    when the caller did not choose an output and is not resuming do we add a
    timestamped directory on the mounted project Volume.
    """

    if _has_option(args, "--output_dir") or _is_resume(args):
        return args

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = Path("/home/liusong/modal_outputs/train") / timestamp
    return (*args, f"--output_dir={output_dir}")


@app.function(
    image=TRAIN_IMAGE,
    gpu="A100-80GB",
    cpu=16.0,
    memory=65_536,
    timeout=24 * 60 * 60,
    max_containers=1,
    secrets=[RUNTIME_SECRET],
    volumes={
        "/home/liusong": PROJECT_VOLUME,
        "/cache": HF_CACHE_VOLUME,
    },
    env={
        "HOME": "/home/liusong",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        "HF_LEROBOT_HOME": "/cache/huggingface/lerobot",
        "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
        "TORCH_HOME": "/cache/torch",
        "TRITON_CACHE_DIR": "/cache/triton",
        "XDG_CACHE_HOME": "/cache/xdg",
        "WANDB_CACHE_DIR": "/cache/wandb",
        "WANDB_DIR": "/home/liusong/modal_outputs/wandb",
        "MUJOCO_GL": "egl",
        "SONG_POINTSEG_REQUIRE_POINTOPS": "1",
    },
)
def run_training(*args: str) -> None:
    """Run the existing training script and persist all mounted changes."""

    forwarded_args = _with_persistent_default_output(tuple(args))
    command = [sys.executable, str(TRAIN_ENTRYPOINT), *forwarded_args]

    try:
        subprocess.run(command, cwd=CONTAINER_PROJECT_ROOT, check=True)
    finally:
        # Successful Modal functions are committed automatically, while these
        # explicit commits also make the lifecycle and failure behavior clear.
        PROJECT_VOLUME.commit()
        HF_CACHE_VOLUME.commit()


@app.local_entrypoint()
def main(*args: str) -> None:
    """Forward the original training CLI verbatim to the remote GPU function."""

    run_training.remote(*args)
