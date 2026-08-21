#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
env_dir="${1:-$repo_root/.venv-smol5090}"
conda_bin="${CONDA_BIN:-/home/liusong/anaconda3/bin/conda}"
max_jobs="${MOLMO2_MAX_JOBS:-8}"
torch_cuda_arch_list="${MOLMO2_TORCH_CUDA_ARCH_LIST:-8.0;12.0}"
flash_cuda_archs="${MOLMO2_FLASH_CUDA_ARCHS:-80;120}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This lock is for Linux x86_64 only." >&2
    exit 2
fi

if [[ ! -x "$conda_bin" ]]; then
    echo "Conda executable not found: $conda_bin" >&2
    exit 2
fi

if [[ -e "$env_dir" ]]; then
    echo "Refusing to overwrite existing environment: $env_dir" >&2
    echo "Move or remove that exact directory, then rerun this script." >&2
    exit 2
fi

"$conda_bin" create \
    --yes \
    --prefix "$env_dir" \
    --file "$repo_root/environment-molmo2-conda.lock.txt"

env_python="$env_dir/bin/python"
"$env_python" -m ensurepip --upgrade
"$env_python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --requirement "$repo_root/requirements-molmo2-lock.txt"

cuda_root="$env_dir/targets/x86_64-linux"
nvidia_root="$env_dir/lib/python3.11/site-packages/nvidia"
include_paths="$cuda_root/include:$nvidia_root/cusparse/include:$nvidia_root/cublas/include:$nvidia_root/cusolver/include"
library_paths="$cuda_root/lib:$nvidia_root/cusparse/lib:$nvidia_root/cublas/lib:$nvidia_root/cusolver/lib"

export PATH="$env_dir/bin:$env_dir/nvvm/bin:$PATH"
export CUDA_HOME="$cuda_root"
export CPATH="$include_paths${CPATH:+:$CPATH}"
export LIBRARY_PATH="$library_paths${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC="$env_dir/bin/x86_64-conda-linux-gnu-cc"
export CXX="$env_dir/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST="$torch_cuda_arch_list"
export MAX_JOBS="$max_jobs"

# The public wheels for these two packages require a newer glibc than this
# machine provides, so build the same pinned versions against the local sysroot.
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS="$flash_cuda_archs" \
    "$env_python" -m pip install \
        --no-build-isolation \
        --no-cache-dir \
        --no-binary=flash-attn \
        --no-deps \
        flash-attn==2.8.3

FORCE_CUDA=1 \
    "$env_python" -m pip install \
        --no-build-isolation \
        --no-cache-dir \
        --no-binary=torch-scatter \
        --no-deps \
        torch-scatter==2.1.2

"$env_python" -m pip install \
    --no-build-isolation \
    --no-deps \
    "$repo_root/src/lerobot/policies/smolvla/litept/libs/pointops"

"$env_python" -m pip install --no-deps --editable "$repo_root"
"$env_python" -m pip check

echo "Molmo2 environment ready: $env_dir"
echo "Activate with: conda activate $env_dir"
