#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPERIMENT_ROOT="$(realpath -e "${SCRIPT_DIR}/../../..")"
PYTHON_BIN="${GPU_GUARD_PYTHON_BIN:-${EXPERIMENT_ROOT}/.venv-smol5090/bin/python}"
GUARD_SCRIPT="${SCRIPT_DIR}/gpu_idle_memory_guard.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "GPU guard Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${GUARD_SCRIPT}" ]]; then
  echo "GPU guard implementation is missing: ${GUARD_SCRIPT}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU guard requires nvidia-smi." >&2
  exit 2
fi

umask 077
exec "${PYTHON_BIN}" "${GUARD_SCRIPT}" "$@"
