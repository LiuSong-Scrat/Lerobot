#!/usr/bin/env bash
set -uo pipefail

# CoppeliaSim 4.10 compatibility entrypoint. All evaluation parameters and
# task execution remain in the maintained s4 script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V410_ROOT="${COPPELIASIM_V410_ROOT:-/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/CoppeliaSim_V4_10_0}"
V410_PYTHON="${PYTHON:-/home/liusong/miniconda3/envs/rlbench/bin/python}"
V410_LAUNCHER="${SCRIPT_DIR}/RE_rlbench_official_eval_v410_launcher.py"
V410_PYTHON_DIR="$(dirname "${V410_PYTHON}")"

if [[ ! -x "${V410_ROOT}/coppeliaSim" ]]; then
    echo "CoppeliaSim 4.10 executable not found: ${V410_ROOT}/coppeliaSim" >&2
    exit 2
fi

if [[ ! -f "${V410_LAUNCHER}" ]]; then
    echo "CoppeliaSim 4.10 launcher not found: ${V410_LAUNCHER}" >&2
    exit 2
fi

export COPPELIASIM_ROOT="${V410_ROOT}"
export COPPELIASIM_V410_ROOT="${V410_ROOT}"
export PYTHON="${V410_PYTHON}"
export EVAL_PYTHON_LAUNCHER="${V410_LAUNCHER}"
# Keep the maintained s4 entrypoint's metadata and version dispatch aligned
# with this wrapper. Without this, s4 sees the injected root/launcher but
# records the run as legacy in eval_config.json and task logs.
export EVAL_COPPELIASIM_VERSION="v410"
export EVAL_V410_RENDER_MODE="${EVAL_V410_RENDER_MODE:-opengl3}"
export LD_LIBRARY_PATH="${V410_ROOT}:${LD_LIBRARY_PATH:-}"
# The 4.10 distribution ships its own Qt platform plugins.  The maintained
# s4 script forwards this variable to the evaluator subprocess.
export QT_QPA_PLATFORM_PLUGIN_PATH="${V410_ROOT}/platforms"
export QT_PLUGIN_PATH="${V410_ROOT}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
# CoppeliaSim 4.10 starts its Python wrapper as `python3`. Ensure that this
# resolves to the same rlbench environment that contains pyzmq and cbor2.
export PATH="${V410_PYTHON_DIR}:${PATH}"

# Keep the maintained s4 Python for configuration and summaries. Only the
# actual evaluator subprocess goes through the CoppeliaSim 4.10 adapter.
exec bash "${SCRIPT_DIR}/s4_RE_rlbench_official_eval.sh" "$@"
