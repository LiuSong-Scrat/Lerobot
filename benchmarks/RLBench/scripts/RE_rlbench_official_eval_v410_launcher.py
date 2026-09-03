#!/home/liusong/miniconda3/envs/rlbench/bin/python
"""Run the existing RLBench evaluator with CoppeliaSim 4.10 scene loading."""

from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
RL_BENCH_ROOT = SCRIPT_PATH.parents[1]
V410_PYREP_ROOT = RL_BENCH_ROOT / "pyrep_v410"
# Load the ABI-matched PyRep before the environment-wide PyRep installation.
sys.path.insert(0, str(V410_PYREP_ROOT))
sys.path.insert(0, str(RL_BENCH_ROOT))


def patch_v410_render_mode() -> None:
    """Map RLBench's default OpenGL3 sensors to a selected 4.10 mode."""
    requested = os.environ.get("EVAL_V410_RENDER_MODE", "opengl3").lower()
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    render_modes = {
        "opengl": RenderMode.OPENGL,
        "opengl3": RenderMode.OPENGL3,
        "opengl3_windowed": RenderMode.OPENGL3_WINDOWED,
    }
    if requested not in render_modes:
        raise SystemExit(
            "EVAL_V410_RENDER_MODE must be opengl, opengl3, or opengl3_windowed"
        )
    selected = render_modes[requested]
    if getattr(VisionSensor.set_render_mode, "_rlbench_v410_patched", False):
        return

    original_set_render_mode = VisionSensor.set_render_mode

    def set_render_mode(self, render_mode):
        if render_mode == RenderMode.OPENGL3:
            render_mode = selected
        return original_set_render_mode(self, render_mode)

    set_render_mode._rlbench_v410_patched = True
    VisionSensor.set_render_mode = set_render_mode


def patch_pyrep_scene_loading() -> None:
    """Load old RLBench scenes explicitly after the 4.10 client starts."""
    from pyrep import PyRep
    from pyrep.backend import sim

    if getattr(PyRep.launch, "_rlbench_v410_patched", False):
        return

    original_launch = PyRep.launch

    def launch_with_explicit_scene(self, scene_file="", *args, **kwargs):
        requested_scene = os.path.abspath(scene_file) if scene_file else ""
        # CoppeliaSim 4.10's OpenGL3 vision renderer needs a live Qt/GLX
        # context. Its emulated headless UI has no usable QGLWidget and
        # crashes on the first camera render, so v410 keeps the hidden GUI
        # context while legacy remains headless.
        if os.environ.get("EVAL_RLBENCH_HEADLESS", "0").lower() in {
            "0", "false", "no", "off"
        }:
            kwargs["headless"] = False
        # CoppeliaSim 4.10 accepts the old client launch API but does not load
        # the scene passed through that API. Start the client first, then use
        # the legacy simLoadScene call exposed by PyRep.
        original_launch(self, "", *args, **kwargs)
        if requested_scene:
            sim.simLoadScene(requested_scene)
            # CoppeliaSim 4.10 updates the client asynchronously after an
            # explicit legacy scene load. Calling step_ui() here can release
            # stale scene data and crash with an allocator error.
            time.sleep(0.5)

    launch_with_explicit_scene._rlbench_v410_patched = True
    PyRep.launch = launch_with_explicit_scene


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: launcher.py EVAL_SCRIPT [EVAL_ARGS ...]")
    patch_v410_render_mode()
    patch_pyrep_scene_loading()
    eval_script = Path(sys.argv[1]).resolve()
    sys.argv = sys.argv[1:]
    sys.argv[0] = str(eval_script)
    runpy.run_path(str(eval_script), run_name="__main__")


if __name__ == "__main__":
    main()
