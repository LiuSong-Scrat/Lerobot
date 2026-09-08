#!/usr/bin/env python3
"""Run RLBench evaluation with a process-local REAP Z offset of -0.09 m.

This wrapper deliberately leaves the canonical source defaults untouched.  It
patches the already-imported geometry module before loading the evaluator, so
both point generation and recorded metadata use the requested experiment-only
offset.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rlbench_reap_gripper as reap  # noqa: E402


REAP_OFFSET_M = 0.09
reap.LIBERO_REAP_GRIPPER_LEN = REAP_OFFSET_M
reap.LIBERO_REAP_LOCAL_OFFSET_M = (0.0, 0.0, -REAP_OFFSET_M)

metadata = reap.canonical_reap_metadata()
assert metadata["virtual_gripper_local_offset_m"] == [0.0, 0.0, -0.09]
print(
    "[experiment-reap-offset] process_local=true local_offset_m=[0.0,0.0,-0.09]",
    flush=True,
)

eval_script = SCRIPT_DIR / "RE_rlbench_official_eval.py"
sys.argv[0] = str(eval_script)
runpy.run_path(str(eval_script), run_name="__main__")
