"""Shared, collision-free path setup for standalone RLBench diagnostic tools."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TOOLS_DIR.parent
RLBENCH_ROOT = SCRIPTS_DIR.parent
LEROBOT_ROOT = RLBENCH_ROOT.parents[1]
SONG_SCRIPTS_DIR = LEROBOT_ROOT / "benchmarks" / "song_real_libero" / "scripts"

for search_path in (SCRIPTS_DIR, LEROBOT_ROOT / "src", SONG_SCRIPTS_DIR):
    value = str(search_path)
    if value not in sys.path:
        sys.path.insert(0, value)
