# RLBench diagnostic tools

This directory contains standalone inspection, loss-analysis, PLY export, and
visualization utilities. They are not part of the collection, cache, training,
or unified evaluation execution chain in the parent `scripts/` directory.

Run a tool by its new path, for example:

```bash
python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/tools/RE_rlbench_per_task_loss.py --help
```

`_rlbench_tool_paths.py` gives relocated tools stable access to the parent RLBench scripts,
the LeRobot source tree, and the Song benchmark helpers. It is an internal
helper rather than a user-facing command.

The tools are grouped by filename:

- `RE_rlbench_*diagnostic.py`, `s4.1_*`, `s4.3_*`: policy/controller diagnostics.
- `RE_rlbench_*loss.py`, `RE_rlbench_summarize_*`: loss and audit reporting.
- `RE_rlbench_export_*`, `export_*`, `RE_rlbench_filter_*`: table and PLY export.
- `render_*`, `visualize_*`: offline or simulator visualization.
- `rename_eval_roots_with_checkpoint.py`: historical evaluation-output maintenance.
