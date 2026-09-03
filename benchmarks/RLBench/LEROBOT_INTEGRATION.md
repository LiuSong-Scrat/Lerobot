# RLBench integration for LeRobot v0.5.0

This directory vendors the RLBench source and simulator scene assets required
by the LeRobot collection, cache, training, action-replay, and online evaluation
pipelines developed in this repository.

## Baseline

- LeRobot base: `origin/wep_vla_v0.4.3_multiview_doubleflow`
- RLBench upstream baseline: `stepjam/RLBench` commit `02720bba`
- Integrated branch: `wep_vla_v0.5.0`

RLBench is stored as an ordinary directory in the LeRobot repository. Its
original nested `.git` directory is intentionally not included.

## Included

- The `rlbench` Python package and its task, robot, and scene assets.
- The maintained collection, conversion, cache, replay, evaluation, and
  diagnostic scripts under `scripts/`.
- The optional local PyRep 4.10 compatibility source under `pyrep_v410/`.
- Source-only examples, tests, documentation, and package metadata.
- LeRobot compatibility updates for the `front` camera and RLBench REAP/Panda
  virtual-gripper geometry.

## Excluded generated content

The following local content is deliberately excluded from version control:

- `datasets/`
- `eval/`
- `outputs/` and checkpoints
- point-segmentation caches
- CoppeliaSim installations and binaries
- logs, videos, PLY files, NumPy artifacts, Python bytecode, and backup files

These exclusions are enforced by this directory's `.gitignore` in addition to
the repository-level ignore rules.

## External runtime requirements

Install RLBench from this checkout with:

```bash
python -m pip install -e benchmarks/RLBench
```

CoppeliaSim is an external runtime dependency and is not redistributable as
part of this source branch. Set `COPPELIASIM_ROOT` and `LD_LIBRARY_PATH` to a
compatible local installation before collection or online evaluation. Dataset,
cache, and checkpoint locations must likewise be supplied at runtime.

The main online evaluator is:

```text
benchmarks/RLBench/scripts/RE_rlbench_official_eval.py
```

The collection and cache entry points are documented in
`scripts/EVAL_DEFAULTS.md`, `RLBENCH_TEST_SONG.md`, and the `s1_*` scripts.
