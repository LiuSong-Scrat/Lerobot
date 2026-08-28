# SmolVLA cumulative ablation protocol

This protocol compares four cumulative architectures from the same fresh
SmolVLA construction and the same SmolVLM2-500M weight source. It does not warm
start any variant from a trained WEP-VLA policy checkpoint.

| `policy.ablation_variant` | RGB | 10k XYZRGB | EffSeg | PointAction |
|---|---:|---:|---:|---:|
| `smolvla_src` | yes | no | no | no |
| `smolvla_pointcloud` | yes | yes, whole-cloud LitePT token | no | no |
| `smolvla_pointcloud_effseg` | yes | yes | foreground/background tokens | no |
| `smolvla_pointcloud_effseg_pointaction` | yes | yes | foreground/background tokens | foreground local tokens to actions |

Named ablation variants force the gates in the table. All four retain the
official RGB path and freeze the same SmolVLM backbone. WorldFlow is disabled
because it is not one of the three cumulative modules being measured here.

For every point-enabled variant, `policy.pointcloud_input_points` is forced to
10,000. Training point clouds and aligned EffSeg labels are deterministically
truncated or cyclically completed together immediately before the model. The
rollout evaluator is explicitly launched with `--num-points 10000`. Thus the
three 3D variants receive the same number of scene-point slots in both training
and evaluation. The 2D source baseline never reads the point cloud.

The fixed dataset is:

```text
/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/world_eef_task6_task8_100ep
```

It contains 50 demonstrations each for LIBERO-10 tasks 6 and 8. Evaluation uses
fixed policy seed 0 and environment seed 7, with 10 episodes per task at every
2,000-step checkpoint.

## Execution

The default resource allocation keeps evaluation from competing for training
VRAM:

- GPUs 0-3: one training run per variant;
- GPUs 4-7: one checkpoint-evaluation watcher per variant;
- four DataLoader workers per training run;
- two tasks by two rollout workers per evaluation run.

Run the complete experiment from the repository root:

```bash
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh preflight
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh all
```

The default maximum is 30,000 steps. Checkpoints and fixed-seed rollouts are
recorded every 2,000 steps. This upper bound is intentionally longer than a
single pass over the 33,450-frame dataset; the checkpoint curves determine the
actual convergence plateau.

Useful commands:

```bash
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh status
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh summarize
```

Results are written below:

```text
/opt/data/private/liusong/benchmarks/song_real_libero/outputs/v043_cumulative_ablation_task6_task8_20260829
```

`ablation_results.csv` contains every checkpoint result. `ABlation_RESULTS.md`
reports the best completed checkpoint and adjacent cumulative deltas. Module
claims must also inspect persistence across the full curve, because selecting a
single best result from only 20 rollouts is noisy.
