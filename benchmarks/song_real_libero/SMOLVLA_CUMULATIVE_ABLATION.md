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
fixed policy seed 0 and environment seed 7, with the first 10 official initial
states per task under strict official initialization (20 episodes total) for
every wall-clock hourly checkpoint.

## Execution

The default resource allocation keeps evaluation from competing for training
VRAM:

- GPUs 0-3: one training run per variant;
- GPUs 4-7: one checkpoint-evaluation watcher per variant;
- two DataLoader workers per training run;
- two tasks by two rollout workers per evaluation run.

The runner admits each trainer and each evaluation only after three consecutive
resource samples pass. Defaults reserve headroom below the requested host
limits: 50 GiB / 36 CPU cores / 400 MiB/s are soft admission thresholds, while
58 GiB / 42 cores / 768 MiB/s / 23 GiB per GPU are audited as hard limits.
Trainer startup is staged until each preceding process has allocated its CUDA
model, preventing four simultaneous VLM weight reads; all four then train
concurrently.
While any trainer is resident, full evaluations share one local lock and run
serially to preserve host-memory headroom. After all four trainers exit, the
waiting variant watchers use one bounded evaluation slot on GPUs 4--7. Variant
starts are staggered by 60 seconds so model loads do not
hit NFS together. Each model directory is first staged under `/tmp` with rsync
limited to 100 MiB/s, together with the pretrained SmolVLA weights and the
SmolVLM tokenizer/config assets; the staged policy config is rewritten to those
local paths. After evaluator startup, clean staged-file pages are released from
the page cache while the files remain available until evaluation exits. Each
watcher still admits work through
the same soft resource guard, and the global watcher terminates evaluations at
57 GiB before the 58 GiB hard memory limit.
Every sample, including all eight GPUs, is appended to
`resource/samples.jsonl`; a persistent hard-limit marker is written under the
same directory if a threshold is crossed, and it blocks new work. The 2D baseline
does not wrap the point-cloud dataset, which removes unnecessary shared-storage
reads and makes the modality ablation exact.
The admission IO rate is the maximum of cgroup block-device bytes and NFS
server-transfer bytes. Per-process `read_bytes + write_bytes` remains a
diagnostic field only because short-lived `rsync` children can transfer their
cumulative accounting to a parent and falsely multiply-count a completed copy.

Run the complete experiment from the repository root:

```bash
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh preflight
bash benchmarks/song_real_libero/scripts/run_v043_cumulative_ablation.sh all
```

The default maximum is 30,000 steps. Checkpoints are recorded every 3,600
seconds of training wall time and the dedicated evaluation GPU consumes each
completed checkpoint in order. This upper bound is intentionally longer than a
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

`ablation_results.csv` contains every checkpoint result. `ABLATION_RESULTS.md`
reports the latest and best completed checkpoints plus adjacent cumulative
deltas. `stability.json` marks a variant stable only after step 8,000 when its
latest three complete 20-episode success rates span no more than three
percentage points. All four variants must pass that rule before the experiment
is considered stable. Best-checkpoint numbers are descriptive only; module
claims use persistence across the full curve and the latest stable results.
