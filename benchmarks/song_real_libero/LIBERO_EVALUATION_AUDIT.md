# LIBERO Evaluation Audit

This document records the evaluation protocol and the local evidence used to
compare WEP-VLA v0.4 checkpoints. It is intentionally separate from training
documentation: a low offline flow-matching loss and an online LIBERO success
rate measure different things.

## Main conclusions

1. The archived `wepvla_v04_20k_after_32k_after_3w_thresh_0.002` result is
   valid as an historical run: `libero_spatial = 99/100`, using the 20,000-step
   checkpoint and a `0.002 m` predicted-width delta threshold.
2. That run is not the current standard protocol. Its report records
   `control_freq=5 Hz`, `max_steps=1000`, `action_index=0`, and
   `exec_action_steps=12`. This allows up to about 200 seconds of simulated
   control time for a spatial task.
3. The standard protocol used here is `20 Hz` with suite horizons
   `220/280/300/520/400` for spatial/object/goal/long-horizon/90. Spatial thus
   has about 11 seconds. A 5 Hz / 1000-step score must not be compared directly
   with a 20 Hz / 220-step score.
4. With the same explicit 20k checkpoint and `0.002` gripper delta threshold,
   the standard serial spatial result was `86/100` for action index 0 / execute
   12, and `89/100` for action index 1 / execute 16.
5. Dynamic inference batching is not numerically equivalent to serial
   inference for this checkpoint. Even very small action differences can
   bifurcate contact-rich rollouts. Comparable scores must use actual inference
   batch size 1.
6. The original LitePT/spconv path also contains CUDA-level variability when
   sparse voxels contain duplicate coordinates. The model implementation has
   deliberately not been replaced with a different deterministic voxelization,
   because doing so changed the checkpoint's evaluation distribution and
   reduced success.
7. Independent model processes are supported as a throughput mode. They avoid
   cross-task batching and give each worker private model/RNG state, but their
   score is still one sample from a stochastic policy and may differ by several
   episodes from the serial run.

## Checkpoint identity

Do not use a mutable `checkpoints/last` path in a report intended for
comparison. Use an explicit checkpoint directory and retain the model hash.

| Checkpoint | Model SHA256 | Local archived suite report |
| --- | --- | --- |
| `016000_after32k_after32k` | `51dd421cea5c5cfe32e6eeb37494dcff81aea487702cabbd2b8d0b14082d4834` | no complete v0.4 report found |
| `020000_after32k_after32k` | `bdeeda177cf0860a6a14b09ca5188970d1f1f9717f564e211ff468339fc4ff43` | historical 20k report plus standard spatial audit |
| `024000_after32k_after32k` | `b32f9dd2bfa0f0627bb2c83990ee04b30b1787fd7f510c442322439b26c3db35` | historical 24k reports; no complete standard-protocol four-suite report |

The historical reports predate checkpoint hashing in the evaluator and often
store `checkpoints/last`. Their directory labels are useful evidence but do not
provide the same immutable identity as a modern report.

## Archived WEP-VLA v0.4 results

All rows below came from `song_real_libero/outputs/wepvla_v04*`. They use
5 Hz, action index 0, and execute 12 unless stated otherwise. A dash means the
suite was not completed, not a zero success rate.

| Archived run | Spatial | Object | LIBERO-10 | Goal |
| --- | ---: | ---: | ---: | ---: |
| `wepvla_v04_28k_after_3w` | 86/100 | 89/100 | — | — |
| `wepvla_v04_32k_after_3w_thresh_0.002` | 91/100 | 96/100 | — | — |
| `wepvla_v04_32k_after_3w_thresh_0.003` | 94/100 | 98/100 | 78/100 | 70/100 |
| `wepvla_v04_20k_after_32k_after_3w_thresh_0.002` | **99/100** | 89/100 | 53/80 | — |
| `wepvla_v04_24k_after_32k_after_3w_thresh_0.004` | 92/100 | 96/100 | 46/80 | — |
| `wepvla_v04_24k_after_32k_after_3w_thresh_0.003` | 89/100 | 95/100 | 36/50 | — |

For the historical 20k spatial run, successful episodes had median 123 steps;
three successful episodes finished after step 220 (283, 244, and 465). More
importantly, changing 5 Hz to 20 Hz changes how long every command is held, so
the standard result cannot be reconstructed merely by truncating the old log.

## Controlled standard-protocol experiments

These experiments explicitly used the current 20k model hash above, the
original checkpoint-compatible LitePT/PointSeg implementation, fixed policy
noise seed 0, environment seed 7, 20 Hz, the 220-step spatial horizon, and
gripper delta threshold 0.002.

| Mode | Action rows | Spatial | Per-task successes | Elapsed |
| --- | --- | ---: | --- | ---: |
| serial, batch 1 | index 0, execute 12 | 86/100 | `9,7,10,10,9,9,10,7,9,6` | 941.8 s |
| serial, batch 1 | index 1, execute 16 | **89/100** | `8,8,10,9,10,8,10,10,9,7` | 746.0 s |
| two independent models, batch 1 | index 0, execute 12 | 83/100 | `9,7,10,10,8,7,10,8,9,5` | 536.3 s |

Index 1 skips the near-identity first UMI row. Executing all 16 trained action
rows reduced model calls by about 31% and improved this controlled run by three
episodes. Earlier short-replan testing with execute 5 was substantially worse,
so non-blocking chunk execution is not the dominant failure source. The model
benefits from the temporal coherence of its trained chunk.

## Why low action loss does not imply nearly 100% success

`loss_action` is flow-matching velocity-field MSE averaged over sampled flow
time, action steps, and dimensions. Online inference then integrates the field,
converts the trajectory to controller targets, executes a chunk, observes a new
scene, and repeats. It does not directly optimize:

- final object placement tolerance;
- contact, grasp, or release success;
- task reward;
- robustness to flow noise or sparse-kernel numerical variation;
- accumulated closed-loop pose error;
- completion before a fixed benchmark horizon.

Consequently, an action loss around `3e-4` can coexist with a wrong subgoal,
one failed grasp, a few millimetres of terminal error, or a rollout that needs
more than the standard horizon. The rollout videos and action diagnostics show
both near-miss failures and genuine retry/subgoal loops; this is not explained
by one gripper-timing bug.

## Gripper control

LIBERO's Panda gripper input is directional (`-1=open`, `+1=close`), not a
physical width target. The default evaluator therefore uses the temporal
derivative of predicted physical width:

```text
delta_i = predicted_width[i + 1] - predicted_width[i]
delta_i > +0.002  -> -1 (open)
delta_i < -0.002  -> +1 (close)
otherwise         ->  0 (hold current actuator target)
```

`target_width` remains a diagnostic mode only. Closed-loop target tracking can
chatter when the measured gripper lags, contacts an object, overshoots, or the
next predicted chunk changes its target. There is no task-specific latch, rim
correction, or manually coded object rule in the standard path.

## Recommended commands

### Strict single-suite comparison

Use this mode to compare checkpoints. It is intentionally serial.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /absolute/path/to/checkpoints/020000_after32k_after32k/pretrained_model \
  --suite libero_spatial \
  --all-tasks \
  --episodes 10 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --action-index 1 \
  --exec-action-steps 16 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --use-suite-max-steps \
  --no-recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /absolute/path/to/eval_spatial_strict
```

### Four GPUs, one suite per GPU, strict within each suite

This is the closest multi-GPU equivalent of four independent serial suite
runs. It uses one model per GPU.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /absolute/path/to/checkpoints/020000_after32k_after32k/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --action-index 1 \
  --exec-action-steps 16 \
  --gripper-delta-threshold 0.002 \
  --use-suite-max-steps \
  --no-recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /absolute/path/to/eval_4suite_strict
```

### Four GPUs, two independent models per GPU

Use this for faster stochastic-policy evaluation on 24 GB cards. It loads
eight checkpoint copies in total and assigns each model five tasks. Do not also
enable task or episode parallelism.

```bash
# Use the same command as above, but replace the worker options with:
  --isolated-policy-workers 2 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1
```

One model used roughly 7–8 GB in the measured setup. Two copies per 24 GB GPU
worked; three copies are not recommended without measuring peak memory. Ten
independent task models per GPU do not fit.

## Modes that must not be mixed in one comparison table

- 5 Hz / 1000 steps versus standard suite horizons;
- mutable `checkpoints/last` versus explicit checkpoint hashes;
- serial batch 1 versus dynamic inference batch 10;
- one model versus two independent stochastic model workers;
- different action index / executed chunk length;
- different gripper delta thresholds;
- different initial scene settling or initial gripper state;
- save-video/viewer runs versus headless timing measurements.

## Saved evidence

Modern reports include the requested and resolved checkpoint paths, model
SHA256, policy/config/code hashes, package versions, GPU/driver inventory,
determinism settings, control settings, suite horizons, and worker mode.
Per-episode `actions.npz` contains model actions, LIBERO commands, controller
targets, achieved poses, tracking errors, chunk-boundary errors, and gripper
diagnostics. `progress.json` and `evaluation_events.jsonl` are updated during
evaluation, including isolated-policy mode.

The fixed flow-noise seed also seeds random choices inside one model forward,
but original spconv duplicate-coordinate handling can remain non-bitwise across
CUDA executions. For a stochastic-policy publication result, report several
seeds or confidence intervals instead of selecting the best single run.
