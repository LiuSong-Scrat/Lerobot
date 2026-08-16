# Single-view WorldFlow: robot-base EEF trajectory contract

## Definition

WorldFlow predicts the future commanded end-effector trajectory in the
**complete robot-base coordinate frame**. It does not predict simulator object
poses or an explicit dense point-flow field.

For any pose or point `X`, conversion is

```text
T_base_X = inverse(T_world_base) @ T_world_X
```

Both the rotation and translation of `T_world_base` are applied. The fixed
camera extrinsic is used only to reconstruct geometry. The current EEF pose
maps the segmented current-EEF foreground cloud into robot-base coordinates.

## Why this still represents object flow

The World branch receives task-relevant foreground points and is supervised by
the EEF trajectory in the same fixed frame. During grasp or contact, the EEF
contact point moves with the object, so its trajectory is a sparse controlled
point-flow trace on the object. The model learns this relationship implicitly;
no object ID, object-pose label, correspondence field, or dense flow target is
introduced.

Free-space approach and post-contact motion remain parts of the commanded EEF
trajectory. Therefore this target should be described as World-frame EEF flow
that implicitly carries object-motion information, not object ground-truth
flow.

## Independent double-flow contract

`worldflow_target_type="world_eef_trajectory"` enforces:

- `worldflow_reference_frame="robot_base"`
- foreground scene XYZ and EEF targets in the complete robot-base frame
- independent World and Ego flow priors
- token cross-attention as the only World/Ego interaction
- no endpoint residual/rate, analytic bridge loss, coordinate augmentation, or
  shared parameters

World uses an SE(3) geodesic flow and a direct six-dimensional twist head even
when the pretrained Ego branch retains its legacy pose chart.

For the focused 100-demo experiment, `worldflow_bootstrap_from_ego=true` is a
one-shot initialization only: compatible trained point encoders, adapters,
time embeddings, and (when configured) the complete Action Expert are copied
by value into distinct World parameters before training. No storage is shared.
The direct World SE(3) output head remains independently initialized, and the
World-to-Ego cross-attention output projection starts at zero so enabling the
branch initially preserves the loaded baseline policy function.

`worldflow_action_expert_mode="independent"` is required for the corrected
dual-expert experiment. The historical `"shared"` mode is retained only so old
checkpoints remain loadable. In shared mode, World owned its token front-end
but its trajectory tokens were still processed by the Ego Action Expert. Once
protected-Ego training froze that expert, World had to learn around a fixed
Ego-specific temporal mapping and could interact with Ego only after the final
expert layer. This was not a genuinely independent double-flow architecture.

In independent mode:

- Ego uses the frozen pretrained Action Expert and unchanged Ego suffix.
- World uses a separately parameterized Action Expert bootstrapped from Ego.
- World Expert input contains Ego/World global scene tokens and only World
  action tokens; it never consumes Ego action tokens.
- the completed Ego and World trajectory-token sequences exchange information
  through explicit bidirectional cross-attention.
- World supervision updates the World Expert directly, while the baseline Ego
  weights and buffers remain byte-identical.

## Focused protected-Ego result before the dual-expert correction

The shared-expert protected-Ego run was evaluated with the same fixed seeds and
hard gate (task 6 stops at its fourth failure because it can no longer exceed
the `46/50` baseline):

| step | task 6 result at stop | outcome |
|---:|---:|---|
| 260 | 41/45, 4 failures | disqualified |
| 520 | 35/40, 5 failures | disqualified |
| 780 | 39/44, 5 failures | disqualified |
| 1040 | 45/50, 5 failures | below baseline |
| 1300 | 46/50, 4 failures | equals baseline |
| 1564 | 46/50, 4 failures | equals baseline |
| 1824 | 39/43, 4 failures | disqualified |
| 2084 | 39/43, 4 failures | disqualified |

All pretrained Ego tensors were audited byte-identical at step 260. Therefore
the result rules out catastrophic forgetting but does not establish WorldFlow
benefit. Extending training past step 1564 made performance worse, so further
step-count or learning-rate tuning is not justified. The next experiment must
change the shared Action Expert bottleneck itself.

## Independent Action Expert result

The corrected independent-Expert run used the same 100 demonstrations, fixed
evaluation seeds, protected pretrained Ego weights, and the original 1564-step
schedule. Every pretrained Ego tensor remained byte-identical. The hard-gated
results were:

| step | task 6 | task 8 | outcome |
|---:|---:|---:|---|
| 260 | 44/50 | not run | below `46/50` baseline |
| 520 | 44/50 | not run | below baseline |
| 780 | 47/50 | 45/50 | `+1 / +0`, total `92/100` |
| 1040 | 46/50 | not run | equals baseline |
| 1300 | 40/44 at hard stop | not run | cannot exceed baseline |
| 1564 | 42/46 at hard stop | not run | cannot exceed baseline |

This removes the shared-Expert bottleneck but still does not establish a
stable WorldFlow gain. Paired failures show both genuine repairs and new
regressions. At step 780, task 6 repairs baseline episodes `2/36/43` but adds
`23/25`; task 8 repairs `7/20/23/34`, retains `30`, and adds
`11/13/26/37`. Thus World information changes behavior, but the historical
final-latent World-to-Ego attention trades one failure set for another.

Two structural causes remain:

1. After the current-EEF foreground cloud is transformed into robot-base
   coordinates, the achieved current EEF pose is discarded as a model
   condition. The same nearly static foreground and instruction can then map
   to different future EEF trajectories at different rollout phases. The
   final training diagnostic (about `6.1 cm` translation error) confirms that
   this under-conditioned World branch is not an execution-quality trajectory
   predictor.
2. A final hidden-state attention update has no coordinate semantics. It asks
   the frozen Ego pose9 head to interpret an unconstrained World latent instead
   of converting the complete World trajectory into the controller's current
   EEF frame.

The next physical-execution contract therefore adds an explicit
`worldflow.current_ee_pose` robot-base token to the independent World Expert.
For arm execution it uses the independently predicted absolute trajectory via

```text
T_current_EEF_target = inverse(T_base_current_EEF) @ T_base_target_EEF
```

and retains Ego's gripper trajectory. Ego-to-World and World-to-Ego token
attention remain available for interaction, but the arm output is no longer
an unaligned latent perturbation. The equation is a left coordinate change,
not global-origin conjugation, endpoint residual, residual rate, learned gate,
or trajectory average.

## Dataset sidecars

- `world_base_ee_poses/episode_XXXXXX.npy`: achieved current EEF pose in
  robot-base coordinates, used to transform the foreground cloud.
- `world_base_action_target_ee_poses/episode_XXXXXX.npy`: commanded model-EEF
  target from each aligned raw LIBERO action, expressed in robot-base
  coordinates. Action chunks index this array exactly like the Ego labels.

The dataset wrapper requires strict robot-base metadata and refuses camera-frame
or achieved-future fallbacks in the new mode.

## Focused experiment

The authoritative single-flow baseline is
`eval_FULL4-9705*` (`481/500` on LIBERO-10). The focused tasks are task 8
(`45/50`) and task 6 (`46/50`), the two lowest baseline tasks. Their complete
50-demo HDF5 files provide exactly 100 training episodes.

Double-flow evaluation must reuse baseline environment seed 7, policy-noise
seed 0, official fixed initialization, control frequency 20, action index 0,
24 execution steps, and 50 episodes per task.

## Current-pose-conditioned direct-World execution result

The independent World Expert was then conditioned on the achieved current EEF
pose in robot-base coordinates. Training kept the original 1564-step LR
schedule but was stopped after the complete step-1300 checkpoint, as required.
The moving 50-batch World translation error improved from about `7.57 cm` near
step 353 to `5.39 cm` near step 1229, confirming that the missing current-pose
condition was a real under-conditioning error.

Directly executing that World trajectory was nevertheless decisively invalid:

| step | task 6 hard-gate result | outcome |
|---:|---:|---|
| 260 | 0/28 | disqualified |
| 520 | 0/28 | disqualified |
| 780 | 0/28 | disqualified |
| 1040 | 0/28 | disqualified |
| 1300 | 0/28 | disqualified |

At step 1300, the median Cartesian controller tracking error was about `5.2
cm`; all 28 completed episodes consumed the long failure horizon. Thus the
failure is not a checkpoint-selection or training-duration issue. A roughly
5-cm World predictor cannot replace the pretrained Ego controller as the
executed arm trajectory.

The current-pose condition remains necessary, but the
`world_trajectory_arm_ego_gripper` output route is rejected. The next causal
diagnostic loads the same immutable checkpoint with
`--worldflow-action-fusion-override cross_attention`. This changes only the
final execution route back to the jointly conditioned Ego trajectory; it is
recorded as a non-benchmark diagnostic and never edits the checkpoint files.

That diagnostic was hard-stopped at task 6 with `37/42` and five failures. Its
best possible final score was only `45/50`, below the `46/50` baseline, so task
8 was not run. It proves the direct-World `0/28` result was caused by the final
execution route rather than a corrupt checkpoint, but it also rejects the old
unconstrained final-latent World-to-Ego attention.

## Physical trajectory interaction

`physical_trajectory_cross_attention` replaces both rejected paths. At every
flow step, the two Action Experts first run independently and decode their own
complete endpoint proposals. Without dividing by the remaining path length:

```text
Ego endpoint in current EEF  -> T_base_current @ T_current_Ego_endpoint
World endpoint in robot base -> inverse(T_base_current) @ T_base_World_endpoint
```

Thus World attends to Ego's complete proposal in robot-base coordinates, while
Ego attends to World's complete proposal in current-EEF coordinates. Only
these physically aligned pose9 trajectory tokens enter the bidirectional
attention. The original raw World/Ego hidden-state attention is bypassed.

The interacted Ego hidden sequence is decoded by the unchanged pretrained Ego
action head; the World sequence is decoded by its independent SE(3) head. Both
attention output matrices start at zero, so bootstrap exactly preserves both
independent predictors, then learns a full token-to-token interaction. This is
not endpoint residual correction, residual rate, a scalar gate, trajectory
averaging, or direct World execution.

A real four-GPU one-step training smoke completed with batch 24 per rank and
six workers per rank, saved and reloaded its checkpoint, and completed one
online model call. After the first optimizer update, both physical interaction
attention output matrices were nonzero, while the frozen pretrained Ego action
head remained byte-identical to the baseline (`max_abs_diff=0.0`). The smoke
artifacts are under `SMOKE_world_eef_physicaltraj_1step_20260816` and
`SMOKE_EVAL_world_eef_physicaltraj_1step_20260816`.

## Focused physical-interaction result

The formal run declared the same 1564-step schedule as the preceding controls,
saved steps `260/520/780/1040/1300`, and was interrupted only after the complete
step-1300 checkpoint was durable (the training process reached step 1303 while
the checkpoint was being written). The output is:

```text
/opt/data/private/liusong/benchmarks/song_real_libero/outputs/
world_eef_task6_task8_100ep_physicaltraj_4gpu_b24_schedule1564_stop1300
```

Every checkpoint used the fixed task6/task8 seeds and 28 same-task episode
workers. A checkpoint had to strictly exceed task 6 baseline `46/50` before
task 8 was run, and task 8 then had to strictly exceed `45/50`.

| step | task 6 | task 8 | outcome |
|---:|---:|---:|---|
| 260 | 39/43 at 4-failure stop | not run | cannot exceed baseline |
| 520 | 48/50 | 45/50 | `+2 / +0`, total `93/100`; not qualified |
| 780 | 44/50 | not run | below baseline |
| 1040 | 41/45 at 4-failure stop | not run | cannot exceed baseline |
| 1300 | 45/50 | not run | below baseline |

Step 520 is causal evidence that physically aligned World interaction can help,
but not stable evidence of a two-task improvement. Against baseline failures
task6 `{2,26,36,43}` and task8 `{7,20,23,30,34}`, step 520 repaired every one
of those episodes. It introduced new task6 failures `{3,23}` and new task8
failures `{13,15,17,36,38}`. World therefore changes decisions in a useful but
still non-robust way rather than merely reproducing the frozen Ego policy.

Performance was not monotonic after step 520, so extending the same run to step
1564 cannot be justified as a convergence continuation. No learning-rate,
gradient, residual-rate, gate, or task-specific patch is applied. Under the
focused-goal stop rule, this experiment is stopped without advancing to
multi-view or all-suite evaluation.

## Corrected stochastic origin and base-frame velocity contract

The physical-interaction result exposed two remaining principle-level errors
that are independent of learning rate and training duration.

First, the World flow used an SE(3) prior sampled near the robot-base identity,
while the Ego flow used a pose prior in the current EEF frame. The two random
origins therefore did not denote the same physical EEF pose. For an absolute
robot-base EEF trajectory the correct coordinate map is

```text
T_base_noise = T_base_current @ T_current_noise
```

not the rejected motion conjugation
`T_base_current @ T_current_noise @ inverse(T_base_current)`. The new
`left_compose_ego` mode couples only this stochastic origin. The independent
World and Ego experts still predict and integrate their own vector fields
afterwards.

Second, an absolute World EEF pose was still integrated with a left-trivialized
SE(3) spatial twist. Its translational channel contains the global-origin
lever-arm term `omega × position`, so rotating an EEF away from the robot-base
origin spuriously moves its position around that origin. The corrected
`base_decoupled` velocity is

```text
[position_velocity_on_robot_base_axes,
 angular_velocity_on_robot_base_axes]
```

Position is added directly. Orientation is left-rotated on robot-base axes,
but that rotation never changes position. Translation follows a straight line
and orientation follows the SO(3) geodesic. The target vector field is constant
and the endpoint is recovered by multiplying by the remaining time; no
`1/(1-t)` residual rate is learned or executed.

Backward compatibility is explicit: old checkpoints that do not serialize
`worldflow_world_eef_velocity_mode` retain `legacy_spatial_twist`. The new
focused recipe explicitly writes both:

```text
worldflow_noise_coupling=left_compose_ego
worldflow_world_eef_velocity_mode=base_decoupled
```

Validation completed on 2026-08-16:

- 108 focused WorldFlow/SmolVLA tests passed.
- the mathematical tests distinguish left composition from conjugation and
  verify that pure EEF rotation has no global-origin translation.
- a real 4-GPU, batch-24-per-rank, six-worker-per-rank update completed and
  saved/reloaded a checkpoint at
  `SMOKE_world_eef_physicaltraj_leftcompose_decoupled_1step_20260816`.
- the reloaded checkpoint completed repeated online LIBERO model calls through
  the corrected integration path. The long 1-step random-head rollout was
  intentionally stopped after the execution path was established; it is not a
  performance measurement.

## Four-GPU checkpoint evaluation

Future checkpoint sweeps use
`scripts/run_world_eef_multicheckpoint_eval.sh`. Up to four evaluator parents
load different checkpoints concurrently, one per GPU, and then remain resident
behind an evaluation gate. The scheduler releases one resident model at a time
with all 28 episode workers. This overlaps checkpoint loading without reducing
the established single-checkpoint simulation parallelism or oversubscribing
the 30-thread host with four simultaneous 28-worker environment pools.

Each parent receives both `--task-id 6` and `--task-id 8` while retaining
`--task-workers 1`, so the two tasks execute sequentially against the same
loaded model rather than loading the checkpoint twice. Example:

```bash
WORLD_EEF_TRAINING_DIR=/path/to/training \
WORLD_EEF_EXPERIMENT_DIR=/path/to/experiment \
WORLD_EEF_STEPS='000260 000520 000780 001040 001300' \
WORLD_EEF_GPU_IDS='0,1,2,3' \
bash benchmarks/song_real_libero/scripts/run_world_eef_multicheckpoint_eval.sh
```

Set `WORLD_EEF_DRY_RUN=1` to inspect GPU assignment, worker allocation, paths,
and complete evaluator commands without launching an evaluation.
