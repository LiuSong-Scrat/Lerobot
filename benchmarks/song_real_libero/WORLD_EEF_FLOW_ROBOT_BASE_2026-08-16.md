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
