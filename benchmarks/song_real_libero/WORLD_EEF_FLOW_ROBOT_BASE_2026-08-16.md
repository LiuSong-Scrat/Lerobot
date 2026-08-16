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
