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
one-shot initialization only: compatible trained point encoders, adapters, and
time embeddings are copied by value into distinct World parameters before
training. No storage is shared and neither branch is frozen. The direct World
SE(3) output head remains independently initialized, and both cross-attention
output projections start at zero so enabling the branch initially preserves
the loaded baseline policy function.

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
