# Single-view Object WorldFlow: robot-base contract

## Coordinate definition

The new WorldFlow reference is the **complete robot-base coordinate frame**, not
only a world frame whose origin was translated to the robot base. For any pose
or point `X`, conversion is

```text
T_base_X = inverse(T_world_base) @ T_world_X
```

Both the rotation and translation of `T_world_base` are applied. Consequently,
the fixed camera's position and orientation do not define the WorldFlow axes.
The camera extrinsic is used only to reconstruct geometry; the current EEF pose
then maps the UMI/Ego-frame foreground cloud into robot-base coordinates.

## Target definition

WorldFlow predicts simulator-derived object motion, not an EEF trajectory. For
the selected non-robot body at observation `t` and future state `t+h+1`:

```text
translation = p_base(t+h+1) - p_base(t)
rotation    = R_base(t+h+1) @ transpose(R_base(t))
```

Translation is object-body-origin displacement. Rotation is stored separately
about that body origin. This centered descriptor avoids the artificial
rotation/translation lever arm produced by conjugating a transform around the
robot-base or global origin.

The current implementation selects the most mobile non-robot MuJoCo body once
per demonstration. Every episode record stores the selected body name and
motion score for auditing. Tasks involving multiple independently moving
objects require a future multi-object target rather than silently combining
their motion into this single-object target.

## Independence contract

`worldflow_target_type="object_centered_motion"` enforces:

- `worldflow_reference_frame="robot_base"`
- global scene coordinates in the robot-base frame
- independent World and Ego noise
- token cross-attention as the only World/Ego interaction
- no endpoint bridge, residual/rate head, equivariance augmentation, or Ego
  weight bootstrap

WorldFlow uses an SE(3) geodesic flow and a direct six-dimensional twist head,
independently of whether the Ego action branch uses its legacy pose chart.

## Dataset sidecars

- `world_base_ee_poses/episode_XXXXXX.npy`: current EEF pose in robot-base
  coordinates, used to map the current-EEF foreground cloud into the base frame.
- `world_object_centered_motion/episode_XXXXXX.npy`: shape `(T, H, 9)`, where
  slot `h` maps observation `t` to achieved object state `t+h+1`.

Both directories have strict metadata declaring `coordinate_frame=robot_base`.
The dataset wrapper refuses to fall back to legacy EEF trajectories.

## Gates before formal training

1. Regenerate the dataset so both new sidecars exist; the old dataset is not
   label-compatible.
2. Audit selected object names and motion scores per task.
3. Visually verify several robot-base foreground clouds and centered targets.
4. Run a small overfit/mechanism test and confirm World loss decreases while the
   bridge loss remains exactly zero.
5. Only then run the baseline-controlled evaluation. Coordinate changes alone
   are not evidence that WorldFlow improves policy success.
