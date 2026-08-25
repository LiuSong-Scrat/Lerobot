# Molmo2-ER V3 Feature-Align Contract

This branch narrows the existing V3 Action Expert; it does not add an
Action/Point residual branch.

## Fixed topology

- Molmo2-ER vision and text parameters remain frozen.
- The complete native `BOS + IMAGE + LANGUAGE` sequence remains first in the
  prefix and keeps Molmo2-ER's pretrained attention semantics exactly:
  `causal OR (image_query AND image_key)` over valid native tokens.
- Trainable Ego/World FG/BG tokens remain inside the evolving Molmo prefix,
  after the complete native sequence and before every action token.
- Native queries never read FG/BG scene tokens. Scene queries read every valid
  native token and all valid scene tokens bidirectionally.
- Action tokens remain in the Expert suffix.
- All 36 VLM/Expert layer pairs remain aligned.
- Even layers retain V3 joint attention: Action reads Native, Scene, and the
  complete Action block; Native and Scene cannot read Action.
- Odd layers retain prefix self-attention followed by Action-to-prefix
  cross-attention, so Action reads Native and Scene without an Action K/V block.
- PointAction keeps its existing `action + point_conditioned_update` output.

This topology deliberately does not make the whole prefix bidirectional. It
preserves the pretrained Native image/language path while using FG/BG as a
trainable multimodal readout block for the Action Expert.

## Feature dimensions

| Field | V3 | Feature-align |
|---|---:|---:|
| Molmo prefix hidden | 2560 | 2560 |
| Expert token hidden | 1920 | 720 |
| Expert MLP | 5120 | 2048 |
| Expert layers | 36 | 36 |
| Point token hidden | 64 | 64 |
| PointAction heads | 4 | 4 |

Molmo's `32Q / 8KV / head_dim=128` attention geometry is retained because
even-layer joint attention concatenates prefix and Expert Q/K/V along the token
axis. Changing the Expert to WEPVLA's `15Q / 5KV / head_dim=64` would require
projecting Molmo prefix Q/K/V and its output, which would no longer preserve the
native Molmo/V3 attention operation.

## MolmoAct2-style context projection

The 18 odd cross-attention layers use one shared pair:

```text
context_k_proj: Molmo KV 1024 -> Expert KV 1024
context_v_proj: Molmo KV 1024 -> Expert KV 1024
```

The output remains 1024 because the unchanged V3 joint-attention topology
requires Molmo's native KV geometry. Sharing follows MolmoAct2's parameter
ownership pattern; it changes neither token reachability nor residual paths.
K/V are not detached: Molmo parameters are frozen, while action loss must still
backpropagate through frozen Molmo operations to train the FG/BG projections.

## Exact parameter budgets

```text
Action Expert                         400,290,128 trainable
Full policy, WorldFlow enabled        467,816,260 trainable
Full policy, WorldFlow enabled      4,955,217,630 total
Frozen parameters                   4,487,401,370
```

The old V3 Action Expert had 1,736,591,232 parameters. Feature-align reduces
that component by 76.95%, and the whole WorldFlow policy's trainable count by
74.34%.

## Checkpoint boundary

The marker is:

```text
v3_feature_align_language_casual
```

Old V3 checkpoints are shape-incompatible (`1920 -> 720`). Earlier
feature-align checkpoints have compatible parameter shapes but incompatible
token order and attention semantics. Neither may be silently resumed. This
topology must be trained from a fresh policy configuration unless a separate,
explicit weight-conversion experiment is implemented.
