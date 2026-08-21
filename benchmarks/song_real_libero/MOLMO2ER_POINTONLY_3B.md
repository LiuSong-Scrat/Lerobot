# Molmo2-ER point-only 3B control

This experiment changes only the language backbone and the aligned Action
Expert depth/width of the point-only v0.4.3 policy.  It deliberately retains
the general-data PointSeg, point/action fusion, flow-matching objective,
optimizer, scheduler, action chunk, seed, and global batch contracts.

## Model contract

- Input prefix: exactly `[one foreground token, one background token, language]`.
- 2-D RGB input: disabled; checkpoint `image_features` must be empty.
- Molmo2-ER source: `/raid5/rongshengwang/Lerobot/Molmo2-ER`.
- Frozen VLM: token embedding + continuous blocks `0..17` of `36` + final norm.
- Excluded: Molmo vision backbone, LM head, and text blocks `18..35`.
- VLM shape: `L=18, H=2560, I=9728, Q/KV heads=32/8, head_dim=128`.
- Action Expert: `L=18, H=1920, I=5120`, with even self-attention layers and
  odd cross-attention layers.
- Molmo-native fused QKV, Q/K RMSNorm, grouped-query attention and RoPE
  `theta=5,000,000` are preserved.  Inputs are not multiplied by `sqrt(H)`.
- Language uses the Molmo tokenizer with exactly one native BOS token
  (`151645`) counted inside the unchanged 48-token limit; training and both
  inference wrappers share the same tokenizer helper.

Exact `nn.Parameter` counts:

| Component | Total | Trainable |
|---|---:|---:|
| Frozen Molmo text half | 2,206,041,088 | 0 |
| Action Expert | 868,296,576 | 868,296,576 |
| Existing WEP modules | 62,711,614 | 62,683,454 |
| Whole policy | **3,137,049,278** | **930,980,030** |

For the same runtime `nn.Parameter` counting convention, the existing 0.5B
checkpoint has 501,208,238 total and 151,032,494 trainable parameters.  The new
policy is 6.26x larger overall and has 6.16x as many trainable parameters.

The training entry point audits these counts, the trainable allowlist, frozen
VLM, absent vision/head parameters, effective RGB views, and mixed-precision
mode before optimizer/DDP construction.

## Locked training contract

- Dataset: `wep_vla_v042_general_data/libero_4suite_lerobot_dataset`.
- Temporal cache: `libero_4suite_lerobot_toolseg_cache`.
- PointSeg/LitePT and point-action fusion settings are unchanged.
- `chunk_size=32`, `n_action_steps=16`, flow inference steps `10`.
- AdamW: LR `1e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `1e-10`,
  gradient clipping `10`.
- Scheduler: warmup `100`, cosine decay `30000`, final LR `2.5e-6`.
- Seed `1000`, no AMP, DDP, global batch `192`.
- W&B runs in explicit offline mode because this host has no API key; local
  scalar/history logging remains enabled and training math is unchanged.
- On 8 GPUs: per-rank batch `24`, accumulation `1`.
- 80,000 optimizer updates; save/eval cadence `2,000`.

The two RTX 5090 sparse-convolution stems use `spconv.ConvAlgo.Native`.  The
default implicit path raises SIGFPE on this Blackwell runtime; this option does
not add parameters or change the network topology.

## Preflight results

- Contract pytest: `12 passed, 1 optional-real-weight test skipped`.
- Selective real load: 147 tensors / 2,206,041,088 parameters; vision 414,
  dropped text blocks 144, and LM head 1 tensor excluded.
- Official Molmo block-0 FP32 parity: max and mean absolute error `0`.
- Real-weight joint forward/backward and prefix-gradient checks passed.
- Tiny joint vs KV-cache suffix output: exact equality.
- The real full-policy compute path passed a two-update train, checkpoint save,
  strict reload, and `(1,32,10)` real-dataset action inference with finite
  outputs.  The launcher additionally requires a BOS-aware exact 8-GPU
  two-update smoke immediately before the formal run.
- 2-GPU DDP, batch 24 per rank, no accumulation: two updates passed; peak
  allocated memory was `20.54 GiB` per observed rank.

## Launch

The exact reusable command is in
`scripts/train_molmo2er_pointonly_3b_8gpu.sh`.  The shared-machine launcher
`scripts/wait_smoke_and_train_molmo2er_3b.sh` waits without killing foreign
processes, runs an exact 8-GPU/two-update smoke, and only then starts the 80k
run.

Checkpoints currently retain an absolute dependency on the local Molmo source
directory for architecture/tokenizer construction.  Parameter loading is
strict, but moving a checkpoint to another machine requires copying the Molmo
directory or updating the recorded paths consistently.
