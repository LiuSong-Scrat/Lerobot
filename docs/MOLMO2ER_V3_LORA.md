# Molmo2-ER V3-LoRA contract

This branch preserves the V3 WEPVLA topology exactly:

- image/language/FG/BG remain the evolving Molmo prefix;
- action tokens remain the Action Expert suffix;
- the 36-layer VLM/Expert attention masks and DoubleFlow paths are unchanged.

LoRA is injected only into the frozen Molmo text decoder's attention blocks:
all 36 `self_attn.att_proj` modules and the first 35
`self_attn.attn_out` modules. The final Molmo attention output has no
downstream action-loss consumer in V3, so omitting that dead adapter keeps all
parameters reachable under DDP `find_unused_parameters=False`. The original
Molmo text/vision parameters stay frozen. The V3 Action Expert, PointSeg,
FG/BG projection, Point Action Adapter and WorldFlow parameters remain fully
trainable as before. This implementation intentionally does not use LeRobot's
generic PEFT wrapper (`policy.use_peft` remains `false`), because that wrapper
would freeze the rest of the policy and save an incomplete adapter-only model.

The default adapter is rank 8 with alpha 8 and no dropout. It adds exactly
4,370,432 trainable parameters. Adapter parameters and optimizer states remain
FP32 for reliable `1e-5` updates; only the small A/B matrices are cast to the
activation dtype during each projection. With WorldFlow enabled, the exact parameter
budget is:

```text
total:     6,315,281,166
trainable: 1,827,879,796
frozen:    4,487,401,370
```

Add these arguments to a V3 training command:

```bash
  --use_policy_training_preset=true \
  --policy.molmo_lora_enable=true \
  --policy.molmo_lora_rank=8 \
  --policy.molmo_lora_alpha=8.0 \
  --policy.molmo_lora_dropout=0.0 \
  --policy.molmo_lora_lr_multiplier=0.1 \
  --policy.use_peft=false
```

The original V3 trainable parameters use `policy.optimizer_lr`; Molmo LoRA
uses `policy.optimizer_lr * policy.molmo_lora_lr_multiplier` (default `1e-5`
when the V3 base learning rate is `1e-4`).

Two checkpoint load modes are supported:

1. A V3-LoRA checkpoint loads every tensor strictly.
2. A base V3 checkpoint may warm-start V3-LoRA only when the complete and exact
   set of 142 LoRA A/B tensors is absent. All historical V3 tensors must still
   match, and the zero-output LoRA initialization is retained.

Use `--resume=false` when upgrading a base V3 checkpoint, because its optimizer
state has no LoRA parameter group. V3-LoRA checkpoints can be resumed normally.
