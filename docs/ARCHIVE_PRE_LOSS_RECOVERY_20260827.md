# Pre-loss-recovery code archive

- Archive date: 2026-08-27 (Asia/Shanghai)
- Source worktree: `/home/liusong/ProgramFiles/Huggingface/lerobot_7B_molmo2_song`
- Source branch: `molmo2-full-local`
- Source HEAD before snapshot: `d47eefd4ac54aa41b91dcb5ee2213f2a4078f5ce`
- Training-code reference: `d5abc45b5501b0cfa06c9c3cc76ac58f2e3968ca`
- Snapshot branch: `archive/molmo2er_v3_feature_align_language_casual_pre_loss_recovery_20260827`

The snapshot contains the current source, benchmark scripts, documentation,
tests, and local Molmo2-ER Python/configuration assets. Model weight shards,
runtime outputs, caches, and datasets are intentionally excluded.

## Immutable training baseline

The experiment baseline is the resolved checkpoint, not the mutable `last`
symlink:

`/opt/data/private/liusong/benchmarks/song_real_libero/outputs/lerobot_7B_molmo2_song/full_molmo2er_worldflow/v3_feature_align_long68_fresh_language_casual_after33000/checkpoints/033500`

- Restored global step: `33500`
- Model SHA256: `2ebaa90f7659a430965b8ab18df58c50ecd96038135a2288f1fdc49438e53a7e`
- Optimizer-state SHA256: `2044d9ce17ed5887d27c668c9862bc09918c275e7cef619edbb87b7f288e4c21`
- Train-config SHA256: `6544e8b2fe5a84c8e133c6390ce9871bfe269dbfb8dd452de5d65dfee327e8ca`
- Saved optimizer LR: `1e-8`
- Adam state: all 1,334 trainable tensors have step `33500`

The LR-recovery experiments must explicitly set
`resume_scheduler_phase_start_step=33500`; otherwise the phase origin stored in
the source train config would place a short restarted schedule directly at its
end LR.
