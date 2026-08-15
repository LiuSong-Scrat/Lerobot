# V51 conservative symmetric point adaptation workflow bundle

This directory is the Git-tracked recovery bundle for the V51 workflow prepared on 2026-08-15.

The executable model branch is intentionally pinned to:

```text
branch: wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation
commit: 93e2d8a3177a8addc40229da00962fcc2e7b7100
```

Do not run the pinned scripts from the workflow-archive commit itself: the scripts deliberately assert the executable worktree is exactly the commit above. In a new container, create a separate worktree at that branch, then install the runtime files into the persistent experiment root.

```bash
git clone git@github.com:LiuSong-Scrat/Lerobot.git lerobot
cd lerobot
git fetch origin --prune
git switch wep_vla_v0.4.3_v51_workflow_archive
git worktree add ../lerobot_v51 \
  origin/wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation

export LEROBOT_REPO="$PWD/../lerobot_v51"
export EXPERIMENT_ROOT=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811

bash benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/install_runtime_bundle.sh
bash "$EXPERIMENT_ROOT/scripts/launch_v51_cache_and_exact_audit_4gpu_tmux.sh"
```

After the cache and exact-index audit complete successfully:

```bash
LEROBOT_REPO="$LEROBOT_REPO" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  bash "$EXPERIMENT_ROOT/scripts/launch_v51_paired_symmetricpoint_training_4gpu_tmux.sh"
```

The first post-recovery launch exposed a step-0 preflight bug: the paired
semantic-contract check rejected the supported v12 dual-view/v11 primary
storage-schema combination. No optimizer update ran. Executable commit
`9db8504a2c575ad386f1d90efa98784c0ea8d701` fixes only that compatibility
check and adds a regression test. Preserve the failed original output and use
the separate restart entry:

```bash
LEROBOT_REPO="$LEROBOT_REPO" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  bash "$EXPERIMENT_ROOT/scripts/launch_v51r1_schemafix_paired_symmetricpoint_training_4gpu_tmux.sh"
```

The cache uses a 1 cm base voxel contract. The 4 cm value is only the secondary-view novelty threshold. Model input remains exactly 10,000 points.

Large checkpoints, datasets, cache shards, videos, and W&B runtime files are not stored in Git. Their authoritative paths and hashes are recorded in `CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md` and the preregistration JSON.
