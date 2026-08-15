# WEP-VLA 容器迁移恢复入口（2026-08-15）

当前代码、历史结构分支和 V51 可复现运行包已推送到 `git@github.com:LiuSong-Scrat/Lerobot.git`。

新容器可直接粘贴的完整 `/goal` 与 `~/.codex/config.toml` 权限配置见：`benchmarks/song_real_libero/GOAL_RESUME_MULTIVIEW_WORLDFLOW_2026-08-15.md`。

## 最短恢复流程

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
```

恢复包 README：

```text
benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/README.md
```

## 当前固定提交

| 用途 | 分支 | 提交 |
|---|---|---|
| 权威项目记录 | `wep_vla_v0.4.3_multiview_doubleflow` | `a1c0ac5f682c7858de88b5a3396365a4c47a9d26` |
| V32 确定性复核 | `wep_vla_v0.4.3_worldflow_v32_eval_recheck` | `dba65f7e0466364dcb43eaa016a7d7fb3c05c90d` |
| V46 1cm/3cm input | `wep_vla_v0.4.3_v46_multiscale_novelty_union` | `d752f14df8c75cfcd9acd9c6b34d8e2e7d5b7296` |
| V49 对称点路径 | `wep_vla_v0.4.3_v49_symmetric_point_adaptation` | `0b8e7b48dbddcbab847737ef6d676709cc8dfbef` |
| V50 保守输入 | `wep_vla_v0.4.3_v50_conservative_multiscale_novelty` | `3daeddd7de6afb7db086dffb3a0ebcf590d5db9c` |
| V51 可执行代码 | `wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation` | `93e2d8a3177a8addc40229da00962fcc2e7b7100` |
| V51 运行归档 | `wep_vla_v0.4.3_v51_workflow_archive` | `c9574e9cbfef3631d15d17c0fc74d3c08f755682` |
| 对称双坐标 | `wep_vla_v0.4.3_worldflow_symmetric_dualcoord` | `4562fd17c8de5c7ee15ae930b1afdebb7881fd93` |
| 并行双坐标 | `wep_vla_v0.4.3_worldflow_parallel_dualcoord` | `a04fc510cf35e61369c4ae3fc656a507c9ed9e00` |
| canonical dualflow | `wep_vla_v0.4.3_worldflow_canonical_dualflow` | `62e8670537ba72339c762ba954c528646757eb5f` |
| shared-state dual adapter | `wep_vla_v0.4.3_worldflow_shared_state_dual_adapter` | `507ffc575067e706e85f7b276662c8dd96a3c0d2` |

## 重要运行状态

- V51 cache 尚未生成。
- V51 training 尚未启动。
- V51 cache/training tmux 均不存在。
- 新一轮 pytest 在切换到归档请求时被中断，没有新结果；历史回归结果为 `78 + 35 + 16` passed，真正启动 cache 前应快速重跑。
- cache 基准 voxel 为 `1 cm`；`4 cm` 只用于副视角新颖性判断。
- 大型 dataset、checkpoint、cache 和 rollout 不进 Git；若 `/opt/data/private` 在新容器继续挂载，仍按原绝对路径复用。
- 不得删除 `/opt/data/private` 下任何文件。

完整对话决策归档位于 V51 workflow archive 分支：

```text
benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md
```
