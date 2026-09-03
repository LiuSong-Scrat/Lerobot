# RLBench 统一评测入口

新评测只使用：

```bash
bash /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/s4_eval.sh \
  --checkpoint 0808_022000 \
  --tasks close_box water_plants
```

参数注册表：

```text
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/RLBench/scripts/rlbench_eval_registry.json
```

`s4_RE_rlbench_official_eval_0808.sh` 保留为内部兼容后端，不再作为日常入口。

## 常用命令

查看全部 checkpoint/profile 和允许的任务：

```bash
bash scripts/s4_eval.sh --list
```

查看最终解析参数，但不启动：

```bash
bash scripts/s4_eval.sh \
  --checkpoint phone_action9_obs9_007000 \
  --tasks phone_on_base \
  --show
```

执行某些任务：

```bash
bash scripts/s4_eval.sh \
  --checkpoint 0808_022000 \
  --tasks close_laptop_lid stack_wine \
  --display 360
```

显式执行该 checkpoint 注册的全部任务：

```bash
bash scripts/s4_eval.sh --checkpoint 0808_022000 --tasks all
```

临时覆盖只对本次运行生效，优先级最高：

```bash
bash scripts/s4_eval.sh \
  --checkpoint 0808_022000 \
  --tasks close_box \
  --episodes 5 \
  --num-points 20000
```

## 参数合并顺序

从低到高：

1. 注册表 `defaults`
2. checkpoint/profile 的 `args` 和 `env`
3. checkpoint 下对应 task 的 `args`
4. `s4_eval.sh` 命令行临时覆盖

后出现的同名参数生效。每次运行还会把注册表路径、checkpoint ID、task
解析参数和最终 Python 配置分别保存到 `README.txt`、`command.txt`、
`task_presets/*.txt` 和任务目录的 `config.json`，便于复现。

## 从历史脚本完整迁移的配置

以下三个 profile 都记录了来源脚本的完整公共测评参数和全部 10 个任务 preset：

- `0808_022000`：`s4_RE_rlbench_official_eval_0808.sh`
- `0817_021000`：`s4_RE_rlbench_official_eval_0817.sh`
- `0820_021000`：`s4_RE_rlbench_official_eval_0820.sh`

checkpoint 下的 `args` 是该版本的公共参数，`tasks.<task>.args` 原样记录任务
preset；解析时任务参数位于最后，因此会正确覆盖公共参数。

## 添加新 checkpoint

只在 `rlbench_eval_registry.json` 的 `checkpoints` 中增加一项：

```json
"experiment_010000": {
  "description": "用途说明",
  "policy_path": "/absolute/path/checkpoints/010000/pretrained_model",
  "args": [
    "--num-points", "20000",
    "--gripper-points", "500",
    "--gripper-template", "reap",
    "--gripper-open-threshold", "0.055"
  ],
  "tasks": {
    "phone_on_base": {},
    "water_plants": {
      "args": ["--max-model-calls", "15"]
    }
  }
}
```

然后校验：

```bash
/home/liusong/miniconda3/envs/rlbench/bin/python \
  scripts/RE_rlbench_eval_registry.py \
  --registry scripts/rlbench_eval_registry.json \
  validate --check-paths
```

不要把训练时才需要的数据集、optimizer 或 cache 路径写进评测参数；注册表只
记录加载策略和执行 RLBench 所需的参数。
