# AutoDL 分阶段训练启动脚本设计

## 目标

为 ToolVista 增加一套可直接在 AutoDL 顺序执行的轻量训练入口，覆盖：

```text
Qwen3-VL base
  -> C1 xLAM SFT
  -> C2 ToolVQA + xLAM replay SFT
  -> C3 MLLM-Tool Router + ToolVQA/xLAM replay SFT
  -> GRPO / BNPO / DAPO / DrGRPO 四个独立 RL 分支
```

ToolVista 只负责数据组合、参数编排、结果目录和调用 ms-swift，不复制训练器、rollout 引擎或优化算法实现。

## 结构选择

采用“共享核心脚本 + 编号包装脚本”：

- `scripts/autodl/sft_common.sh` 保存公共 SFT 参数。
- C1、C2、C3 各有一个可直接启动的编号脚本。
- `scripts/autodl/rl_common.sh` 保存四个 RL 分支的公共参数。
- GRPO、BNPO、DAPO、DrGRPO 各有一个薄包装脚本，只选择算法配置。
- `scripts/build_sft_replay.py` 负责固定 seed 的 JSONL 抽样、混合、打乱和报告。
- `scripts/autodl/12_run_light_eval.sh` 对指定 checkpoint 调用现有统一评测入口。
- `scripts/autodl/README.md` 给出 AutoDL 目录、准备步骤和逐条命令。

不增加调度器、数据库、后台守护进程或细粒度断点生命周期系统。

## SFT 权重继承

C1 从原始 Qwen3-VL 基座开始。C2 和 C3 使用 ms-swift 的以下方式加载上一阶段结果：

```bash
--adapters <previous-stage-checkpoint>
```

这表示继承上一阶段可训练 LoRA 权重，但为新阶段重新创建 trainer state、optimizer、
学习率调度、训练步数和数据游标。阶段间不能使用 `--resume_from_checkpoint`，因为它是
中断恢复入口，会让 Transformers 读取上一阶段 `trainer_state.json` 和 `global_step`。

各阶段输出目录固定为：

```text
outputs/c1_xlam_sft
outputs/c2_toolvqa_replay_sft
outputs/c3_mllm_router_replay_sft
```

调用者通过 `C1_CHECKPOINT`、`C2_CHECKPOINT` 和 `C3_CHECKPOINT` 明确指定实际的 `checkpoint-*` 目录。脚本不猜测“最新 checkpoint”，避免静默使用错误模型。

## Replay 数据组合

`build_sft_replay.py` 接收一个或多个具名 JSONL 来源，每个来源指定目标条数或比例。第一版只处理已导出的 ms-swift JSONL，不重新运行 adapter。

默认组合为：

```text
C2: 80% ToolVQA + 20% xLAM
C3: 60% MLLM-Tool + 20% ToolVQA + 20% xLAM
```

包装脚本按主数据集实际条数反推 replay 条数；比例和 seed 可由环境变量覆盖。若请求条数大于来源条数，默认循环抽样并在报告中记录重复数量，保证小数据 smoke 也能构建出目标比例。

输出包括：

```text
<dataset>.jsonl
<dataset>.report.json
```

报告记录 seed、来源文件、读取条数、选取条数、重复条数、最终总数和 SHA-256。输入行必须是非空 JSON object；错误行直接终止，不生成半成品输出。

## RL 四分支

四个分支全部从同一个 `C3_CHECKPOINT` 开始，并共享：

- Visual-RFT GRPO JSONL；
- `toolvista/rewards/visual_rft.py` 外部 reward 插件与 ms-swift 内置 format reward；
- seed、学习率、batch、生成数量、最大长度和训练步数；
- LoRA、视觉冻结、vLLM colocate 和 W&B 设置。

统一调用：

```bash
swift rlhf --rlhf_type grpo
```

分支差异为：

| 分支 | ms-swift 配置 |
|---|---|
| GRPO | `--loss_type grpo` |
| BNPO | `--loss_type bnpo` |
| DrGRPO | `--loss_type dr_grpo --scale_rewards none` |
| DAPO | `--loss_type dapo --epsilon_high 0.28 --dynamic_sample true --max_resample_times 3 --overlong_filter true` |

DAPO 不默认追加 `soft_overlong` reward，因为当前 Visual-RFT 实验首先要保证四个分支共享相同 reward。插件按 task 将 classification/VQA 路由到上游式答案匹配、将 grounding/detection 路由到 IoU，格式分数直接复用 ms-swift 内置 reward。需要研究完整 DAPO tricks 时，可通过额外参数显式加入，作为另一个实验而不是混入本次 A/B。

四个分支输出到独立目录：

```text
outputs/rl/grpo_seed42
outputs/rl/bnpo_seed42
outputs/rl/dapo_seed42
outputs/rl/drgrpo_seed42
```

## 参数与 W&B

所有脚本支持用环境变量覆盖路径和规模参数。必要路径不存在、数据文件为空或上一阶段 checkpoint 未指定时立即失败。

默认 W&B 命名：

```text
WANDB_PROJECT=toolvista-qwen3vl
WANDB_GROUP=c1-xlam / c2-toolvqa-replay / c3-router-replay / rl-ab
WANDB_NAME=<stage>-seed<seed>
```

SFT 复用 ms-swift/Transformers 的 loss、learning rate、grad norm、token accuracy、吞吐和运行时间日志。RL 额外启用 reward、reward std、completion length、entropy 和 rollout off-policy 诊断；完整 completion table 默认关闭，可通过环境变量开启。

## 评测入口

`12_run_light_eval.sh` 不管理模型服务，只调用现有 `scripts/evaluate.py`。调用者先用 `swift deploy` 部署与 `CHECKPOINT` 相同的模型，然后传入：

```text
CHECKPOINT
RUN_ID
STAGE
SUITE
API_BASE
API_MODEL
```

这样保持模型生命周期简单，也避免启动脚本隐式占用 GPU 或错误重启服务。

## 验证

本地验证分三层：

1. Python 单元测试覆盖 replay 的确定性、比例、重复抽样、坏 JSON、空输入和原子写入。
2. `bash -n` 检查所有 Shell 语法。
3. 使用一个假的 `swift` 命令运行 wrapper contract test，检查 C1/C2/C3 权重传递、四个 RL `loss_type` 和共同参数，不实际加载模型。

GPU 上的最终验证仍需在 AutoDL 执行：SFT 1 至 5 step、部署后轻量评测，以及四个 RL 分支各 1 step。仓库测试通过不等于真实 CUDA、vLLM 和 ms-swift 版本已经完成训练验证。
