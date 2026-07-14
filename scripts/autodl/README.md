# ToolVista AutoDL 训练启动顺序

这套脚本只做数据组合和调用 ms-swift。训练器、LoRA、GRPO rollout、四种 loss、checkpoint 和训练指标都由本地安装的 ms-swift 提供。

## 1. 目录结构

推荐保持各上游项目为兄弟目录：

```text
/root/autodl-tmp/
|-- ToolVista/
|-- ms-swift/
|-- ToolVQA-release/
|-- xLAM/
|-- MLLM-Tool/
|-- Visual-RFT/
|-- model/
`-- datasets/
```

训练脚本的默认原始数据位置来自 `configs/autodl.yaml`。也可以通过命令行参数或环境变量覆盖，不需要把上游源码复制到 ToolVista。

## 2. 环境和模型

在 AutoDL 的 ms-swift/训练环境运行：

```bash
cd /root/autodl-tmp/ToolVista

# 已有兼容 vLLM 的 AutoDL 镜像时保持默认，避免替换镜像中的 PyTorch/CUDA。
bash scripts/autodl/00_setup_env.sh

# 只有确认当前 CUDA/PyTorch 与 vLLM 兼容但尚未安装时才开启。
INSTALL_VLLM=true bash scripts/autodl/00_setup_env.sh

bash scripts/autodl/01_download_qwen3_vl.sh
wandb login
```

默认模型：

```text
/root/autodl-tmp/model/Qwen3-VL-4B-Instruct
```

使用其他模型时，在每个训练命令前设置同一个 `BASE_MODEL=/path/to/model`。

## 3. 导出三阶段 SFT 数据

以下命令假设 `configs/autodl.yaml` 已指向真实 xLAM、ToolVQA 和 MLLM-Tool 数据。

### 3.1 xLAM

```bash
cd /root/autodl-tmp/ToolVista
python scripts/export_xlam.py \
  --config configs/autodl.yaml \
  --output data/exports/msswift_sft/xlam_train.jsonl \
  --report data/reports/xlam_train.report.json \
  --errors data/reports/xlam_train.errors.jsonl \
  --max-trajectories 0
```

这部分训练函数调用协议、函数名选择和参数生成。

### 3.2 ToolVQA

```bash
bash scripts/autodl/01_download_toolvqa.sh

MAX_ITEMS=0 \
OUTPUT=/root/autodl-tmp/ToolVista/data/exports/msswift_sft/toolvqa_train.jsonl \
bash scripts/autodl/02_export_toolvqa_ms_swift.sh
```

这部分训练带图片输入的多模态工具选择、参数生成和多步历史理解。

### 3.3 MLLM-Tool text Router

```bash
python scripts/export_mllm_tool.py \
  --config configs/autodl.yaml \
  --file-keys text \
  --output data/exports/msswift_sft/mllm_tool_text_train.jsonl \
  --report data/reports/mllm_tool_text_train.report.json \
  --errors data/reports/mllm_tool_text_train.errors.jsonl \
  --max-items 0
```

这里明确只用 `text` 子集，训练 Router/tool ID 选择，不依赖缺失的多媒体文件。

## 4. 冻结评测集和 C0 基座

正式训练前只生成一次固定 seed 的 xLAM 和 MLLM Router 开发集：

```bash
python scripts/prepare_eval_data.py \
  --config configs/autodl.yaml \
  --xlam-output data/eval/xlam/dev.jsonl \
  --mllm-output data/eval/mllm_router/dev.jsonl \
  --xlam-limit 200 \
  --mllm-limit 200 \
  --seed 42
```

部署基座模型：

```bash
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct
swift deploy \
  --model "$CHECKPOINT" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000
```

在另一个终端记录 C0：

```bash
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
RUN_ID=c0_base_seed42 \
STAGE=c0-base \
SUITE=c3_light \
API_MODEL=Qwen3-VL-4B-Instruct \
WANDB_GROUP=c0-base \
bash scripts/autodl/12_run_light_eval.sh
```

`c3_light` 当前包含：

| 测试 | 目的 | 数据/评测来源 |
|---|---|---|
| xLAM function calling | 函数名和参数协议 | xLAM ActionStudio 固定 seed 开发集，ToolVista 轻量 scorer |
| MLLM Router | Router top-1 和 macro F1 | MLLM-Tool text 固定 seed 开发集，ToolVista 轻量 scorer |
| 通用视觉保留 | 视觉、数学、综合理解和 OCR | RealWorldQA、MathVision MINI、MMMU DEV/VAL、OCRBench，经 ms-swift + VLMEvalKit |

这里的 xLAM/MLLM 数据是内部固定开发回归集，不是上游官方 leaderboard test。

## 5. 串行 SFT

### 5.1 C1 xLAM

正式训练：

```bash
SEED=42 \
WANDB_PROJECT=toolvista-qwen3vl \
bash scripts/autodl/05_train_xlam_sft.sh
```

1 step smoke：

```bash
MAX_STEPS=1 \
SPLIT_DATASET_RATIO=0 \
EVAL_STRATEGY=no \
SAVE_STEPS=1 \
DATASET_NUM_PROC=1 \
DATALOADER_NUM_WORKERS=0 \
OUTPUT_DIR=/root/autodl-tmp/ToolVista/outputs/smoke/c1_xlam \
bash scripts/autodl/05_train_xlam_sft.sh
```

训练完成后显式查看 checkpoint，不要让后续脚本猜测：

```bash
find outputs/c1_xlam_sft -maxdepth 1 -type d -name 'checkpoint-*' -print
```

### 5.2 C2 ToolVQA + xLAM replay

C2 保留全部 ToolVQA，并按默认 80/20 比例加入 xLAM。`C1_CHECKPOINT` 必须指向上一阶段实际产生的目录：

```bash
C1_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c1_xlam_sft/checkpoint-XXX \
SEED=42 \
bash scripts/autodl/06_build_and_train_toolvqa_replay_sft.sh
```

该脚本先生成：

```text
data/exports/msswift_sft/toolvqa_replay_train.jsonl
data/exports/msswift_sft/toolvqa_replay_train.report.json
```

然后用 `--resume_from_checkpoint C1 --resume_only_model true` 加载 C1 LoRA 权重。它不会继承 C1 optimizer、step 或数据游标。

### 5.3 C3 MLLM Router + ToolVQA/xLAM replay

C3 保留全部 MLLM-Tool text，并按默认 60/20/20 混入 ToolVQA 和 xLAM：

```bash
C2_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c2_toolvqa_replay_sft/checkpoint-XXX \
SEED=42 \
bash scripts/autodl/07_build_and_train_mllm_router_replay_sft.sh
```

生成：

```text
data/exports/msswift_sft/mllm_router_replay_train.jsonl
data/exports/msswift_sft/mllm_router_replay_train.report.json
```

`C2_CHECKPOINT` 同样只加载模型/LoRA 权重。最终明确记录 C3 路径：

```bash
find outputs/c3_mllm_router_replay_sft -maxdepth 1 -type d -name 'checkpoint-*' -print
```

## 6. 每阶段轻量回测

先停止旧的 `swift deploy`，再部署当前 checkpoint。以 C3 为例：

```bash
C3_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c3_mllm_router_replay_sft/checkpoint-XXX
swift deploy \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --adapters "$C3_CHECKPOINT" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000
```

另一个终端运行：

```bash
CHECKPOINT="$C3_CHECKPOINT" \
RUN_ID=c3_router_seed42 \
STAGE=c3-router-replay \
SUITE=c3_light \
API_MODEL=Qwen3-VL-4B-Instruct \
BASELINE_SUMMARY=/root/autodl-tmp/ToolVista/runs/c0_base_seed42/summary.json \
HISTORY_SUMMARY=/root/autodl-tmp/ToolVista/runs/c2_toolvqa_seed42/summary.json \
WANDB_GROUP=c3-router-replay \
bash scripts/autodl/12_run_light_eval.sh
```

C1 建议用 `SUITE=c1_light`；C2/C3 和四个 RL 分支用 `SUITE=c3_light`。ToolVQA 官方 step/e2e 目前仍由独立的 `scripts/evaluate_toolvqa.py` 调上游 OpenCompass，命令见 `docs/phase2_toolvqa_evaluation_usage.md`。

## 7. 导出 Visual-RFT RL 数据

默认 reward 为 `visual_rft format`。`visual_rft` 是一个精简的任务路由插件：classification/VQA 类任务复用 Visual-RFT 上游的 `<answer>` 归一化字符串匹配，grounding/detection 类任务使用 IoU；`format` 直接复用 ms-swift 内置的 `<think>...<answer>...` 格式奖励。因此仓库中的 aircraft classification 示例可用于验证 1 step RL，但正式 A/B 仍应使用同一份真实 Visual-RFT 数据。

以真实 Visual-RFT grounding 数据为例：

```bash
python scripts/export_visual_rft.py \
  --config configs/autodl.yaml \
  --data-path /root/autodl-tmp/datasets/Visual-RFT/grounding \
  --split train \
  --task grounding \
  --output data/exports/msswift_grpo/visual_rft_train.jsonl \
  --report data/reports/visual_rft_train.report.json \
  --errors data/reports/visual_rft_train.errors.jsonl \
  --max-items 0

python scripts/validate_visual_rft_rl_dataset.py \
  data/exports/msswift_grpo/visual_rft_train.jsonl
```

如果使用 Hugging Face `save_to_disk` 数据集，`--data-path` 指向该目录即可。图片会由 adapter 物化到配置的媒体目录。

## 8. 四个 RL A/B 分支

四个分支必须使用完全相同的 C3、数据、seed 和通用超参数：

```bash
export C3_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c3_mllm_router_replay_sft/checkpoint-XXX
export DATASET=/root/autodl-tmp/ToolVista/data/exports/msswift_grpo/visual_rft_train.jsonl
export SEED=42
export WANDB_PROJECT=toolvista-qwen3vl

bash scripts/autodl/08_train_visual_rft_grpo.sh
bash scripts/autodl/09_train_visual_rft_bnpo.sh
bash scripts/autodl/10_train_visual_rft_dapo.sh
bash scripts/autodl/11_train_visual_rft_drgrpo.sh
```

实际运行时按顺序逐个执行，避免单卡同时启动四个任务。默认输出：

```text
outputs/rl/grpo_seed42
outputs/rl/bnpo_seed42
outputs/rl/dapo_seed42
outputs/rl/drgrpo_seed42
```

| 分支 | 唯一算法差异 |
|---|---|
| GRPO | `loss_type=grpo` |
| BNPO | `loss_type=bnpo` |
| DAPO | `loss_type=dapo`，加 asymmetric clip、dynamic sample、overlong filter |
| DrGRPO | `loss_type=dr_grpo`，`scale_rewards=none` |

DAPO 本轮不额外加入 `soft_overlong` reward，目的是保证四个分支共享相同的 `visual_rft + format` reward。要评估完整 DAPO tricks，应另开实验，不要混入本轮 loss normalization A/B。

RL 1 step smoke：

```bash
MAX_STEPS=1 \
NUM_GENERATIONS=2 \
GRAD_ACC=2 \
STEPS_PER_GENERATION=2 \
MAX_COMPLETION_LENGTH=128 \
SAVE_STEPS=1 \
LOG_COMPLETIONS=false \
OUTPUT_DIR=/root/autodl-tmp/ToolVista/outputs/smoke/rl_grpo \
bash scripts/autodl/08_train_visual_rft_grpo.sh
```

RL 默认 `MAX_LENGTH=4096`、`VLLM_MAX_MODEL_LEN=4096`、`VLLM_GPU_MEMORY_UTILIZATION=0.4`，并开启 ms-swift 的 model/optimizer offload，优先保证单卡 smoke 能启动。显存足够时可以显式增大 context 或关闭 offload；四个 A/B 分支必须使用相同覆盖值。

RL 重点指标由 ms-swift 直接发到 W&B：总 reward、`VisualRFTORM` 与 `Format` 均值/标准差、`reward_std`、零方差 group 比例、completion 长度、KL、clip ratio、entropy 和 rollout/off-policy 指标。

## 9. 本地合同测试和 AutoDL GPU smoke

普通测试不加载模型：

```bash
python -m unittest \
  tests.test_build_sft_replay \
  tests.test_validate_visual_rft_rl_dataset \
  tests.test_autodl_launchers -v
```

Linux/AutoDL 可额外执行 Shell 参数展开合同：

```bash
TOOLVISTA_RUN_BASH_TESTS=1 \
python -m unittest tests.test_autodl_launchers.AutoDLLauncherRuntimeContractTest -v
```

这些测试只调用假的 `swift`。最终仍需要分别完成 C1 1 step、C2 1 step、C3 1 step、四个 RL 分支各 1 step，才能确认当前 GPU、CUDA、vLLM、TRL 和 ms-swift commit 的真实兼容性。
