# ToolVista AutoDL 训练启动顺序

> 正式 C0 和训练前，先按 `docs/c0_data_and_training_readiness.md` 运行
> `scripts/prepare_training_and_eval_data.py`。该脚本会重建无泄漏的
> `xlam_train.jsonl`、text/image/video `mllm_tool_router_train.jsonl`，并生成
> xLAM/MLLM 的 full 与 light manifest。下面旧的逐数据集导出命令只用于单独调试 adapter。

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

### 3.3 MLLM-Tool Router 正式数据

```bash
python scripts/prepare_training_and_eval_data.py \
  --config configs/autodl.yaml \
  --seed 42 \
  --xlam-eval-fraction 0.05 \
  --mllm-eval-fraction 0.20 \
  --light-limit 200 \
  --mllm-file-keys text image video \
  --overwrite
```

该命令同时生成无泄漏的 xLAM 正式训练集，以及 MLLM-Tool text/image/video Router 正式训练集。缺媒体或无法转换的少量记录会被跳过并写入报告。

## 4. 冻结评测集和 C0 基座

正式训练前只运行一次上面的固定 seed 拆分。然后执行 readiness：

```bash
/root/autodl-tmp/venvs/toolvista/bin/python scripts/prefetch_vlmevalkit_data.py \
  --cache-dir /root/autodl-tmp/cache/vlmevalkit
```

该命令预取 RealWorldQA、MathVision MINI、MMMU DEV/VAL、OCRBench，
并以 VLMEvalKit 官方 MD5 校验后原子落盘。不要让五轮评测各自在运行中下载；
上游 downloader 不是原子写入，网络失败可能留下同名不完整 TSV。

```bash
python scripts/check_c0_readiness.py \
  --root /root/autodl-tmp/ToolVista \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --visual-rft-root /root/autodl-tmp/Visual-RFT \
  --coco-root /root/autodl-tmp/datasets/COCO2017
```

部署前先创建独立 ToolVQA evaluator 环境。脚本固定已验证版本、接入本地 OpenCompass/AgentLego fork，并通过 `hf-mirror.com` 预取官方评分器使用的 `all-mpnet-base-v2`：

```bash
bash scripts/autodl/00_setup_toolvqa_eval.sh
```

然后确认严格运行时；当前五轮需要 vLLM、`pycocotools`、上述 ToolVQA Python 环境，以及可以导入 `bfcl_eval` 的独立 BFCL Python 环境：

```bash
TOOLVQA_PYTHON=/root/autodl-tmp/venvs/toolvqa/bin/python \
python scripts/check_c0_readiness.py \
  --root /root/autodl-tmp/ToolVista \
  --require-gpu-runtime \
  --require-five-round-benchmarks
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

在另一个终端先记录与 C1-C4 完全同构的 C0 light，再记录 C0 full：

```bash
TOOLVQA_PYTHON=/root/autodl-tmp/venvs/toolvqa/bin/python \
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
RUN_ID=c0_base_light_seed42 \
STAGE=c0-base-light \
SUITE=c0_light \
JUDGE_API_BASE=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
API_MODEL=Qwen3-VL-4B-Instruct \
WANDB_GROUP=c0-base-light \
bash scripts/autodl/12_run_light_eval.sh
```

C0 full：

```bash
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
JUDGE_API_BASE=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
RUN_ID=c0_base_full_seed42 \
STAGE=c0-base-full \
API_MODEL=Qwen3-VL-4B-Instruct \
WANDB_GROUP=c0-base-full \
bash scripts/autodl/13_run_full_eval.sh
```

`c0_light` 到 `c4_light` 五轮使用完全相同的测试集合：

| 测试 | 测试能力 | light 数据 | full 数据 | 评测来源 |
|---|---|---|---|---|
| xLAM function calling | JSON/schema、函数名、参数、并行调用 | 固定 200 条 | trajectory holdout 4,767 条 | xLAM ActionStudio + ToolVista scorer |
| ToolVQA step | 图片和多步历史下的工具选择、参数、回答切换 | 官方 GTA 前 100 条 | 官方 GTA 1,622 条 | ToolVQA OpenCompass `every_with_gt` + `GTABenchEvaluator` |
| BFCL non-live | AST 函数调用、并行/多调用和无关请求拒答 | 分层固定 200 条 | 官方 v4 non-live 1,390 条 | Berkeley Function-Calling Leaderboard 官方 evaluator |
| MLLM Router | text/image/video 的 tool ID 路由 | 固定 200 条 | prompt-group holdout 1,726 条 | MLLM-Tool + ToolVista scorer |
| 通用视觉保留 | 现实视觉、数学、综合理解、OCR | 每集 100 条 | 上游完整默认 split | RealWorldQA、MathVision MINI、MMMU DEV/VAL、OCRBench，经 ms-swift + VLMEvalKit |
| Visual-RFT COCO | novel-15 开放词汇检测与定位 | 200 个 query/169 张 val 图 | 2,224 个 query/1,844 张 val 图 | Visual-RFT 候选图、COCO val2017、上游 `CocoDetectionEvaluator` |

light 用于五轮 checkpoint 内部同口径比较；full 用于 C0 建档和 C4 最终复测。xLAM/MLLM 是内部固定 holdout，不是上游 leaderboard test。Visual-RFT 测试来自 COCO val2017，与 6,000 条 COCO-base65 RL 训练集分离。

## 5. 串行 SFT

### 5.1 C1 xLAM

正式训练：

```bash
SEED=42 \
WANDB_PROJECT=toolvista-qwen3vl \
bash scripts/autodl/05_train_xlam_sft.sh
```

按当前 Qwen3-VL/Hermes 模板对 186,592 条正式 xLAM 全量精确编码：
最大 6,877 token，1,303 条（0.698%）超过默认 `MAX_LENGTH=4096`。
ms-swift 默认 `truncation_strategy=delete`，这些极端长 trajectory 会被删除；
99% 样本不超过 3,833 token。为控制单卡显存，本轮保留 4,096 作为固定口径，
四阶段不要单独改变该值。

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

C3 保留 MLLM-Tool text/image/video 无泄漏 Router 训练集，并按默认 60/20/20 混入 ToolVQA 和 xLAM：

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

先应用并检查当前 ms-swift 4.5.0.dev0 的 VLMEvalKit candidate alias 兼容补丁：

```bash
cd /root/autodl-tmp/ToolVista
PYTHON_BIN=/root/autodl-tmp/venvs/toolvista/bin/python \
bash scripts/autodl/patch_msswift_vlmeval.sh
```

先停止旧的 `swift deploy`，再部署当前 checkpoint。候选 adapter 必须用命名映射
`alias=/path/to/checkpoint` 部署；`API_MODEL` 必须等于该 alias。否则请求会命中基座，
而不是 LoRA。以 C3 为例：

```bash
C3_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c3_mllm_router_replay_sft/checkpoint-XXX
CANDIDATE_ALIAS=$(basename "$C3_CHECKPOINT")

swift deploy \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --adapters "$CANDIDATE_ALIAS=$C3_CHECKPOINT" \
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
API_BASE=http://127.0.0.1:8000/v1 \
API_MODEL="$CANDIDATE_ALIAS" \
JUDGE_API_BASE=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
BASELINE_SUMMARY=/root/autodl-tmp/ToolVista/runs/c0_base_light_seed42/summary.json \
HISTORY_SUMMARY=/root/autodl-tmp/ToolVista/runs/c2_toolvqa_seed42/summary.json \
WANDB_GROUP=c3-router-replay \
bash scripts/autodl/12_run_light_eval.sh
```

候选模型路由由 `API_BASE + API_MODEL` 决定，C0-C4 随 checkpoint 改变；裁判路由由
`JUDGE_API_BASE + JUDGE_MODEL` 决定，五轮必须冻结。默认内部 A/B 使用基座
`Qwen3-VL-4B-Instruct` 作为固定本地 judge。MathVision MINI 分数必须标记为
fixed local Qwen judge，不能与官方 GPT-4o-mini leaderboard 分数直接比较。

当前 EvalScope 1.9.0 会有损解析带下划线的数据集名。ToolVista 因此从 VLMEvalKit 原始
artifact 严格读取 RealWorldQA Overall、MathVision Overall acc、OCRBench Final Score；
MMMU DEV/VAL 按实际被测样本的逐样本 `hit` 计算联合 micro accuracy。缺任一主指标时
整项失败，不会生成残缺的可比较 summary。

五轮必须显式使用对应 suite：C0=`c0_light`、C1=`c1_light`、C2=`c2_light`、
C3=`c3_light`、四个 RL 分支分别使用 `c4_light`。这些 light suite 内容完全相同。
ToolVQA step 和 Visual-RFT COCO 由统一 runner 调上游 evaluator；ToolVQA e2e 仍使用
独立 `scripts/evaluate_toolvqa.py`，因为它额外要求 AgentLego 工具服务。

`BASELINE_SUMMARY` 和 `HISTORY_SUMMARY` 必须来自相同 profile。light 只能和 C0
light 比较，不能把 `c0_full` summary 当 light baseline。

## 7. 导出 Visual-RFT RL 数据

默认 reward 为 `visual_rft format`。`visual_rft` 是一个精简的任务路由插件：classification/VQA 类任务复用 Visual-RFT 上游的 `<answer>` 归一化字符串匹配，grounding/detection 类任务使用 IoU；`format` 直接复用 ms-swift 内置的 `<think>...<answer>...` 格式奖励。因此仓库中的 aircraft classification 示例可用于验证 1 step RL，但正式 A/B 默认使用已经导出的 6,000 条 COCO-base65 数据：

```text
data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl
```

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
export DATASET=/root/autodl-tmp/ToolVista/data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl
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

四个 RL checkpoint 都要各自用命名 alias 部署后运行 C4 light，不能只测最后完成的分支：

```bash
CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/rl/grpo_seed42/checkpoint-XXX
CANDIDATE_ALIAS=$(basename "$CHECKPOINT")

# 部署终端：
swift deploy \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --adapters "$CANDIDATE_ALIAS=$CHECKPOINT" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000

# 评测终端：
CHECKPOINT="$CHECKPOINT" \
RUN_ID=c4_grpo_light_seed42 \
STAGE=c4-grpo \
ALGORITHM=grpo \
SUITE=c4_light \
API_MODEL="$CANDIDATE_ALIAS" \
JUDGE_API_BASE=http://127.0.0.1:8000/v1 \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
BASELINE_SUMMARY=/root/autodl-tmp/ToolVista/runs/c0_base_light_seed42/summary.json \
HISTORY_SUMMARY=/root/autodl-tmp/ToolVista/runs/c3_router_seed42/summary.json \
bash scripts/autodl/12_run_light_eval.sh
```

BNPO、DAPO、DrGRPO 使用各自 checkpoint 和唯一 RUN_ID 重复同一命令。选出最终分支后，再用 `SUITE=c4_full` 运行完整复测。

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
