# C0 数据冻结与正式训练准备

本文只覆盖当前最先要跑通的部分：冻结 C0 评测数据、生成无泄漏 SFT 训练数据、检查 AutoDL 是否可以切换到 GPU，并启动 C0 core full。少量无法转换或缺媒体的训练记录会被跳过并计入报告，不阻塞整条流程。

## 1. 本次数据拆分规则

| 数据集 | 正式训练数据 | C0 full | checkpoint light | 防泄漏单位 |
|---|---|---|---|---|
| xLAM ActionStudio | 约 95% trajectory 的全部可转换 assistant target | 约 5% trajectory 中可独立评分的 tool-call target | full 中固定 200 条 | `unique_trajectory_id` |
| MLLM-Tool | text + image + video 中约 80% prompt group | 约 20% prompt group，保留媒体路径并合并合法目标 | full 中固定 200 条 | 归一化 user prompt |
| ToolVQA | 官方 train，当前少量转换失败允许跳过 | 上游 OpenCompass 发布的 1,622 条 GTA benchmark | `--limit 100` 的 step 测试 | 官方 train/test split |
| Visual-RFT | 6,000 条 COCO-base65 作为 RL 训练集 | COCO val2017 + instances_val2017，由上游 evaluator 跑 | RL 日志和小规模任务回测 | 上游独立训练/验证数据 |

xLAM 不按导出的 JSONL 行拆分，因为一个 trajectory 可以导出多个 assistant target。MLLM-Tool 不按原始行拆分，因为同一句 prompt 可能重复出现并对应多个合法 tool ID。

## 2. 无卡模式：一次性生成正式数据

在 AutoDL 的 ToolVista 环境中执行一键准备：

```bash
cd /root/autodl-tmp/ToolVista
source /root/autodl-tmp/venvs/toolvista/bin/activate

SEED=42 bash scripts/autodl/04_prepare_c0_and_training_data.sh
```

该脚本先生成正式数据，再运行 readiness；任一正式训练文件规模不足、JSONL 损坏、媒体路径缺失、模型分片不完整都会返回非零退出码。需要单独调试拆分参数时，再直接调用 `scripts/prepare_training_and_eval_data.py`。

脚本还会在 readiness 前预取五轮通用视觉评测需要的四个 TSV：
RealWorldQA、MathVision_MINI、MMMU_DEV_VAL、OCRBench。预取使用
`.part` 临时文件，尺寸和 VLMEvalKit 官方 MD5 均通过后才原子替换正式缓存。
当前 OpenXLab HTTPS 证书已过期，因此脚本默认使用同一公开 Aliyun OSS
对象的 HTTP 地址，并依靠固定 MD5 做完整性校验；不会全局关闭 TLS 校验。

主要输出：

```text
data/exports/msswift_sft/xlam_train.jsonl
data/exports/msswift_sft/mllm_tool_router_train.jsonl
data/eval/xlam/full.jsonl
data/eval/xlam/light.jsonl
data/eval/mllm_router/full.jsonl
data/eval/mllm_router/light.jsonl
data/splits/xlam/seed42.json
data/splits/mllm_router/seed42.json
data/reports/c0/training_and_eval_data.report.json
```

脚本会验证 MLLM image/video 的媒体文件。缺失媒体的记录直接跳过。如果想先排查标注而不验证媒体，可以临时加 `--skip-media-validation`，但正式 C0 和正式训练不要使用该参数。

## 3. 无卡模式：检查数据和上游项目

```bash
cd /root/autodl-tmp/ToolVista

python scripts/check_c0_readiness.py \
  --root /root/autodl-tmp/ToolVista \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --visual-rft-root /root/autodl-tmp/Visual-RFT \
  --coco-root /root/autodl-tmp/datasets/COCO2017
```

重点看输出中的 readiness 布尔值：

| 字段 | 含义 |
|---|---|
| `data_ready` | C0 manifest 和 C1/C2/C3/RL 正式训练输入均存在且非空 |
| `training_data_ready` | C1/C2/C3/RL 正式训练输入、light 回测和 split manifest 均达到正式规模且媒体存在 |
| `c0_data_ready` | xLAM full 与 MLLM Router full 已达到正式规模 |
| `c0_core_ready` | 数据、模型、`c0_full` 配置和 `swift` 命令已准备好 |
| `gpu_runtime_ready` | 当前实例同时有 CUDA 和 vLLM；无卡模式为 `false` 是正常的 |
| `five_round_benchmark_data_ready` | 六项同构 benchmark 的 xLAM、ToolVQA、BFCL、MLLM、VLMEvalKit 配置和 Visual-RFT COCO manifest/evaluator 均已准备 |
| `five_round_benchmark_runtime_ready` | 六项数据已就绪，且当前环境同时具备 CUDA、vLLM、swift、pycocotools、可导入 lagent 的 ToolVQA Python 和可导入 bfcl_eval 的 BFCL Python |
| `optional_benchmarks_ready` | ToolVQA step/e2e 相关资产、Visual-RFT COCO 和 BFCL 运行环境均已准备；AgentLego e2e 服务仍需独立启动 |
| `bfcl_ready` | `BFCL_PYTHON` 是否能导入官方 `bfcl_eval`；BFCL 已属于正式六项 suite |

readiness 使用宽松的正式规模下限，例如 xLAM train 至少 100,000 行、ToolVQA train 至少 20,000 行、Visual-RFT train 至少 5,000 行、两份 light manifest 至少各 100 行。这样允许少量转换损失，但不会把一条或十条 smoke 文件误判为正式训练数据。

2026-07-17 本机实测：`torch 2.8.0+cu128`、`transformers 4.57.6`、
`vllm 0.11.0`、ms-swift `4.5.0.dev0` 可导入，CUDA 可用；ToolVQA、
BFCL、COCO 桥和六项 suite 的运行时检查通过。AutoDL launcher 默认使用
`/root/autodl-tmp/venvs/toolvista` 和 `https://hf-mirror.com`，仍可由环境变量覆盖。

不要依赖 VLMEvalKit 首轮边跑边下载。其上游 downloader 直接写正式缓存文件，
网络失败时可能留下不完整文件。可单独执行 verified prefetch：

```bash
/root/autodl-tmp/venvs/toolvista/bin/python scripts/prefetch_vlmevalkit_data.py \
  --cache-dir /root/autodl-tmp/cache/vlmevalkit
```

readiness 会逐项校验四个官方 TSV 的尺寸和 MD5。只有目录中其他缓存、或只准备
其中一两个数据集，不再被误判为通用视觉 benchmark 已完整离线就绪。

无卡阶段只要求 `data_ready=true`、`c0_core_ready=true`。切 GPU 后再使用严格检查：

```bash
python scripts/check_c0_readiness.py \
  --root /root/autodl-tmp/ToolVista \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --visual-rft-root /root/autodl-tmp/Visual-RFT \
  --coco-root /root/autodl-tmp/datasets/COCO2017 \
  --require-gpu-runtime \
  --require-five-round-benchmarks
```

## 4. 切 GPU 后：启动 Qwen3-VL 服务

终端 A：

```bash
cd /root/autodl-tmp/ToolVista
source /root/autodl-tmp/venvs/toolvista/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/ms-swift:/root/autodl-tmp/ToolVista:${PYTHONPATH:-}

swift deploy \
  --model /root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000
```

MLLM image/video full manifest 保存本地媒体路径。ToolVista 会把 `<image>`、`<video>` 占位符转换为 ms-swift OpenAI-compatible API 的 `image_url`、`video_url` content part，因此 Router full 不是伪装成多模态的纯文本评测。

## 5. C0 core full

终端 B：

```bash
cd /root/autodl-tmp/ToolVista
source /root/autodl-tmp/venvs/toolvista/bin/activate

CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct \
RUN_ID=c0_base_full_seed42 \
STAGE=c0-base-full \
API_MODEL=Qwen3-VL-4B-Instruct \
WANDB_GROUP=c0-base-full \
bash scripts/autodl/13_run_full_eval.sh
```

`c0_full` 包含：

| 测试 | 测试目的 | 数据/评测来源 | 本地代码 |
|---|---|---|---|
| xLAM function calling full | 函数名、参数和调用协议 | xLAM ActionStudio 固定 trajectory holdout | `toolvista/evaluation/scorers.py` |
| ToolVQA step full | 图片、多步历史下的工具选择和参数 | ToolVQA OpenCompass GTA 1,622 条 | 上游 `GTABenchEvaluator` |
| BFCL non-live full | AST、并行调用、多调用和无关请求拒答 | BFCL v4 non-live 1,390 条 | 官方 `bfcl_eval` evaluator |
| MLLM Router full | text/image/video Router top-1、macro F1、分模态准确率 | MLLM-Tool 固定 prompt-group holdout | `toolvista/evaluation/scorers.py` |
| Qwen3-VL general full | 检查视觉、OCR、数学、综合能力是否保留 | RealWorldQA、MathVision MINI、MMMU DEV/VAL、OCRBench | ms-swift `swift eval` + VLMEvalKit |
| Visual-RFT COCO full | novel-15 开放词汇检测 | COCO val2017 2,224 query/1,844 图 | 上游 `CocoDetectionEvaluator` |

full 配置不传 `eval_limit`，因此通用视觉部分按上游默认完整验证集运行。xLAM 和 MLLM full 也不是 200 条 light，而是冻结 holdout 中全部可评分记录。

## 6. C0 的三个上游专项测试

### 6.1 ToolVQA 官方 step full

原始 `raw_test_2550.json` 有 2,550 条标注，但 ToolVQA 上游仓库随 OpenCompass 发布、可直接复现其 evaluator 的 `dataset.json` 实际有 1,622 条。为了最大程度复用上游且保持分数可比，当前 full 指这 1,622 条 GTA benchmark 全量。终端 B 在 C0 core full 完成后执行，省略 `--limit`：

```bash
cd /root/autodl-tmp/ToolVista
export OPENAI_API_KEY=EMPTY

python scripts/evaluate_toolvqa.py \
  --mode step \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --data-root /root/autodl-tmp/ToolVQA-release/opencompass/data/ToolVQA \
  --media-root /root/autodl-tmp/ToolVista/data/raw/ToolVQA/images \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model Qwen3-VL-4B-Instruct \
  --output-dir runs/c0_toolvqa_step_full \
  --workers 2
```

它直接复用 ToolVQA 上游仓库自带的正式 `opencompass/data/ToolVQA/dataset.json` 与 `toolmeta.json`，并把其中作者机器上的图片路径重映射到 ToolVista 已下载的图片目录。评分继续由 OpenCompass、GTABenchDataset、AgentInferencer 和 GTABenchEvaluator 完成。C0 先跑 step；需要 AgentLego 工具服务的 e2e 留到关键节点环境完全验证后再跑。

### 6.2 Visual-RFT COCO full

当前已准备 COCO val2017、`instances_val2017.json`、冻结的 novel-15 light/full query manifest 和上游 `Visual-RFT/coco_evaluation`。当前 Qwen3-VL API 桥位于 `toolvista/evaluation/visual_rft_coco.py`：它将 0-1000 坐标转换为 COCO 像素框，并复用上游 `CocoDetectionEvaluator`。light 为 200 query/169 图，full 为 2,224 query/1,844 图，测试数据来自 COCO val2017，与 6,000 条 COCO-base65 RL 训练数据分离。

不要直接运行上游写死 Qwen2-VL 和至少两卡的 `Qwen2_VL_coco_infere.py`；统一 suite 已使用当前 API 桥。

### 6.3 BFCL 状态

BFCL v4 non-live 已进入 `c0_light` 到 `c4_light` 及 C0/C4 full suite。冻结 manifest 为分层 light 200 条和完整 full 1,390 条，覆盖 simple Python/Java/JavaScript、multiple、parallel、parallel-multiple 和 irrelevance 七类。统一 runner 通过 `scripts/bfcl_toolvista_bootstrap.py` 注册本地 OpenAI-compatible Qwen3-VL handler，并复用官方 AST 与 irrelevance scorer。

BFCL 使用独立环境，避免把其 Python 3.10 依赖混入 ToolVista 训练环境：

```bash
export BFCL_PYTHON=/root/autodl-tmp/venvs/bfcl/bin/python
"$BFCL_PYTHON" -c "import bfcl_eval"
"$BFCL_PYTHON" scripts/bfcl_toolvista_bootstrap.py \
  --project-root /tmp/toolvista-bfcl-registration-check \
  --id-manifest data/eval/bfcl/light.json \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model Qwen3-VL-4B-Instruct \
  --check-registration
```

这只验证 handler 注册，不调用模型。正式评分由统一 `c*_light` 或 `c0_full`/`c4_full` suite 执行。

## 7. C0 完成后开始 SFT

C1 直接使用新生成的无泄漏 xLAM：

```bash
cd /root/autodl-tmp/ToolVista

SEED=42 \
WANDB_PROJECT=toolvista-qwen3vl \
bash scripts/autodl/05_train_xlam_sft.sh
```

C2 仍使用官方 ToolVQA train + 新 xLAM replay。C3 的默认 MLLM 输入已改为：

```text
data/exports/msswift_sft/mllm_tool_router_train.jsonl
```

即 text + image + video 的无泄漏 Router 训练集，不再只训练 `mllm_tool_text_train.jsonl`。

四个 RL A/B 分支默认共同读取正式 6,000 条 COCO-base65 数据：

```text
data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl
```

仓库中的 32 条 aircraft 示例只用于 adapter、reward 和 1 step smoke 验证，不会被正式 RL 启动器默认使用。需要临时改用其他 Visual-RFT 导出文件时，再显式设置 `DATASET=/absolute/path/to/data.jsonl`。
