# Phase 1 轻量评测使用说明

> 本文保留早期 200 条 light evaluator 工作流，仅用于兼容与调试。
> 正式 C0、无泄漏 train/full/light 拆分和 AutoDL 训练准备请使用
> `docs/c0_data_and_training_readiness.md` 与
> `scripts/prepare_training_and_eval_data.py`。

本文以下主体保留最初三个评测入口的历史说明。当前正式 C0-C4 五轮已经扩展为六项同构 light suite，另外统一接入 ToolVQA step、BFCL v4 non-live 和 Visual-RFT COCO；以 `configs/evaluation/suites.yaml` 与 `scripts/autodl/README.md` 为准。专项评测尽量复用上游的数据定义；xLAM 和 MLLM-Tool 缺少适合本项目流程的 held-out scorer，所以 ToolVista 只补充两个小型 scorer。通用视觉评测直接调用 ms-swift 已集成的 EvalScope/VLMEvalKit，不复制 evaluator 源码。

| 测试 | 测试方式 | 测评集及 split | 来源项目 | evaluator/scorer | 轻量限制 | 主要指标 |
|---|---|---|---|---|---:|---|
| xLAM 函数调用 | OpenAI-compatible 推理后做协议评分 | 固定 seed 的 one-step dev manifest | xLAM ActionStudio | `toolvista.evaluation.scorers.score_xlam` | 建议 200 | JSON/schema 合法率、函数名准确率、参数 F1/准确率、精确匹配、并行调用精确匹配 |
| MLLM Router | OpenAI-compatible 推理后做多标签 Router 评分 | MLLM-Tool `text` 固定 seed dev manifest | MLLM-Tool | `toolvista.evaluation.scorers.score_mllm_router` | 建议 200 | top-1、candidate hit、macro F1、非法 route、按模态准确率 |
| 通用视觉理解 | `swift eval` 调用 VLMEvalKit | RealWorldQA、MathVision_MINI、MMMU_DEV_VAL、OCRBench 的上游 split | ms-swift + VLMEvalKit | `swift eval --eval_backend VLMEvalKit` | 每个数据集 100 | 保留上游指标名称和量纲，再按 registry 做显式映射 |

前两项是从 xLAM ActionStudio 和 MLLM-Tool 上游数据中固定 seed 选出的 ToolVista 内部开发回归集，不是上游官方 held-out test，也不能当作 leaderboard 成绩。通用视觉项由 VLMEvalKit 按上游默认 split 下载和评分，可用于与相同 evaluator、相同版本和相同生成配置下的公开结果对照。

ToolVQA step、BFCL v4 non-live 和 Visual-RFT COCO 现已进入正式五轮 suite，并继续复用各自上游 evaluator。ToolVQA e2e 和 MAT 仍未进入五轮统一 light suite。

## 1. 准备冻结评测文件

AutoDL 中先确认 `configs/autodl.yaml` 指向实际 xLAM 和 MLLM-Tool 数据根目录，然后运行：

```bash
cd /root/autodl-tmp/ToolVista
python scripts/prepare_eval_data.py \
  --config configs/autodl.yaml \
  --xlam-output data/eval/xlam/dev.jsonl \
  --mllm-output data/eval/mllm_router/dev.jsonl \
  --xlam-limit 200 \
  --mllm-limit 200 \
  --seed 42
```

正式训练前，对带 `--include-metadata` 导出的训练 JSONL 做 sample ID 与规范化文本重合检查，并训练过滤后的文件：

```bash
python scripts/prepare_eval_data.py \
  --config configs/autodl.yaml \
  --xlam-output data/eval/xlam/dev.jsonl \
  --mllm-output data/eval/mllm_router/dev.jsonl \
  --xlam-limit 200 \
  --mllm-limit 200 \
  --seed 42 \
  --filter-training data/exports/msswift_sft/xlam_train_with_metadata.jsonl data/exports/msswift_sft/xlam_train_filtered.jsonl \
  --filter-training data/exports/msswift_sft/mllm_tool_text_with_metadata.jsonl data/exports/msswift_sft/mllm_tool_text_filtered.jsonl \
  --overwrite
```

`prepare_eval_data.py` 当前没有 filter-only 模式；带 `--filter-training` 时仍会重建并覆盖两个评测 manifest 及 report。因此第二条命令必须显式复用第一次完全相同的 output、limit 和 seed。它会在内存中完成过滤并校验过滤结果，再把 filtered 数据和相同的 manifest 以事务方式一次性落盘。不要在同一次命令中把尚未落盘的 filtered 输出传给 `--training-file`。生成后不要为每个 checkpoint 重新随机切分；相同 seed、manifest 和 generation 配置才能比较遗忘与增益。

## 2. 启动专项评测 API

xLAM 和 MLLM Router 直接复用 ms-swift 的部署能力，不在 ToolVista 重写模型服务：

```bash
# 第一次先评 C0 基座；评其他阶段时必须先停止旧服务，再改成该阶段的 checkpoint。
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct
API_MODEL=$(basename "$CHECKPOINT")
swift deploy \
  --model "$CHECKPOINT" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000
```

切换 checkpoint 时，由用户停止并重启该进程。下面执行 C0 suite 时使用上述基座；执行 C1 suite 前必须部署 `c1_xlam_sft`；执行 C3 suite 前必须先停止 C1 服务，再用同一模板部署 `c3_mllm_router_sft`。`--checkpoint`、`--api-model` 和 API 实际加载的模型必须指向同一个阶段。Phase 1 不实现服务守护、自动重启或复杂生命周期管理。

## 3. 基座与每次训练后的轻量回测

训练前先用完整 Phase 1 轻量 suite 记录 C0 基座，生成后续命令引用的 `runs/c0_base_light_seed42/summary.json`：

```bash
CHECKPOINT=/root/autodl-tmp/model/Qwen3-VL-4B-Instruct
API_MODEL=$(basename "$CHECKPOINT")
python scripts/evaluate.py \
  --suite c0_light \
  --checkpoint "$CHECKPOINT" \
  --run-id c0_base_light_seed42 \
  --stage c0-base \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model "$API_MODEL" \
  --eval-url http://127.0.0.1:8000/v1 \
  --seed 42 \
  --wandb \
  --wandb-group c0-base
```

C1 xLAM SFT 后，先停止基座服务并按第 2 节模板部署 `c1_xlam_sft`：

```bash
CHECKPOINT=/root/autodl-tmp/checkpoints/c1_xlam_sft
API_MODEL=$(basename "$CHECKPOINT")
python scripts/evaluate.py \
  --suite c1_light \
  --checkpoint "$CHECKPOINT" \
  --run-id c1_xlam_sft_seed42 \
  --stage c1-xlam-sft \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model "$API_MODEL" \
  --eval-url http://127.0.0.1:8000/v1 \
  --seed 42 \
  --baseline-summary runs/c0_base_light_seed42/summary.json \
  --wandb \
  --wandb-group c1-xlam-sft
```

C3 MLLM Router SFT 后，先停止旧服务并按第 2 节模板部署 `c3_mllm_router_sft`，再回测 xLAM、MLLM Router 和通用视觉保留能力：

```bash
CHECKPOINT=/root/autodl-tmp/checkpoints/c3_mllm_router_sft
API_MODEL=$(basename "$CHECKPOINT")
python scripts/evaluate.py \
  --suite c3_light \
  --checkpoint "$CHECKPOINT" \
  --run-id c3_mllm_router_sft_seed42 \
  --stage c3-mllm-router-sft \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model "$API_MODEL" \
  --eval-url http://127.0.0.1:8000/v1 \
  --seed 42 \
  --baseline-summary runs/c0_base_light_seed42/summary.json \
  --history-summary runs/c1_xlam_sft_seed42/summary.json \
  --wandb \
  --wandb-group c3-mllm-router-sft
```

通用视觉项等价于调用上游：

```bash
swift eval \
  --model /root/autodl-tmp/checkpoints/c3_mllm_router_sft \
  --eval_backend VLMEvalKit \
  --infer_backend vllm \
  --eval_dataset RealWorldQA MathVision_MINI MMMU_DEV_VAL OCRBench \
  --eval_limit 100 \
  --eval_output_dir runs/c3_mllm_router_sft_seed42/eval/qwen3vl_general_light/swift_eval \
  --eval_generation_config '{"max_tokens":512,"temperature":0}' \
  --eval_url http://127.0.0.1:8000/v1
```

已经用 `swift deploy` 占用同一张 GPU 时，suite 必须传 `--eval-url`，让 `swift eval` 复用现有服务。只有单独运行通用视觉项且没有外部 API 时，才省略 `--eval-url` 让 ms-swift 自行部署。

## 4. 简单恢复与重新评分

`--resume` 会按 benchmark 读取已有 `result.json`。只有 JSON/结构有效，且其中的 run ID、checkpoint、benchmark、profile、主指标和结果路径都与本次请求一致时才会跳过：

```bash
python scripts/evaluate.py <原参数> --resume
```

JSON 损坏、结构错误或身份不匹配会把该 benchmark 标记为 failed，不会当成可恢复结果；要重新执行时去掉 `--resume`，fresh run 会先使所选 benchmark 的旧 `result.json` 失效。

仅本地专项 benchmark 支持复用完整 `predictions.jsonl`。单 benchmark 重新评分会重写 run 级 `manifest.json` 和 `summary.json`，所以必须使用独立 run ID，并先放入要复用的完整 predictions：

```bash
RESCORE_RUN=c1_xlam_sft_seed42_xlam_rescore
mkdir -p "runs/$RESCORE_RUN/eval/xlam_function_calling"
cp \
  runs/c1_xlam_sft_seed42/eval/xlam_function_calling/predictions.jsonl \
  "runs/$RESCORE_RUN/eval/xlam_function_calling/predictions.jsonl"
python scripts/evaluate.py \
  --benchmark xlam_function_calling \
  --checkpoint /root/autodl-tmp/checkpoints/c1_xlam_sft \
  --run-id "$RESCORE_RUN" \
  --score-only
```

Phase 1 不实现逐样本 resume，也不自动检查旧 artifact 是否仍完整；需要严格复现实验时保留原 run 目录，只在新的 run ID 下重跑或重新评分。

## 5. 结果与 artifact

```text
runs/<run_id>/
|-- manifest.json
|-- eval/
|   `-- <benchmark_id>/
|       |-- predictions.jsonl
|       |-- upstream_result.json
|       |-- result.json
|       |-- stdout.log
|       |-- stderr.log
|       `-- swift_eval/
`-- summary.json
```

上图是 Phase 1 可能出现的 artifact 集合，不是每个 benchmark 都必然生成全部文件。native 专项会生成或复用 `predictions.jsonl`；swift/VLMEvalKit 会把上游工作目录保留在 `swift_eval/`，成功后再复制本轮原始报告为 `upstream_result.json`；启动、解析或校验失败时可能只有 `result.json`、`stdout.log` 和 `stderr.log`。

`result.json.test_summary` 和命令行末尾都会报告测试方式、测评集及 split、来源项目、evaluator/scorer、样本数或限制、实际命令、主要指标和路径。`runs/<run_id>/summary.json` 是跨 checkpoint 比较入口；原始上游结果仍保留在各 benchmark 目录。

## 6. 训练指标复用 ms-swift 与 W&B

ToolVista 不实现第二套训练监控器。SFT 至少启用：

```bash
--report_to wandb \
--logging_steps 5
```

ms-swift/Transformers 可直接记录 `loss`、学习率、epoch、global step、`grad_norm`、token accuracy、训练速度、耗时和显存日志，实际字段取决于训练后端。

GRPO、BNPO、DAPO 正式诊断时可增加：

```bash
--report_to wandb \
--log_completions true \
--log_entropy true \
--log_rollout_offpolicy_metrics true
```

RL 重点记录总 reward、各 reward function 均值/标准差、`reward_std`、零方差 group 比例、completion length、KL、entropy 和 rollout/off-policy 指标。轻量 smoke 可关闭详细 rollout 日志以减少存储和执行时间。

ToolVista 外层只补充 benchmark 开始/结束时间、持续时间和退出码。`peak_gpu_memory_mb` 在 Phase 1 允许为 `null`，因为不为它引入轮询守护进程。

## 7. 本地合同测试与真实模型结果的边界

仓库里的 `examples/evaluation/*.jsonl` 是冻结预测 fixture，只用于验证 JSONL、scorer、summary 和 ms-swift 报告解析合同。它们不是 Qwen3-VL 的真实预测，不能报告为模型 benchmark 成绩。

真实结果必须在 AutoDL 上先部署目标 checkpoint，再运行 `python scripts/evaluate.py`。建议第一次用 `--xlam-limit 2`、`--mllm-limit 2` 和单个轻量 benchmark 验证网络、媒体、模型服务和上游 evaluator，之后再运行完整轻量 suite。
