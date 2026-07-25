# C3 datafixed AutoDL 使用说明

这条链与现有 C3 control 并行。它只新建名称含 `datafixed` 的数据、训练输出、
评测 run 和 gate；不会改写 `mllm_tool_router_train.jsonl`、
`mllm_tool_router_v2_*`、旧 C3 output、`c3_light` 或
`toolvista_c3_retention_gate.json`。

## 1. 准备数据

先确认现有 `toolvqa_train.jsonl`、`xlam_train.jsonl` 和 `configs/autodl.yaml`
指向的数据源有效，再运行：

```bash
cd /root/autodl-tmp/ToolVista
SEED=42 bash scripts/autodl/prepare_data.sh --stage c3 --variant datafixed
```

已有 datafixed 产物时脚本默认失败；只有明确重建本实验数据时才设置
`OVERWRITE=true`。主要产物为：

```text
data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl
data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl
data/eval/mllm_router_datafixed/{light,full}.jsonl
data/splits/mllm_router_datafixed/seed42.json
data/reports/c3/router_datafixed.report.json
```

Router 行带 `[TOOL_ROUTER_MODE]` 与逐行候选 catalog；direct-answer 行不带
marker、tools 或工具轨迹。准备脚本完成前会审计这两个互斥契约。direct-answer
主要用于抑制模式塌缩，不能单独证明解释能力已保留，最终以 retention gate 为准。

## 2. 从 C2 训练

```bash
C2_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c2_toolvqa_replay_sft/vX/checkpoint-XXX \
SEED=42 \
bash scripts/autodl/train_sft.sh --stage c3 --variant datafixed
```

默认 replay 为 Router 40%、direct-answer 30%、ToolVQA 15%、xLAM 15%；
学习率为 `2e-5`，默认且最多运行 `1063` steps；按当前 replay 的 `17,001` 条训练样本和
有效 batch size `16` 计算，这恰好覆盖完整 `1 epoch`。脚本通过 `--adapters` 加载 C2 权重，并写入
`outputs/c3_mllm_router_datafixed_sft`。它拒绝 `MAX_STEPS>1063` 以及不含
`datafixed` 的实验输出路径。构建 replay 前还会读取
`router_datafixed.report.json`，要求报告中的 Router/direct-answer artifact
解析路径与实际输入一致，并重新计算 SHA256；数据被替换或报告过期时不会
启动训练。

本步骤先构建 replay，再启动训练；replay 产物为：

```text
data/exports/msswift_sft/mllm_router_datafixed_replay_train.jsonl
data/exports/msswift_sft/mllm_router_datafixed_replay_train.report.json
```

## 3. 建立同口径 C2 基线

为了固定 runner manifest 可记录的 benchmark 字段、seed、inference 和 judge
协议，部署 C2 adapter 后用 datafixed registry 重测。候选服务与固定 judge
必须使用不同 endpoint；
下面约定候选在 `8010`、固定 judge 在 `8011`。先完成环境和兼容性预检：

```bash
cd /root/autodl-tmp/ToolVista
export PYTHON_BIN=/root/autodl-tmp/venvs/toolvista/bin/python
export LMUData=/root/autodl-tmp/cache/vlmevalkit
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TOOLVQA_PYTHON=/root/autodl-tmp/venvs/toolvqa/bin/python
export BFCL_PYTHON=/root/autodl-tmp/venvs/bfcl/bin/python

"${TOOLVQA_PYTHON}" -c 'from lagent.agents import ReAct; from opencompass.datasets.gta_bench import GTABenchEvaluator; from opencompass.models.lagent import LagentAgentVisual'
"${BFCL_PYTHON}" -c 'import bfcl_eval'
CHECK_ONLY=true \
MS_SWIFT_ROOT=/root/autodl-tmp/ms-swift \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/autodl/patch_msswift_vlmeval.sh

curl -fsS http://127.0.0.1:8010/v1/models
curl -fsS http://127.0.0.1:8011/v1/models

"${PYTHON_BIN}" scripts/evaluate.py \
  --config configs/evaluation/benchmarks_datafixed.yaml \
  --suite-config configs/evaluation/suites_datafixed.yaml \
  --suite c2_datafixed_light \
  --checkpoint /path/to/c2/checkpoint \
  --run-id c2_datafixed_light_seed42 \
  --stage c2-datafixed-baseline \
  --api-base http://127.0.0.1:8010/v1 \
  --api-model c2-datafixed-baseline \
  --eval-url http://127.0.0.1:8010/v1 \
  --judge-api-base http://127.0.0.1:8011/v1 \
  --judge-model Qwen3-VL-4B-Instruct \
  --seed 42
```

保留整个 `runs/c2_datafixed_light_seed42/` 目录；gate 会同时读取
`summary.json` 与同目录 `manifest.json`。该 suite 使用冻结的
`mllm_router_datafixed/light.jsonl`，同时测 RealWorldQA、MathVision MINI、
MMMU DEV/VAL、OCRBench 及其他既有 light benchmark。

## 4. 评测 C3 候选并生成 gate

统一 benchmark launcher 会合并 adapter、启动候选与固定 judge，并进入实时百分比窗口：

```bash
bash scripts/autodl/benchmark.sh --stage c3 --variant datafixed
```

该 launcher 只读取 datafixed registry，并固定评估 `c3_datafixed_light`；它在候选
checkpoint 内写入 `toolvista_c3_datafixed_retention_gate.json`。gate 要求 Router
top-1 至少 0.60，RealWorldQA/MMMU/OCRBench 至少保留 C2 的 80%，MathVision
至少保留 70%，xLAM/ToolVQA/BFCL 至少保留 80%，Visual-RFT COCO（若存在）
至少保留 70%。
底层 evaluator 只接受 `c3` 前缀且包含 `datafixed` 的 stage。它从
`HISTORY_SUMMARY` 同目录读取唯一的 `manifest.json`，不接受独立的
`HISTORY_MANIFEST` 覆盖；benchmark 启动前会核对固定 judge 的 policy、endpoint
和模型 ID。C3 必须复用 C2 manifest 中记录的 judge，候选模型仍通过独立的
`API_BASE`/`EVAL_URL` 提供服务。


候选必须使用 datafixed Router v2；C2 基线可以包含旧 Router benchmark，但两边的
manifest 可记录的通用 benchmark 字段、seed、inference 和 judge 协议必须一致。
gate 还会交叉核对 summary/manifest 的 run ID 与 checkpoint，并只授权命令行
指定的 C3 adapter。

当前 runner manifest 不包含 `dataset_path`、完整 benchmark settings 或底层文件
哈希，因此 gate 不能单独证明数据文件字节完全相同。保留 C2/C3 run 目录时，也要
一并归档 `benchmarks_datafixed.yaml`、`suites_datafixed.yaml` 和对应数据清单。
gate 通过前不要从该 checkpoint 启动 C4。
