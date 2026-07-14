# ToolVQA 官方评测桥接使用说明

## 1. 这项测试来自哪里

本桥接直接调用本地 `ToolVQA-release` 项目的修改版 OpenCompass，不在 ToolVista 中重新实现评分器。

实际复用的上游代码包括：

- `opencompass.datasets.gta_bench.GTABenchDataset`
- `opencompass.openicl.icl_inferencer.AgentInferencer`
- `opencompass.models.lagent.LagentAgentVisual`
- `opencompass.datasets.gta_bench.GTABenchEvaluator`
- end-to-end 模式使用的 AgentLego 远程工具

ToolVista 只负责图片路径适配、生成 OpenCompass 配置、调用上游 `run.py`、保存日志，以及读取官方 `gta_bench.json`。

## 2. 两种评测模式

| 模式 | 上游行为 | 是否执行真实工具 | 适用节点 |
|---|---|---:|---|
| `step` | 使用标注历史逐步预测下一条 assistant 行为，运行 `every_with_gt` | 否 | 普通 checkpoint 轻量回测 |
| `e2e` | 从原问题和图片开始自主执行多步工具，运行 `every` | 是 | SFT、GRPO、BNPO、DAPO 最终或关键节点 |

`step` 主要测试多模态场景下的下一步工具选择、参数和最终回答切换。它与 xLAM 的函数调用协议测试有交集，但加入了图片和多步标注历史。

`e2e` 包含 Router 和 schema 能力，同时测试工具实际执行、工具返回理解、多步规划、终止决策和错误累积。

## 3. 本地 smoke 数据

仓库自带一条 ToolVQA 训练样例及对应原图：

```text
examples/data/toolvqa/dataset.json
examples/data/toolvqa/images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg
```

`dataset.json` 已整理为上游 `GTABenchDataset` 可直接读取的格式，但它来自训练样例，不是官方 2,550 条 test 集。

smoke 结果中固定包含：

```json
{
  "dataset_kind": "train_fixture",
  "comparable": false,
  "warning": "Not an official ToolVQA test score."
}
```

它只证明图片读取、模型 API、上游推理、官方 evaluator 和结果解析可以连通，不能与论文分数比较。

## 4. 运行前提

1. 本地或 AutoDL 已准备 `ToolVQA-release/opencompass` 环境，且该 Python 可以导入 OpenCompass、Lagent、AgentLego、datasets 和 sentence-transformers。
2. Qwen3-VL checkpoint 已经通过 ms-swift、vLLM 或兼容服务暴露为 OpenAI-compatible API。
3. 本地无鉴权服务可以设置 `OPENAI_API_KEY=EMPTY`；实际服务则设置对应 key。
4. `e2e` 模式额外要求 AgentLego server 已经启动。
5. 脚本不会自动部署模型、启动 AgentLego 或探测服务健康状态。

## 5. Windows CMD 启动 smoke

CMD 中必须使用物理单行命令，不要使用 PowerShell 的反引号。

```cmd
set OPENAI_API_KEY=EMPTY
python scripts\evaluate_toolvqa.py --mode step --toolvqa-root F:\code\ToolVQA-release --data-root F:\code\ToolVista\examples\data\toolvqa --api-base http://127.0.0.1:8000/v1 --api-model qwen3-vl --output-dir runs\toolvqa-smoke --limit 1
```

运行这条命令之前必须先启动模型 API。该命令产生的是一条训练 fixture 的 smoke 结果，不是官方 benchmark 分数。

## 6. PowerShell 启动 smoke

PowerShell 才能使用反引号续行：

```powershell
$env:OPENAI_API_KEY = "EMPTY"
python scripts\evaluate_toolvqa.py `
  --mode step `
  --toolvqa-root F:\code\ToolVQA-release `
  --data-root F:\code\ToolVista\examples\data\toolvqa `
  --api-base http://127.0.0.1:8000/v1 `
  --api-model qwen3-vl `
  --output-dir runs\toolvqa-smoke `
  --limit 1
```

## 7. AutoDL/Linux 启动 step

假设 ToolVista、ToolVQA-release 和官方测试数据位于同级或明确目录：

```bash
export OPENAI_API_KEY=EMPTY
python scripts/evaluate_toolvqa.py \
  --mode step \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --data-root /root/autodl-tmp/datasets/ToolVQA \
  --media-root /root/autodl-tmp/datasets/ToolVQA/images \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model qwen3-vl \
  --output-dir runs/toolvqa-step \
  --limit 100 \
  --workers 2
```

`--limit 100` 适合频繁的轻量 checkpoint 回测。省略 `--limit` 时运行配置中的完整数据集。

## 8. AutoDL/Linux 启动 end-to-end

先按照 ToolVQA 原项目说明启动 AgentLego，再运行：

```bash
export OPENAI_API_KEY=EMPTY
python scripts/evaluate_toolvqa.py \
  --mode e2e \
  --toolvqa-root /root/autodl-tmp/ToolVQA-release \
  --data-root /root/autodl-tmp/datasets/ToolVQA \
  --media-root /root/autodl-tmp/datasets/ToolVQA/images \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model qwen3-vl \
  --tool-url http://127.0.0.1:16181 \
  --output-dir runs/toolvqa-e2e \
  --workers 2
```

脚本只把 `--tool-url` 写入动态配置，真实工具定义、执行和返回格式仍由 ToolVQA/AgentLego 负责。

## 9. 结果指标

所有数值直接来自上游官方 `gta_bench.json`，ToolVista 只增加稳定前缀，不改变数值或重新评分。

step 指标：

| 官方字段 | ToolVista 字段 |
|---|---|
| `inst_align` | `toolvqa_sbs/inst_align` |
| `tool_acc` | `toolvqa_sbs/tool_accuracy` |
| `arg_acc` | `toolvqa_sbs/argument_accuracy` |
| `answer_acc` | `toolvqa_sbs/answer_accuracy` |

end-to-end 指标：

| 官方字段 | ToolVista 字段 |
|---|---|
| `answer_acc` | `toolvqa_e2e/answer_accuracy` |
| `p_f1` | `toolvqa_e2e/perception_f1` |
| `o_f1` | `toolvqa_e2e/operation_f1` |
| `l_f1` | `toolvqa_e2e/logic_f1` |
| `c_f1` | `toolvqa_e2e/creativity_f1` |

官方 `tool_call` 和 `tool_call_error` 仍完整保留在 `upstream_metrics`，但 MVP 不把它们作为主要比较指标。

## 10. 产物目录

每次运行的 `--output-dir` 中保留：

```text
<output-dir>/
├── runtime_config.py
├── data/
│   └── dataset.json
├── upstream/
│   └── <OpenCompass timestamp>/
│       ├── configs/
│       ├── predictions/
│       ├── results/
│       └── summary/
├── stdout.log
├── stderr.log
└── result.json
```

`result.json` 是便于 ToolVista 后续接入 suite/W&B 的精简结果；`upstream/` 中的数据才是完整的 ToolVQA/OpenCompass 原始产物。

## 11. 正式数据媒体路径

官方 `dataset.json` 可能保留作者机器上的绝对路径。脚本按以下顺序处理：

1. 原路径存在则直接使用。
2. `data-root` 下的相对路径存在则使用。
3. 否则在 `--media-root` 中按文件名递归查找。
4. 恰好一个匹配时重映射。
5. 没有匹配或出现多个同名文件时直接报错。

脚本只在 `<output-dir>/data/dataset.json` 写重映射副本，不修改 ToolVQA-release 或原始数据标注。
