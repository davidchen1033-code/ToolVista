# ToolVQA Evaluation Bridge MVP Design

**Date:** 2026-07-14
**Status:** Proposed simplified design for review

## 1. Goal / 目标

Build the smallest bridge that can run the existing ToolVQA OpenCompass evaluator against a Qwen3-VL OpenAI-compatible API.

中文：先实现最小可运行桥接层，让 Qwen3-VL 的 OpenAI-compatible API 能直接调用 ToolVQA 原项目的 OpenCompass 推理和官方 evaluator。

The MVP supports both previously selected modes:

| Mode | Upstream behavior | Main use |
|---|---|---|
| `step` | `AgentInferencer(every_with_gt)` + `GTABenchEvaluator(every_with_gt)` | Frequent lightweight checkpoint tests |
| `e2e` | `AgentInferencer(every)` + `GTABenchEvaluator(every)` + AgentLego | SFT/RL critical-node tests |

The implementation must reuse these upstream classes directly. ToolVista does not copy their source or reproduce their scoring rules.

## 2. MVP Boundary / MVP 边界

The first implementation does only five things:

1. Validate the local ToolVQA repository and data paths.
2. Prepare an upstream-readable `dataset.json` when media paths need remapping.
3. Generate one OpenCompass runtime config.
4. Run the upstream `opencompass/run.py` command.
5. Read the official `gta_bench.json` and write one small ToolVista result JSON.

中文：第一版只做路径检查、数据路径适配、配置生成、调用上游命令和读取官方结果，不建设额外平台能力。

The MVP explicitly does not include:

- integration into `scripts/evaluate.py` or evaluation suites;
- W&B upload;
- automatic model deployment;
- automatic AgentLego startup or restart;
- HTTP health checks;
- resume or score-only orchestration;
- lifecycle databases or per-sample state;
- generic external-evaluator abstractions;
- copying or patching the ToolVQA repository.

These can be added only after the dedicated bridge completes a real smoke run.

中文：统一 suite、W&B、自动服务管理、断点恢复和通用编排都推迟到真实 smoke 跑通之后。

## 3. Minimal Code Surface / 最小文件范围

The MVP changes only:

```text
toolvista/evaluation/toolvqa.py
scripts/evaluate_toolvqa.py
examples/data/toolvqa/dataset.json
tests/test_toolvqa_evaluation.py
docs/phase2_toolvqa_evaluation_usage.md
```

No existing evaluation contract or runner needs to change in the MVP.

中文：第一版不修改 `BenchmarkSpec`、主 runner、benchmark registry 或 suite 配置，降低回归风险和实施时间。

## 4. Upstream Reuse / 上游复用

The generated config imports only upstream implementations:

```python
from lagent.agents import ReAct
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
from opencompass.models import OpenAIVisual
from opencompass.models.lagent import LagentAgentVisual
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
```

ToolVista does not implement:

- ToolVQA answer matching;
- tool-name or argument scoring;
- tool-category F1;
- ReAct parsing;
- multimodal message construction;
- AgentLego remote tool execution.

All of those remain owned by `F:\code\ToolVQA-release`.

## 5. Local Smoke Fixture / 本地 Smoke 样例

The repository already contains one real ToolVQA training record and its original image:

```text
examples/data/toolvqa/train_sample.jsonl
examples/data/toolvqa/images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg
```

The MVP adds one static `examples/data/toolvqa/dataset.json` in the exact structure expected by upstream `GTABenchDataset`. It references the existing image by relative path and contains the same annotated trajectory.

This avoids adding runtime conversion code for the smoke sample. It still uses the official loader, inferencer, model wrapper, and evaluator.

中文：把现有一条训练样例预先整理成上游原生 `dataset.json`，避免为 smoke 编写额外转换流程。评分仍全部由 ToolVQA 原项目完成。

Smoke results are labeled as:

```json
{
  "dataset_kind": "train_fixture",
  "comparable": false,
  "warning": "Not an official ToolVQA test score."
}
```

They verify connectivity only and never represent a paper-comparable benchmark result.

## 6. Official Data Path Adaptation / 正式数据路径适配

The released ToolVQA `dataset.json` may contain absolute media paths from the authors' machine. The bridge therefore accepts an optional `--media-root`.

For each `files[].path`:

1. Keep it if the path already exists.
2. Otherwise search `--media-root` by filename.
3. Use the match only when exactly one file is found.
4. Rewrite the corresponding media-valued tool-call argument to the same resolved path.
5. Fail before OpenCompass starts if the file is missing or ambiguous.

The adapted copy is written to:

```text
<output-dir>/data/dataset.json
```

The original upstream manifest remains unchanged. No complex suffix-ranking or media database is added in the MVP.

中文：正式数据只做最简单的唯一文件名重映射。缺失或同名冲突就直接报错，不猜测，不修改上游原文件。

## 7. Runtime Config / 运行配置

The bridge writes:

```text
<output-dir>/runtime_config.py
```

Common model configuration:

```python
type = LagentAgentVisual
agent_type = ReAct
llm = dict(
    type=OpenAIVisual,
    path="<api-model>",
    key="ENV",
    openai_api_base="<normalized-api-base>",
)
```

`--api-base` accepts either `http://host:port` or `http://host:port/v1` and is normalized to exactly one `/v1`. The API key is read from `OPENAI_API_KEY`; the generated config never contains the real value.

Step mode adds:

```python
infer_mode = "every_with_gt"
evaluator_mode = "every_with_gt"
tool_meta = "<toolmeta.json>"
```

`toolmeta.json` is resolved with one simple rule: use `<data-root>/toolmeta.json` when present; otherwise use `<toolvqa-root>/opencompass/data/ToolVQA/toolmeta.json`. This lets the local smoke fixture reuse the upstream tool definitions without copying them.

End-to-end mode adds:

```python
infer_mode = "every"
evaluator_mode = "every"
tool_server = "<agentlego-url>"
max_turn = 8
```

The script requires `--tool-url` only for `e2e`. It does not probe or manage the service.

## 8. Dedicated Command / 独立启动命令

Smoke step-by-step:

```powershell
python scripts\evaluate_toolvqa.py `
  --mode step `
  --toolvqa-root F:\code\ToolVQA-release `
  --data-root F:\code\ToolVista\examples\data\toolvqa `
  --api-base http://127.0.0.1:8000/v1 `
  --api-model qwen3-vl `
  --output-dir runs\toolvqa-smoke
```

Official end-to-end:

```powershell
python scripts\evaluate_toolvqa.py `
  --mode e2e `
  --toolvqa-root F:\code\ToolVQA-release `
  --data-root D:\datasets\ToolVQA `
  --media-root D:\datasets\ToolVQA\images `
  --api-base http://127.0.0.1:8000/v1 `
  --api-model qwen3-vl `
  --tool-url http://127.0.0.1:16181 `
  --output-dir runs\toolvqa-e2e
```

Optional `--python` selects the Python executable of the upstream OpenCompass environment. The default is the current interpreter. Optional `--workers` defaults to a conservative small value.

Optional `--limit` injects OpenCompass `reader_cfg.test_range="[0:<limit>]"`. The smoke command always uses one record; lightweight checkpoint tests can select a small fixed prefix, while an omitted limit runs the complete configured dataset.

## 9. Upstream Command / 上游命令

The script invokes:

```text
<python> <toolvqa-root>/opencompass/run.py \
  <output-dir>/runtime_config.py \
  --work-dir <output-dir>/upstream \
  --eval_mode toolvista_qwen3vl \
  --max-num-workers <workers> \
  --debug
```

The working directory is `<toolvqa-root>/opencompass`. Stdout and stderr are written to:

```text
<output-dir>/stdout.log
<output-dir>/stderr.log
```

The bridge does not reuse any hardcoded upstream IP, API key, proxy, Slurm partition, or internal path.

## 10. Result Parsing / 结果解析

After a successful upstream process, the bridge finds the newest official file matching:

```text
<output-dir>/upstream/*/results/toolvista_qwen3vl/gta_bench.json
```

It writes a compact `<output-dir>/result.json` containing:

```json
{
  "mode": "step",
  "dataset_kind": "train_fixture",
  "comparable": false,
  "upstream_result": ".../gta_bench.json",
  "upstream_metrics": {},
  "metrics": {},
  "command": "<sanitized upstream command>",
  "exit_code": 0
}
```

Metric mapping preserves official values:

| Step upstream | ToolVista key |
|---|---|
| `inst_align` | `toolvqa_sbs/inst_align` |
| `tool_acc` | `toolvqa_sbs/tool_accuracy` |
| `arg_acc` | `toolvqa_sbs/argument_accuracy` |
| `answer_acc` | `toolvqa_sbs/answer_accuracy` |

| E2E upstream | ToolVista key |
|---|---|
| `answer_acc` | `toolvqa_e2e/answer_accuracy` |
| `p_f1` | `toolvqa_e2e/perception_f1` |
| `o_f1` | `toolvqa_e2e/operation_f1` |
| `l_f1` | `toolvqa_e2e/logic_f1` |
| `c_f1` | `toolvqa_e2e/creativity_f1` |

Counts such as `tool_call` and `tool_call_error` are also preserved in `upstream_metrics`, but they are not required primary metrics for the MVP.

If the upstream process fails, the script exits non-zero and preserves logs. If the process succeeds but the official result JSON is missing or malformed, the script also exits non-zero. It never extracts scores from terminal text.

## 11. Tests / 测试

Only focused boundary tests are required:

1. Step config contains `every_with_gt` and `tool_meta`, not `tool_server`.
2. E2E config contains `every` and `tool_server`.
3. API base normalization produces exactly one `/v1`.
4. The static smoke `dataset.json` references the existing image.
5. Media remapping accepts one unique filename and rejects missing or duplicate files.
6. Official step and E2E result fixtures map to the expected metric keys.
7. A fake subprocess verifies command construction and log persistence.

The automated tests do not recreate OpenCompass or ToolVQA scoring. A real smoke run is performed only when a Qwen3-VL API and the upstream OpenCompass environment are available.

中文：单元测试只测 ToolVista 自己的配置、路径、命令和解析边界；真实 smoke 才验证上游环境和模型服务。

## 12. Acceptance Criteria / 验收标准

The MVP is complete when:

- both `step` and `e2e` configs are generated using upstream classes;
- the existing local image can be loaded through the static smoke fixture;
- the upstream `run.py` command is executed without modifying ToolVQA-release;
- stdout, stderr, runtime config, adapted data, official result, and compact result JSON are retained;
- only official `gta_bench.json` metrics are reported;
- the smoke result is visibly non-comparable;
- API credentials are not persisted;
- focused tests and existing repository tests pass.

After this MVP completes a real smoke run, a separate small follow-up may register it in `scripts/evaluate.py`, suites, W&B, and training-stage comparison summaries.

中文：先完成真实可运行的独立 bridge；确认链路稳定后，再用一个小任务接入统一评测系统。
