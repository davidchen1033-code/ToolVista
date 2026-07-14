# ToolVQA Evaluation Bridge Design

**Date:** 2026-07-14
**Status:** Approved for implementation planning
**Scope:** ToolVQA official OpenCompass evaluation bridge only

## 1. Goal / 目标

Add a small ToolVQA-specific bridge to ToolVista so that ToolVista can generate an upstream-compatible OpenCompass runtime configuration, invoke the existing ToolVQA evaluation stack, preserve its raw artifacts, and normalize the official metrics into ToolVista's existing `EvaluationResult`, suite summary, history comparison, and W&B flow.

中文：为 ToolVista 增加一个精简的 ToolVQA 专用桥接层。ToolVista 只生成兼容上游 OpenCompass 的运行配置、调用现有 ToolVQA 评测流程、保留原始产物，并把官方指标包装成 ToolVista 已有的 `EvaluationResult`、suite 汇总、历史对比和 W&B 数据。

The bridge must maximize upstream reuse. It must not copy or reimplement ToolVQA's dataset loader, agent inferencer, model wrapper, AgentLego tools, answer matching, tool categorization, or evaluator.

中文：桥接层必须最大程度复用上游实现，不复制或重写 ToolVQA 的数据加载器、Agent 推理器、模型包装器、AgentLego 工具、答案匹配、工具分类或 evaluator。

## 2. Non-Goals / 不做什么

- Do not copy `GTABenchDataset`, `AgentInferencer`, `LagentAgentVisual`, `GTABenchEvaluator`, or AgentLego tool implementations into ToolVista.
- Do not permanently edit files in the sibling `ToolVQA-release` repository.
- Do not turn ToolVista into a generic external-process orchestration platform.
- Do not implement fine-grained lifecycle state, distributed scheduling, service supervision, or automatic tool deployment.
- Do not report the one-sample local training fixture as an official ToolVQA test score.
- Do not infer metrics by scraping terminal tables when the official result JSON is missing.
- Do not silently choose a media file when path remapping is ambiguous.

中文：本阶段不复制上游实现、不修改上游仓库、不建设通用编排平台、不管理复杂生命周期、不把本地一条训练样例当作正式测试分数，也不从终端文本猜测指标或在媒体路径冲突时静默选图。

## 3. Upstream Facts / 上游事实

The local upstream repository is expected to be a sibling repository, normally:

```text
F:\code\ToolVista
F:\code\ToolVQA-release
```

The official ToolVQA stack is under:

```text
ToolVQA-release/
├── agentlego/
├── opencompass/
├── xtuner/
└── requirements/
```

The upstream OpenCompass implementation provides two relevant evaluation modes:

| Mode | Upstream inferencer | Upstream evaluator | Executes tools | Official metrics |
|---|---|---|---:|---|
| Step-by-step | `AgentInferencer(infer_mode="every_with_gt")` | `GTABenchEvaluator(mode="every_with_gt")` | No | `inst_align`, `tool_acc`, `arg_acc`, `answer_acc`, `tool_call`, `tool_call_error` |
| End-to-end | `AgentInferencer(infer_mode="every")` | `GTABenchEvaluator(mode="every")` | Yes | `answer_acc`, `p_f1`, `o_f1`, `l_f1`, `c_f1`, `tool_call`, `tool_call_error` |

Step-by-step predicts each assistant step using the annotated preceding history. It measures multimodal next-step policy, tool selection, arguments, and answer transitions without allowing an early prediction error to corrupt every later step.

中文：step-by-step 使用标注历史逐步预测下一条 assistant 行为，测试多模态下一步策略、工具选择、参数和回答切换，前面一步预测错误不会污染所有后续步骤。

End-to-end starts from the original question and image, allows the model to execute remote AgentLego tools, observes their outputs, and continues until a final answer or turn limit. It includes Router and schema capability but also tests tool execution, observation understanding, multi-step planning, termination, and error accumulation.

中文：end-to-end 从原始问题和图片开始，由模型真实执行远程 AgentLego 工具、读取返回结果并继续推理，覆盖 Router、schema、工具执行、观察结果理解、多步规划、终止决策和错误累积。

## 4. Architecture / 架构

The selected approach is dynamic upstream configuration generation:

```text
scripts/evaluate.py
  -> load BenchmarkSpec and suite
  -> run_toolvqa_benchmark()
  -> ToolVQA preflight
  -> freeze/remap dataset paths into the run directory
  -> generate runtime_config.py
  -> invoke ToolVQA-release/opencompass/run.py
  -> preserve upstream predictions, results, summaries, stdout, and stderr
  -> parse the official gta_bench.json
  -> create EvaluationResult
  -> include metrics in ToolVista summary/history/W&B
```

Ownership remains explicit:

| Component | Responsibility |
|---|---|
| ToolVista | Validate inputs, remap media paths, generate runtime config, invoke upstream CLI, preserve artifacts, normalize metric names, compose suites, and upload summaries |
| ToolVQA OpenCompass | Load GTA/ToolVQA data, run step-by-step or end-to-end inference, and calculate official metrics |
| AgentLego | Execute real tools in end-to-end mode |
| Qwen3-VL OpenAI-compatible API | Predict the next tool call or final answer |
| ms-swift | Optionally serve a checkpoint through an OpenAI-compatible API; it does not score ToolVQA |

中文：ToolVista 负责输入适配和编排，ToolVQA/OpenCompass 负责官方推理与评分，AgentLego 负责真实工具执行，Qwen3-VL API 负责生成动作或答案，ms-swift 仅可作为模型部署入口。

## 5. Fixed Benchmark Entries / 固定 Benchmark

The implementation adds three fixed entries rather than a general external benchmark DSL:

| Benchmark ID | Profile | Data source | Comparable | Purpose |
|---|---|---|---:|---|
| `toolvqa_smoke` | `smoke` | One ToolVQA training fixture already stored in ToolVista | No | Verify the local image, generated config, Qwen3-VL API, upstream inference, evaluator, and result parser are connected |
| `toolvqa_step_by_step` | `light` or `full` | Official ToolVQA test data | Yes | Lightweight checkpoint regression for multimodal tool policy and arguments |
| `toolvqa_end_to_end` | `critical` | Official ToolVQA test data | Yes | Critical-node evaluation of the complete agent loop with real tools |

The local fixture is:

```text
examples/data/toolvqa/train_sample.jsonl
examples/data/toolvqa/images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg
```

The smoke run materializes this record into the `dataset.json` structure expected by upstream `GTABenchDataset`. It reuses the upstream `toolmeta.json` and all upstream inference and scoring classes. Its result must contain an explicit non-comparability marker and warning.

中文：本地 smoke 使用仓库已有的一条 ToolVQA 训练轨迹及对应原图，将其临时物化为上游 loader 所需格式。它只验证链路，不属于官方 test，也不能进入论文分数对比。

## 6. Runtime Parameters / 运行参数

The unified CLI adds only ToolVQA-specific arguments required to locate and invoke the upstream project:

| Argument | Meaning | Default behavior |
|---|---|---|
| `--toolvqa-root` | ToolVQA-release repository root | Try the sibling `../ToolVQA-release` path |
| `--toolvqa-data-root` | Official ToolVQA test directory containing `dataset.json`, `toolmeta.json`, and media | Use `<toolvqa-root>/opencompass/data/ToolVQA` |
| `--toolvqa-python` | Python executable for the upstream OpenCompass environment | Use the current Python executable |
| `--toolvqa-tool-url` | AgentLego server URL | Required only for end-to-end |
| `--toolvqa-workers` | OpenCompass worker count | A conservative small default |

The existing `--api-base` and `--api-model` arguments identify the Qwen3-VL OpenAI-compatible endpoint. The API key is read only from `OPENAI_API_KEY` in the process environment. It must not be embedded in the generated config, displayed command, stdout metadata, manifest, or W&B summary.

中文：模型地址和模型名复用已有 CLI 参数；API key 只通过环境变量传递，任何持久化配置和报告都不得包含真实凭据。

## 7. Frozen Data and Media Remapping / 冻结数据与媒体重映射

The released ToolVQA `dataset.json` can retain absolute media paths from the authors' machines. A downloaded media directory therefore may still be unusable without path remapping.

The bridge must preserve the original upstream file and generate a frozen copy under:

```text
runs/<run_id>/eval/<benchmark_id>/frozen_data/dataset.json
```

Only media references are rewritten:

- `files[].path`
- media-valued assistant tool-call arguments that refer to the same source asset

The following fields remain semantically unchanged:

- tool definitions
- dialog roles and content
- tool responses
- `gt_answer`
- question and answer annotations
- the official inputs consumed by the evaluator

The remapper first attempts a preserved relative suffix and then a unique filename lookup within the configured data root. It must reject missing media and ambiguous matches. It must report the total missing or ambiguous count and a small sample of paths. It must not modify `ToolVQA-release/opencompass/data/ToolVQA/dataset.json`.

中文：桥接层只在当前 run 目录生成路径重映射副本，仅修改媒体引用，不改对话、答案和评分语义。找不到图片或同名冲突时立即阻断，不猜测路径，也不修改上游原文件。

For the local smoke fixture, the same frozen-data interface is used, but the input is the existing training JSONL and image rather than the official test manifest.

## 8. Generated OpenCompass Configuration / 生成的配置

Each benchmark writes:

```text
runs/<run_id>/eval/<benchmark_id>/runtime_config.py
```

The file imports and composes existing upstream classes:

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

The bridge does not import ToolVQA code into the ToolVista Python process. The generated configuration is interpreted inside the selected upstream Python environment.

Step-by-step uses:

```python
infer_mode = "every_with_gt"
evaluator_mode = "every_with_gt"
tool_meta = "<upstream-toolmeta.json>"
```

End-to-end uses:

```python
infer_mode = "every"
evaluator_mode = "every"
tool_server = "<agentlego-url>"
max_turn = 8
```

Both modes use `LagentAgentVisual` with `OpenAIVisual`, the requested API model, and the normalized OpenAI SDK base URL. The bridge accepts either `http://host:port` or `http://host:port/v1` and normalizes both to exactly one `/v1`. It does not append `/chat/completions`, because upstream `OpenAIVisual` passes the value to the OpenAI SDK as `base_url`.

中文：runtime config 只组装上游已有类。step-by-step 使用 dummy tool metadata，不执行工具；end-to-end 使用 AgentLego server。API 地址按 OpenAI SDK 的 `base_url` 语义规范化为单个 `/v1`。

## 9. Upstream Invocation and Artifacts / 上游调用与产物

The bridge invokes the upstream CLI with an explicit work directory:

```text
<toolvqa-python> <toolvqa-root>/opencompass/run.py \
  <benchmark-dir>/runtime_config.py \
  --work-dir <benchmark-dir>/upstream \
  --eval_mode toolvista_qwen3vl \
  --max-num-workers <workers> \
  --debug
```

The subprocess working directory is `<toolvqa-root>/opencompass`. It receives a sanitized environment. No hardcoded upstream server address, partition, proxy, or example API key is reused.

The bridge preserves:

```text
runs/<run_id>/eval/<benchmark_id>/
├── runtime_config.py
├── frozen_data/
│   ├── dataset.json
│   └── toolmeta.json or a stable reference record
├── upstream/
│   └── <opencompass timestamp directory>/
│       ├── configs/
│       ├── predictions/
│       ├── results/
│       └── summary/
├── stdout.log
├── stderr.log
└── result.json
```

The stable official metric source is the generated OpenCompass JSON equivalent to:

```text
upstream/<timestamp>/results/toolvista_qwen3vl/gta_bench.json
```

The parser searches only within the benchmark's upstream directory and requires exactly one current official result file. It does not parse summary tables or console logs.

中文：上游原始 prediction、result、summary 和配置全部保留。ToolVista 只读取官方 `gta_bench.json`，不从日志或表格猜测分数。

## 10. Resume and Score-Only / 恢复与仅评分

Lifecycle behavior remains coarse:

- `--resume` reuses the existing benchmark upstream work directory and maps to OpenCompass `--reuse` behavior.
- `--score-only` for a single ToolVQA benchmark invokes OpenCompass with `--mode eval --reuse <existing-run>` and reuses existing predictions.
- A score-only request without reusable upstream predictions fails before starting OpenCompass.
- ToolVista does not implement per-sample job state or service restart logic.

中文：断点恢复和仅评分直接映射上游 OpenCompass 能力，不新增细粒度任务生命周期系统。

## 11. Metrics / 指标

`upstream_metrics` preserves the official result fields and values. ToolVista's normalized `metrics` adds stable benchmark prefixes without changing values.

Step-by-step mapping:

| Upstream | ToolVista |
|---|---|
| `inst_align` | `toolvqa_sbs/inst_align` |
| `tool_acc` | `toolvqa_sbs/tool_accuracy` |
| `arg_acc` | `toolvqa_sbs/argument_accuracy` |
| `answer_acc` | `toolvqa_sbs/answer_accuracy` |
| `tool_call` | `toolvqa_sbs/tool_calls` |
| `tool_call_error` | `toolvqa_sbs/tool_call_errors` |

End-to-end mapping:

| Upstream | ToolVista |
|---|---|
| `answer_acc` | `toolvqa_e2e/answer_accuracy` |
| `p_f1` | `toolvqa_e2e/perception_f1` |
| `o_f1` | `toolvqa_e2e/operation_f1` |
| `l_f1` | `toolvqa_e2e/logic_f1` |
| `c_f1` | `toolvqa_e2e/creativity_f1` |
| `tool_call` | `toolvqa_e2e/tool_calls` |
| `tool_call_error` | `toolvqa_e2e/tool_call_errors` |

Accuracy and F1 fields are already returned by the official evaluator on a 0-100 scale and remain on that scale. Counts remain counts. Primary metric directions are `maximize`; `tool_call_errors` is retained as a diagnostic and can be declared `minimize` when used in comparisons.

中文：官方 accuracy/F1 数值原本就是 0-100，桥接层不再次缩放；调用数和错误数保持计数语义。

## 12. Suites and Training Schedule / 套件与训练节点

The suite registry adds:

```yaml
c2_light:
  - xlam_function_calling
  - toolvqa_step_by_step
  - qwen3vl_general_light

c3_light:
  - xlam_function_calling
  - toolvqa_step_by_step
  - mllm_router
  - qwen3vl_general_light

toolvqa_critical:
  - xlam_function_calling
  - mllm_router
  - toolvqa_step_by_step
  - toolvqa_end_to_end
  - qwen3vl_general_light
```

Recommended use:

| Training point | Suite |
|---|---|
| Ordinary ToolVQA or Router SFT checkpoint | `c2_light` or `c3_light` |
| End of all SFT stages | `toolvqa_critical` |
| Intermediate GRPO, BNPO, or DAPO checkpoint | `c3_light` |
| Final checkpoint of each RL algorithm | `toolvqa_critical` |
| Final A/B comparison | Identical full ToolVQA configuration for all algorithms |

中文：普通 checkpoint 使用 step-by-step 轻量回测；SFT 完成、各 RL 算法最终节点和最终 A/B 对比再运行 end-to-end。

## 13. Preflight and Failure Semantics / 前置检查与失败语义

Preflight checks are mode-specific and intentionally simple.

Common checks:

- upstream root and `opencompass/run.py` exist;
- selected upstream Python exists or is executable by name;
- `dataset.json` and `toolmeta.json` exist;
- every required image can be mapped uniquely and opened;
- API base and model name are present;
- the model endpoint is reachable through a lightweight OpenAI-compatible check.

End-to-end additionally checks that the AgentLego URL is present and reachable.

Status rules:

| Condition | Status |
|---|---|
| Missing repository, data, media, model endpoint, or required AgentLego service | `blocked` |
| Upstream process exits non-zero | `failed` |
| Upstream exits zero but official result JSON is missing or malformed | `failed` |
| Official result JSON contains all required finite metrics | `passed` |

Step-by-step remains runnable when AgentLego is unavailable. End-to-end blockage must not retroactively invalidate a completed step-by-step result.

中文：缺依赖或服务属于 `blocked`，上游执行或结果解析错误属于 `failed`。AgentLego 不可用只阻断 end-to-end，不影响 step-by-step。

## 14. Security and Reproducibility / 安全与可复现性

- Generated configs use `key="ENV"`; they never contain the actual API key.
- Displayed commands redact credentials and sensitive URL components consistently with the existing CLI.
- Upstream example IPs, keys, proxies, Slurm partitions, and internal paths are never copied into ToolVista defaults.
- `result.json` records the sanitized expanded upstream command, selected upstream root, data root, mode, model name, sample limit, artifact paths, and process exit code.
- The frozen dataset and generated runtime config make the exact evaluation input inspectable without modifying the upstream repository.
- Smoke results include `comparable=false` and a warning that they are not official ToolVQA test results.

中文：真实凭据不落盘；评测输入、配置、命令和原始上游结果均可复查；smoke 明确标注为不可比较。

## 15. Planned Code Surface / 计划文件范围

The implementation is limited to:

```text
toolvista/evaluation/bridges/__init__.py
toolvista/evaluation/bridges/toolvqa.py
toolvista/evaluation/models.py
toolvista/evaluation/runner.py
scripts/evaluate.py
configs/evaluation/benchmarks.yaml
configs/evaluation/suites.yaml
tests/test_toolvqa_evaluation_bridge.py
tests/test_evaluate_script.py
docs/phase2_toolvqa_evaluation_usage.md
```

The bridge module owns ToolVQA-specific preflight, frozen-data creation, runtime config rendering, command construction, official result discovery, and metric mapping. The existing runner owns suite sequencing and generic `EvaluationResult` handling.

No upstream repository file is part of the change set.

## 16. Test Strategy / 测试方式

Automated tests do not install or imitate the complete OpenCompass stack. They test the ToolVista-owned boundary:

1. **Runtime config contract:** step-by-step contains `tool_meta`, `every_with_gt`, and no `tool_server`; end-to-end contains `tool_server`, `every`, and no dummy-only mode.
2. **Fixture materialization:** the existing ToolVQA training record and image become a valid one-record upstream `dataset.json` directory.
3. **Media remapping:** unique suffix and filename matches succeed; missing and ambiguous files block with actionable diagnostics.
4. **Official result parsing:** representative upstream `gta_bench.json` fixtures map to the expected normalized metrics while preserving raw metrics.
5. **Subprocess contract:** a fake command runner verifies executable, working directory, environment handling, logs, failure status, resume, and score-only arguments.
6. **CLI and suite regression:** the unified CLI accepts ToolVQA arguments and dispatches only ToolVQA entries to the bridge.
7. **Repository regression:** all existing Phase 1 and adapter/exporter tests continue to pass.

An optional real smoke run requires a running Qwen3-VL OpenAI-compatible endpoint and a usable ToolVQA OpenCompass environment. It uses the repository's actual sample image and the real upstream evaluator. Passing the optional smoke proves integration but does not create an official benchmark score.

中文：自动化测试覆盖 ToolVista 自己负责的边界；可选真实 smoke 使用真实原图、模型 API 和上游 evaluator 验证整条链路，但不作为正式成绩。

## 17. Acceptance Criteria / 验收标准

The ToolVQA bridge is complete when:

- ToolVista generates valid upstream configs for both official modes.
- The local one-image smoke fixture is converted into upstream loader format without duplicating evaluator logic.
- Official test media paths are frozen and remapped deterministically without editing the upstream source manifest.
- Step-by-step does not require AgentLego; end-to-end requires and checks it.
- The bridge invokes the sibling ToolVQA OpenCompass CLI in its own Python environment.
- Raw predictions, official results, summaries, generated config, frozen input, stdout, and stderr are preserved.
- Only official `gta_bench.json` values are normalized into `EvaluationResult`.
- Smoke results are visibly non-comparable; official test results remain comparable.
- Credentials and upstream example infrastructure values are not persisted.
- New focused tests and the complete existing test suite pass.
- `ToolVQA-release` has no modifications caused by the bridge.

中文：最终交付必须真正复用上游 CLI/evaluator，保留完整产物，正确区分 smoke 与官方 test，安全处理凭据，并保证上游仓库零修改和现有测试无回归。
