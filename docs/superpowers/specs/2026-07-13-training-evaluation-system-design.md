# ToolVista Training and Evaluation System Design

Date: 2026-07-13

Status: Approved design, pending implementation

## 1. Purpose

ToolVista needs a reproducible training and evaluation workflow for Qwen3-VL that quantifies capability gains, regressions, forgetting, and efficiency across staged SFT and reinforcement-learning experiments.

The planned progression is:

```text
Qwen3-VL base
  -> xLAM SFT
  -> ToolVQA SFT with xLAM replay
  -> MLLM-Tool Router SFT with historical replay
  -> unified SFT checkpoint
  -> GRPO / BNPO / DAPO comparison on common Visual-RFT data
  -> MAT agentic RL
```

The evaluation system must maximize reuse of upstream datasets, evaluators, scorers, tool environments, and metric definitions. ToolVista provides a thin conversion and orchestration layer rather than replacing mature upstream frameworks.

## 2. Goals

The system must:

1. Run lightweight cumulative regression after every training stage and reviewed RL run.
2. Run comprehensive evaluation at major checkpoints.
3. Reuse upstream evaluators without copying them into ToolVista.
4. Expose one ToolVista command for planning, executing, resuming, scoring, aggregating, and uploading an evaluation suite.
5. Preserve raw upstream predictions, trajectories, logs, results, metric names, and scales.
6. Produce a common result envelope for cross-checkpoint comparison.
7. Track retention, forgetting, training-token efficiency, generated-token efficiency, and GPU-hour efficiency.
8. Treat local JSON and JSONL artifacts as the source of truth and W&B as the online view.
9. Prevent train/evaluation leakage with immutable manifests and stable hashes.
10. Enforce a fair GRPO, BNPO, and DAPO comparison.

## 3. Non-Goals

The first implementation will not:

1. Copy upstream repositories into ToolVista.
2. Rewrite OpenCompass, VLMEvalKit, EvalScope, BFCL, COCO, LISA, or MAT evaluators.
3. Install all upstream projects into one Python environment.
4. Build a multi-node cluster scheduler.
5. Automatically download every large benchmark and media archive.
6. Select checkpoints from training loss or online RL reward alone.
7. Put evaluation logic in ToolVista dataset adapters or exporters.

The existing training-data path remains separate:

```text
raw dataset -> adapter -> CanonicalSample -> exporter -> training JSONL
```

The evaluation path is:

```text
checkpoint or model API
  -> ToolVista evaluation bridge
  -> upstream-compatible prediction format
  -> upstream evaluator or minimal ToolVista scorer
  -> raw upstream result
  -> ToolVista result envelope
  -> local reports and W&B
```

## 4. Thin Evaluation Bridge

Upstream projects retain ownership of:

- test data and split semantics;
- model-independent evaluator and scorer behavior;
- executable tool environments where required;
- metric definitions and original scales;
- benchmark-specific prediction formats.

ToolVista owns:

- checkpoint and API serving orchestration;
- test-manifest preparation and validation;
- prediction conversion into upstream formats;
- command construction and process lifecycle;
- parsing upstream outputs without changing metric semantics;
- a common result envelope and metric namespace;
- caching, resume, coverage, and failure accounting;
- retention, forgetting, efficiency, and multi-seed aggregation;
- local artifact storage and W&B upload.

The system has one command-line entry point, but it is not one large Python file. Benchmark-specific converters and parsers remain small, independently tested modules behind a registry.

## 5. Training Topology

Replay ratios are measured by effective supervised tokens, not sample count.

| Stage | Parent | Training data | Initial token mix | Required cumulative regression |
|---|---|---|---:|---|
| C0 Base | Qwen3-VL | None | None | All available baselines |
| C1 xLAM SFT | C0 | xLAM | 100% xLAM | xLAM and general retention |
| C2 ToolVQA SFT | C1 | ToolVQA plus replay | 70% ToolVQA, 30% xLAM | xLAM, ToolVQA, general retention |
| C3 MLLM Router SFT | C2 | MLLM plus replay | 60% MLLM, 20% ToolVQA, 20% xLAM | xLAM, ToolVQA, MLLM, general retention |
| C4 Unified SFT | C3 | all three SFT sources | 35% xLAM, 35% ToolVQA, 30% MLLM | Full SFT capability suite |
| R1 GRPO | C4 | Visual-RFT | Common RL data | All SFT suites, Visual-RFT, general retention |
| R2 BNPO | C4 | Visual-RFT | Same as R1 | Same evaluation as R1 |
| R3 DAPO | C4 | Visual-RFT | Same as R1 | Same evaluation as R1 |
| R4 MAT RL | selected RL checkpoint | MAT plus replay | Frozen before formal runs | All previous suites and MAT |

Every training run records planned and actual token ratios, source sample counts, supervised token counts, text and visual token counts, truncation counts, and total consumed training tokens. C4 is the single shared starting checkpoint for GRPO, BNPO, and DAPO.

## 6. Evaluation Cadence

### 6.1 Online Metrics

Record loss, learning rate, gradient norm, throughput, GPU memory, reward components, KL, completion length, truncation rate, and generated tokens at every available logging step.

### 6.2 Lightweight Cumulative Regression

Run after every SFT stage and after each reviewed RL run or checkpoint. Each specialty suite uses a frozen manifest of approximately 100 to 500 samples where upstream data permits. It always covers the new capability, all historical capabilities, and general retention.

### 6.3 Comprehensive Evaluation

Run full specialty and core evaluation at C0, C4, all nine final GRPO/BNPO/DAPO checkpoints, one representative checkpoint per RL algorithm, and the final MAT checkpoint.

Expensive general suites run at C0, C4, representative GRPO, representative BNPO, representative DAPO, and the final model. VideoMME is comprehensive-only.

## 7. Required Regression by Checkpoint

| Checkpoint | Required specialty regression |
|---|---|
| C1 xLAM SFT | xLAM function-calling protocol |
| C2 ToolVQA SFT | xLAM and ToolVQA |
| C3 MLLM Router SFT | xLAM, ToolVQA, and MLLM-Tool Router |
| C4 Unified SFT | all three SFT suites |
| R1/R2/R3 RL | all SFT suites and Visual-RFT |
| R4 MAT RL | all previous suites and MAT |

## 8. Metrics

### 8.1 xLAM

JSON validity, tool-call schema validity, function-name accuracy, argument-key precision/recall/F1, argument-value accuracy, argument exact match, parallel-call exact match, and trajectory success where labels exist.

### 8.2 ToolVQA

`every_with_gt` reports instruction alignment, tool accuracy, argument accuracy, answer accuracy, tool-call count, and tool-call error count.

`every` reports answer accuracy, tool-call count, tool-call error count, and perception/operation/logic/creativity F1.

### 8.3 MLLM-Tool Router

Output-format validity, top-1 accuracy, top-k accuracy, valid candidate-set hit rate, macro F1, per-modality accuracy, confusion matrix, and invalid-route rate. For multi-label samples, any listed valid tool is correct.

### 8.4 Visual-RFT

Format accuracy, mean IoU, box precision/recall, matched-box IoU, no-object accuracy, COCO mAP/AP50/AP75/size AP, LISA global and cumulative IoU, fine-grained classification accuracy, reward distribution, zero-reward rate, and completion length.

### 8.5 MAT and BFCL

MAT-Coding reports simple, hard, and overall F1. MAT-Search reports simple, hard, and overall F1 and exact match, plus tool count, exceptions, media validity, and latency. BFCL v3 reports AST accuracy, executable accuracy, relevance detection, and multi-turn accuracy.

## 9. General Retention

The lightweight suite uses RealWorldQA, MMMU, MathVision Mini, OCRBench, GSM8K, IFEval, and BFCL subsets.

The comprehensive suite may include MMBench, MMStar, MMVet, OCRBench, DocVQA, ChartQA, MathVista, HallusionBench, MMLU-Pro, GSM8K, IFEval, BFCL v3, and VideoMME.

Qwen3-VL official reproduction code is the preferred reference for MMMU, MathVision, ODinW-13, RealWorldQA, and VideoMME when available. ms-swift's EvalScope, OpenCompass, and VLMEvalKit backends are the default execution path for standard suites.

## 10. Upstream Inventory

### 10.1 ms-swift

```text
Repository: F:\code\ms-swift
Commit: 0a4f08e68d936c7d5d803c687b26f9d103dd0161
Documentation: docs/source/Instruction/Evaluation.md
Execution: command
Primary command: swift eval
Backends: Native, OpenCompass, VLMEvalKit
```

Light evaluation sets `--eval_limit`; full evaluation uses the same frozen generation configuration without a limit.

Upstream command template:

```bash
swift eval \
  --model /root/autodl-tmp/checkpoints/<checkpoint> \
  --eval_backend VLMEvalKit \
  --infer_backend vllm \
  --eval_dataset RealWorldQA MathVision_MINI MMMU_DEV_VAL OCRBench \
  --eval_limit 200 \
  --eval_generation_config '{"max_tokens":512,"temperature":0}'
```

### 10.2 xLAM

```text
Repository: F:\code\xLAM
Commit: a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec
Training data: actionstudio/datasets/unified_data/actionstudio-98k
Upstream benchmark reference: BFCL v3
Repository-local held-out scorer: absent
Execution: toolvista_native
```

ToolVista implements only the minimal held-out function-call scorer. Official BFCL is integrated separately when installed.

### 10.3 MLLM-Tool

```text
Repository: F:\code\MLLM-Tool
Commit: 8b74aad63ac511f6cbd7a74fb12aceee33975cf8
Legacy inference: code/inference.py
Evaluation candidate: data/IT_data_ins/T+X-T_data/combined_data.json
Repository-local scorer: absent
Execution: toolvista_native
```

The legacy script is tied to OpenLLaMA/Vicuna, ImageBind, and hard-coded paths and is not used for Qwen3-VL inference. `combined_data.json` contains 2,397 rows and 2,350 unique question-label pairs. An exact question-label comparison found no overlap with the four local source files. The final manifest must still apply normalized text and media-hash deduplication.

### 10.4 ToolVQA

```text
Repository: F:\code\ToolVQA-release
Commit: 99caf3a5cc613bf44d4bb60066fead87e871c430
OpenCompass entry: opencompass/run.py
Dataset config: opencompass/configs/datasets/gta_bench.py
Dataset loader/scorer: opencompass/opencompass/datasets/gta_bench.py
Test annotations: opencompass/data/ToolVQA/dataset.json
Tool metadata: opencompass/data/ToolVQA/toolmeta.json
Execution: command
```

The local OpenCompass dataset has 1,622 examples. All media references point to missing author-machine paths. The scorer and annotations are usable after media download and path remapping.

ToolVista dynamically generates an OpenCompass config instead of using `configs/eval.py`, which contains author-machine API and tool-server addresses. Step evaluation uses `every_with_gt`; end-to-end evaluation uses `every` and starts AgentLego.

Upstream command template after ToolVista generates the config:

```bash
cd /root/autodl-tmp/ToolVQA-release/opencompass
python run.py /root/autodl-tmp/ToolVista/runs/<run_id>/eval/toolvqa/generated_config.py \
  -p llmit \
  -q auto \
  --max-num-workers 32 \
  --debug \
  --eval_mode toolvista
```

### 10.5 Visual-RFT and Visual-ARFT

```text
Repository: F:\code\Visual-RFT
Commit: 2ffad63b25ddd79bfe25d3e046645401201c89d6
COCO scorer: coco_evaluation/coco_evaluation.py
LISA scorer chain: lisa_evaluation
Classification annotations: classification/val_data
MAT coding: Visual-ARFT/evaluation_coding
MAT search: Visual-ARFT/evaluation_search
Execution: python_callable or command
```

The classification annotations contain 3,333 FGVC-Aircraft, 2,463 Oxford Flowers, 3,669 Oxford Pets, and 8,041 Stanford Cars validation rows. Their image paths are missing author-machine paths, so ToolVista must create a remapped immutable manifest after downloading official images.

COCO, LVIS, and LISA require external image and annotation packages. For MAT, prefer current VLMEvalKit MAT-Bench support; the repository prediction formats and notebook formulas are compatibility fallbacks because the old scripts hard-code Qwen2.5-VL classes and paths.

### 10.6 Specialty Evaluation Launch Matrix

The paths below are the exact local upstream locations inspected for this design. Commands marked `legacy` reproduce the upstream entry point but are not the final ToolVista interface. Angle-bracket values in command templates are explicit operator inputs.

| Capability | Upstream location | Current upstream launch | Current readiness |
|---|---|---|---|
| xLAM function calling | `F:\code\xLAM` | No repository-local held-out evaluator | Requires Phase 1 ToolVista scorer |
| MLLM Router | `F:\code\MLLM-Tool\code\inference.py` | `cd F:\code\MLLM-Tool\code` then `python inference.py` | Legacy Vicuna/ImageBind inference only; no scorer |
| ToolVQA step | `F:\code\ToolVQA-release\opencompass` | `python run.py <generated-config> -p llmit -q auto --max-num-workers 32 --debug --eval_mode toolvista` | Evaluator ready; media remapping required |
| ToolVQA end-to-end | Same as step plus `F:\code\ToolVQA-release\agentlego` | Start `agentlego-server`, then run OpenCompass | Media, tool dependencies, and credentials required |
| Visual-RFT classification | `F:\code\Visual-RFT\classification\Qwen2_VL_classification_infere.py` | `cd F:\code\Visual-RFT\classification` then `python Qwen2_VL_classification_infere.py` | Legacy hard-coded Qwen2-VL paths; official images missing |
| Visual-RFT COCO | `F:\code\Visual-RFT\coco_evaluation` | Legacy prediction script followed by upstream COCO evaluator/notebook | COCO images and annotations missing |
| Visual-RFT LISA | `F:\code\Visual-RFT\lisa_evaluation` | `bash Qwen2_VL_lisa_infere.sh`, then `box2mask.py` and `mask_iou.py` | LISA and SAM data/models missing; paths require bridge injection |
| MAT-Coding | `F:\code\Visual-RFT\Visual-ARFT\evaluation_coding` | `python evaluation_mat_coding_ngpu_7b_df.py`, then run `evaluation.ipynb` | Legacy paths/classes hard-coded; prefer VLMEvalKit |
| MAT-Search | `F:\code\Visual-RFT\Visual-ARFT\evaluation_search\MAT-Search-Benchmark` | `python evaluation_mat_search_ngpu_7b_df.py`, then run `evaluation.ipynb` | MAT media and search environment required; prefer VLMEvalKit |
| General multimodal | `F:\code\ms-swift` | `swift eval ... --eval_backend VLMEvalKit` | Ready after installing `ms-swift[eval]` and benchmark dependencies |
| BFCL v3 | External official BFCL repository | Official BFCL generation and evaluation commands | Not installed locally |

After the corresponding implementation phase, all rows are launched through the target interface:

```bash
python scripts/evaluate.py \
  --checkpoint <checkpoint-path> \
  --benchmark <benchmark-id> \
  --profile light
```

The orchestrator records the actual expanded upstream command in `task.json`; reports must never show only the wrapper command.

## 11. Execution Modes

`command` generates configuration and invokes an upstream CLI. It is used for ms-swift/EvalScope, VLMEvalKit, ToolVQA OpenCompass, BFCL, and MAT-Bench.

`python_callable` imports a model-independent upstream scorer and supplies upstream-compatible predictions. It is used for Visual-RFT COCO and LISA scoring.

`toolvista_native` is permitted only when upstream scoring is absent. Initial native scorers are xLAM function calling, MLLM Router, and retention/forgetting/efficiency aggregation.

## 12. Registry and Suite Contracts

Every benchmark registry entry declares benchmark ID/version, upstream project/root/commit, working directory, execution mode, runtime type, data manifest/hash, converter, command or callable, parser, metric mapping/scaling, environment, resources, timeout/retry policy, services, and credentials.

Suites describe what to evaluate, not how. The registry determines conversion, execution, scoring, and parsing.

```yaml
name: c4_full
benchmarks:
  - id: xlam_function_calling
    profile: full
  - id: mllm_router
    profile: full
  - id: toolvqa_step
    profile: full
  - id: toolvqa_e2e
    profile: full
  - id: realworldqa
    profile: full
  - id: mmmu
    profile: full
  - id: mathvision
    profile: full
  - id: ocrbench
    profile: full
  - id: bfcl_v3
    profile: full
```

## 13. Unified Command

The following is the target ToolVista CLI contract. It does not exist in the current repository until Phase 1 is implemented.

```bash
python scripts/evaluate.py \
  --checkpoint /root/autodl-tmp/checkpoints/c4_unified_sft \
  --suite c4_full \
  --run-id c4_unified_sft_seed42 \
  --mode full
```

Focused operations support one benchmark, `--predict-only`, `--score-only`, and `--resume`.

## 14. Data Flow

```text
load suite and registry
  -> create evaluation manifest
  -> validate checkpoint, data, media, commits, dependencies, and credentials
  -> create execution plan
  -> start or reuse model and tool services
  -> generate or reuse predictions
  -> convert predictions to upstream format
  -> invoke upstream evaluator or minimal local scorer
  -> preserve raw upstream output
  -> parse ToolVista result envelope
  -> validate coverage and comparability
  -> compute retention, forgetting, and efficiency
  -> atomically write local artifacts
  -> upload completed local results to W&B
```

## 15. Model Runtime

OpenAI-compatible API mode serves xLAM, MLLM Router, ToolVQA, BFCL, and compatible upstream evaluators. The bridge starts one `swift deploy` service, waits for health, and shares it across compatible benchmarks.

Direct checkpoint loading is used by `swift eval`, native VLMEvalKit integrations, Visual-RFT grounding/detection, and MAT evaluators that need model internals or a local processor.

## 16. Frozen Inference Configuration

The comparability contract includes chat template, system prompt, temperature, top-p/top-k, maximum tokens, image preprocessing, tool order/schema, stop words, seed, timeout, retry policy, concurrency, and reasoning visibility. Changing any value invalidates the prediction cache.

## 17. Prediction, Scoring, and Cache

Each specialty benchmark separates prediction from scoring:

```text
predict: manifest -> predictions.jsonl or trajectories.jsonl
score: predictions -> upstream evaluator -> upstream result
```

The cache key includes checkpoint fingerprint, benchmark manifest hash, prompt/template hash, generation configuration hash, media preprocessing hash, tool-schema hash, bridge version, and upstream commit.

Changing the checkpoint, prompt, generation settings, data, or media preprocessing requires new predictions. Changing only the result parser or upstream evaluator permits prediction reuse but requires re-scoring. Changing only W&B upload logic requires neither.

Every result records `fresh_prediction`, `cached_prediction`, or `rescored_prediction`.

## 18. Tool Execution Isolation

### 18.1 ToolVQA

`every_with_gt` uses ground-truth historical context, does not start a real tool server, measures per-step behavior, and is the default lightweight evaluation.

`every` starts the AgentLego tool server, lets the model complete the trajectory, and is used at major checkpoints.

### 18.2 MAT-Coding

Each sample executes generated code in an isolated temporary directory. Preserve generated code, stdout, stderr, exit status, timeout status, generated media, and final answer. Apply explicit CPU, memory, disk, and wall-clock limits.

A model-generated code timeout is a task result. A bridge process crash is an infrastructure failure.

### 18.3 MAT-Search

Support two modes:

```text
frozen
  cache search responses by normalized query
  use identical responses across algorithms and seeds
  default for formal RL comparison

live
  query the current search service
  use for final real-environment evaluation
```

Frozen and live scores remain separate and never appear in the same primary comparison table.

## 19. Resource Scheduling

Each benchmark declares GPU count, expected GPU memory, CPU count, and whether GPU access is exclusive.

First-version scheduling rules are:

1. Direct-load GPU benchmarks run sequentially unless explicitly shareable.
2. API benchmarks share one `swift deploy` service and use bounded concurrency.
3. CPU-only scorers may run in parallel.
4. Tool servers have explicit start, health-check, and stop lifecycles.
5. Training distribution remains the responsibility of ms-swift and `torchrun`.

The first version is a single-machine AutoDL orchestrator, not a cluster scheduler.

## 20. Run Artifacts

```text
runs/<run_id>/
|-- manifest.json
|-- plan.json
|-- services.json
|-- train_metrics.jsonl
|-- eval/
|   `-- <benchmark_id>/
|       |-- task.json
|       |-- predictions.jsonl
|       |-- failures.jsonl
|       |-- upstream/
|       |-- result.json
|       |-- stdout.log
|       `-- stderr.log
|-- artifacts/
|-- summary.json
`-- wandb.json
```

Raw upstream output is never deleted after parsing.

`manifest.json` records run/checkpoint IDs, parent checkpoint, stage, algorithm, seed, ToolVista commit, ms-swift and upstream commits, base-model revision, dataset versions and hashes, training-token budget, replay ratios, inference settings, hardware/software environment, and timestamps.

## 21. Result Protocol

Each benchmark writes `result.json`:

```json
{
  "schema_version": "1.0",
  "run_id": "c4_unified_sft_seed42",
  "checkpoint_id": "c4_unified_sft",
  "benchmark": "toolvqa_step",
  "benchmark_version": "toolvqa@99caf3a5",
  "profile": "light",
  "status": "passed",
  "prediction_provenance": "fresh_prediction",
  "comparability_key": "sha256:<computed-digest>",
  "coverage": {
    "planned": 200,
    "attempted": 200,
    "completed": 200,
    "infrastructure_failed": 0,
    "invalid_predictions": 7,
    "ratio": 1.0
  },
  "upstream_metrics": {
    "tool_acc": 81.2,
    "arg_acc": 73.6
  },
  "metrics": {
    "toolvqa/tool_acc": 0.812,
    "toolvqa/arg_acc": 0.736
  },
  "artifacts": {
    "predictions": "predictions.jsonl",
    "failures": "failures.jsonl",
    "upstream_result": "upstream/result.json",
    "stdout": "stdout.log",
    "stderr": "stderr.log"
  }
}
```

`upstream_metrics` preserves upstream names and scales. `metrics` contains explicitly mapped ToolVista names. Scale conversion, such as percent-to-ratio, is declared in the registry and never guessed by a parser.

## 22. Coverage Semantics

An invalid model prediction counts as a completed sample and receives the benchmark-defined failure score, normally zero. It does not lower coverage.

Coverage decreases only for infrastructure failures such as an unfinished request, unreadable media, unavailable tool service, or an upstream process losing the sample.

Formal reports require 100% coverage. Lightweight trend reports require at least 98% infrastructure coverage and may compare checkpoints only on an identical completed sample-ID set.

## 23. Benchmark Status

```text
pending
running
passed
partial
failed
blocked
skipped
```

- `passed`: all planned samples completed and the result schema validated.
- `partial`: scoring succeeded but infrastructure failures reduced coverage.
- `failed`: prediction, evaluator, parsing, or result validation failed.
- `blocked`: a required checkpoint, dataset, media file, dependency, credential, or service is unavailable.
- `skipped`: the suite explicitly excluded the benchmark.

Independent benchmarks continue after another benchmark fails. Suite status is the most severe member status.

## 24. Failure and Retry Policy

Only transient infrastructure failures are retried:

```text
API requests: up to 3 retries with 2s, 8s, and 30s delays
tool calls: up to 2 retries for transport failure only
model service: one normal start and one retry on a new port
W&B upload: up to 5 retries, then queue for later upload
upstream evaluator process: no automatic restart by default
```

Invalid JSON, unknown tool names, invalid arguments, and incorrect answers are model results and are never retried.

Configuration and data failures block execution during preflight. Evaluator crashes produce `failed`, preserve stdout and stderr, and do not create a synthetic score. W&B failure never changes a locally completed benchmark result.

Final JSON and JSONL indexes are written to temporary files and atomically replaced.

## 25. Resume Rules

```text
valid result.json exists
  -> skip benchmark

complete predictions exist
  -> score only

partial predictions exist
  -> generate missing sample IDs, then score

only raw upstream result exists
  -> parse again

only W&B upload is missing
  -> upload only

cache or comparability key differs
  -> reject reuse and generate fresh predictions
```

## 26. Comparability Contract

Formal checkpoint comparison requires equality of benchmark ID/version, manifest hash, sample-ID set, prompt/template hash, generation configuration, media preprocessing, tool schema/order, upstream evaluator version, ToolVista parser version, and required coverage.

If any requirement differs, derived comparison metrics are `null` with an explicit reason. They are never replaced with zero.

## 27. Data Leakage Prevention

The split policy is:

```text
official train -> training only
official validation -> tuning and checkpoint selection
official test -> final evaluation only
```

When no official split exists, ToolVista creates a stable hash-based train/dev/test partition.

Every evaluation manifest stores manifest version, source project/commit/split, stable sample IDs, normalized text hashes, media hashes, tool-schema hashes, and a manifest hash.

Cross-source deduplication checks stable IDs, normalized text, media hashes, and tool/answer content. Test solutions, tool outputs, and reward labels are not exposed to training prompts.

## 28. Checkpoint Selection

Checkpoint selection does not use training loss alone and does not select directly from online RL reward.

Selection order:

1. Enforce historical capability and general-retention constraints on validation manifests.
2. Among eligible checkpoints, maximize the current stage's primary validation metric.
3. Use test manifests only for final reporting.

Initial configurable constraints are:

```text
general_retention >= 0.95
xlam_forgetting <= 0.03
toolvqa_forgetting <= 0.03
mllm_router_forgetting <= 0.03
```

## 29. Derived Metrics

```text
new_gain = current_new_score - previous_checkpoint_new_score
gain_vs_base = current_score - base_score
forgetting = historical_best_old_score - current_old_score
retention = current_general_score / base_general_score
gain_per_million_tokens = gain / consumed_training_tokens_in_millions
gain_per_gpu_hour = gain / consumed_gpu_hours
reward_per_generated_token = reward_sum / generated_token_count
success_per_1k_output_tokens = successes / output_tokens * 1000
```

Each registry entry declares metric direction and its primary metric. The aggregator does not average unrelated metrics into an opaque score unless a named, versioned composite is explicitly defined.

## 30. RL A/B Protocol

Smoke runs use GRPO, BNPO, and DAPO with seed 42. They validate data loading, rollout, reward registration, gradient updates, logs, checkpoint saving, evaluation, and resume, but do not rank algorithms.

Formal runs use all three algorithms with seeds 42, 1234, and 2026, for nine total runs.

Every formal run uses the same C4 checkpoint, Visual-RFT manifest, reward functions, prompt/template, generation parameters, effective training/generated-token budget, evaluation manifests, and seed set.

Compare results against training step, cumulative generated tokens, and cumulative GPU hours. Report every seed and `mean +/- std`.

## 31. W&B and Local Storage

Local JSON and JSONL remain the source of truth. W&B provides training curves, reward distributions, checkpoint tables, algorithm/seed comparisons, retention/forgetting curves, efficiency metrics, artifact references, and manifest hashes.

Recommended W&B project:

```text
toolvista-qwen3vl
```

Groups are `c1-xlam-sft`, `c2-toolvqa-sft`, `c3-mllm-router-sft`, `c4-unified-sft`, `r1-grpo`, `r2-bnpo`, `r3-dapo`, and `r4-mat`.

Only locally written and validated results are uploaded. W&B can be rebuilt from local artifacts without rerunning inference.

## 32. Testing Strategy

### 32.1 Unit Tests

Cover registry validation, suite expansion, converters, parsers, cache/comparability keys, status and coverage calculation, metric scaling, and resume planning.

### 32.2 Contract Tests

Every upstream integration has a small frozen prediction fixture and expected upstream result. Contract tests validate the bridge without loading a model.

### 32.3 Evaluator Parity Tests

For the same frozen prediction, the upstream evaluator score must equal ToolVista `upstream_metrics`, except for a declared floating-point serialization tolerance.

### 32.4 Smoke Tests

Every benchmark runs 1 to 10 samples through preflight, prediction, conversion, scoring, result validation, and artifact writing.

### 32.5 Recovery Tests

Simulate API timeout, HTTP 429/5xx, partial predictions, evaluator crash, missing result files, W&B failure, and tool-server failure.

### 32.6 Leakage and Cache Tests

Training/evaluation manifests must not overlap under the configured ID, text, media, schema, answer, and tool-output hash rules. Changing model, prompt, generation settings, data, preprocessing, tool schema, or evaluator version must invalidate the correct cache layer.

## 33. Implementation Phases

### Phase 1: Core and Immediately Usable Evaluation

```text
unified CLI
registry and suite loader
run manifest and artifact layout
command and native execution modes
swift eval / VLMEvalKit integration
xLAM scorer
MLLM Router scorer
result protocol
retention and forgetting aggregation
W&B upload
```

### Phase 2: ToolVQA Step Evaluation

```text
ToolVQA manifest preparation
dynamic OpenCompass configuration
OpenAI-compatible model adapter
GTABenchEvaluator result parser
every_with_gt parity and smoke tests
```

### Phase 3: Media and Tool Execution

```text
ToolVQA media download/remapping workflow
ToolVQA end-to-end AgentLego orchestration
Visual-RFT classification manifests
COCO and LISA scorer bridges
```

### Phase 4: External and Agentic Benchmarks

```text
official BFCL v3 integration
MAT-Bench through VLMEvalKit
MAT-Coding isolated execution
MAT-Search frozen and live environments
GRPO / BNPO / DAPO multi-seed aggregation
```

## 34. Acceptance Criteria

Phase 1 is accepted when:

1. One command evaluates a checkpoint with xLAM, MLLM Router, and a standard ms-swift/VLMEvalKit suite.
2. A stopped run resumes without regenerating completed predictions.
3. Model, prompt, data, and evaluator changes invalidate the correct cache layer.
4. Raw upstream artifacts and ToolVista result envelopes are both preserved.
5. W&B can be reconstructed from local artifacts without inference.
6. Retention and forgetting are computed only for comparable results.
7. Evaluator parity tests pass for every implemented bridge.

The complete system is accepted when every training stage runs cumulative lightweight regression; major checkpoints run comprehensive evaluation; ToolVQA, Visual-RFT, and MAT preserve upstream semantics; nine formal RL runs are aggregated per seed and as mean/std; and all formal rows have immutable manifests, required coverage, complete provenance, and valid comparability keys.

## 35. Final Boundary

ToolVista is the control plane for training-data conversion and evaluation orchestration. It is not a replacement for ms-swift, EvalScope, OpenCompass, VLMEvalKit, BFCL, AgentLego, COCO, LISA, Visual-RFT, or Visual-ARFT.

The implementation preference is:

```text
reuse upstream command
  -> import upstream scorer
  -> add a minimal ToolVista scorer only when upstream scoring is absent
```

This boundary keeps results comparable to the original projects while giving ToolVista one reproducible workflow for staged SFT, RL algorithm comparison, regression detection, and experiment reporting.
