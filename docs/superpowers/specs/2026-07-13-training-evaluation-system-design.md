# ToolVista Training and Evaluation System Design / ToolVista 训练与评测系统设计

Date: 2026-07-13

日期：2026-07-13

Status: Approved design, pending implementation

状态：设计已确认，等待实施

## 1. Purpose / 目的

ToolVista needs a reproducible training and evaluation workflow for Qwen3-VL that quantifies capability gains, regressions, forgetting, and efficiency across staged SFT and reinforcement-learning experiments.

中文翻译：ToolVista 需要为 Qwen3-VL 建立一套可复现的训练与评测流程，用来量化分阶段 SFT 和强化学习实验中的能力提升、能力回退、遗忘程度以及训练效率。

The planned progression is:

中文翻译：计划采用以下训练演进路线：

```text
Qwen3-VL base
  -> xLAM SFT
  -> ToolVQA SFT with xLAM replay
  -> MLLM-Tool Router SFT with historical replay
  -> unified SFT checkpoint
  -> GRPO / BNPO / DAPO comparison on common Visual-RFT data
  -> MAT agentic RL
```

```text
Qwen3-VL 基座模型
  -> xLAM SFT
  -> ToolVQA SFT，并回放 xLAM 历史数据
  -> MLLM-Tool Router SFT，并回放历史数据
  -> 统一 SFT checkpoint
  -> 在同一份 Visual-RFT 数据上对比 GRPO / BNPO / DAPO
  -> MAT Agent 强化学习
```

The evaluation system must maximize reuse of upstream datasets, evaluators, scorers, tool environments, and metric definitions. ToolVista provides a thin conversion and orchestration layer rather than replacing mature upstream frameworks.

中文翻译：评测系统应最大程度复用上游数据集、evaluator、scorer、工具执行环境和指标定义。ToolVista 只提供轻量的格式转换与流程编排层，不替代已经成熟的上游评测框架。

## 2. Goals / 目标

The system must:

中文翻译：系统必须满足以下要求：

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

中文翻译：

1. 每个训练阶段以及每个需要审查的 RL 运行结束后，都执行轻量级累计回归测试。
2. 在关键 checkpoint 上执行完整评测。
3. 复用上游 evaluator，而不是将其代码复制进 ToolVista。
4. 提供一个统一的 ToolVista 命令，用于规划、执行、恢复、评分、聚合和上传整套评测。
5. 保留上游原始预测、轨迹、日志、结果、指标名称和指标量纲。
6. 生成统一的结果外层协议，用于跨 checkpoint 对比。
7. 跟踪能力保持率、遗忘程度、训练 token 效率、生成 token 效率和 GPU 小时效率。
8. 将本地 JSON/JSONL 作为事实来源，将 W&B 作为在线展示与对比界面。
9. 通过不可变 manifest 和稳定哈希防止训练集与评测集泄漏。
10. 确保 GRPO、BNPO 和 DAPO 的对比公平。

## 3. Non-Goals / 非目标

The first implementation will not:

中文翻译：第一版实现不负责以下内容：

1. Copy upstream repositories into ToolVista.
2. Rewrite OpenCompass, VLMEvalKit, EvalScope, BFCL, COCO, LISA, or MAT evaluators.
3. Install all upstream projects into one Python environment.
4. Build a multi-node cluster scheduler.
5. Automatically download every large benchmark and media archive.
6. Select checkpoints from training loss or online RL reward alone.
7. Put evaluation logic in ToolVista dataset adapters or exporters.

中文翻译：

1. 不把上游仓库复制进 ToolVista。
2. 不重写 OpenCompass、VLMEvalKit、EvalScope、BFCL、COCO、LISA 或 MAT evaluator。
3. 不强制把所有上游项目安装到同一个 Python 环境。
4. 不构建多节点集群调度器。
5. 不自动下载所有大型 benchmark 和媒体压缩包。
6. 不只根据训练 loss 或在线 RL reward 选择 checkpoint。
7. 不把评测逻辑放入 ToolVista 的数据集 adapter 或 exporter。

The existing training-data path remains separate:

中文翻译：现有训练数据转换链路继续保持独立：

```text
raw dataset -> adapter -> CanonicalSample -> exporter -> training JSONL
```

```text
原始数据集 -> adapter -> CanonicalSample -> exporter -> 训练 JSONL
```

The evaluation path is:

中文翻译：新的评测链路如下：

```text
checkpoint or model API
  -> ToolVista evaluation bridge
  -> upstream-compatible prediction format
  -> upstream evaluator or minimal ToolVista scorer
  -> raw upstream result
  -> ToolVista result envelope
  -> local reports and W&B
```

```text
checkpoint 或模型 API
  -> ToolVista 评测桥接器
  -> 与上游兼容的预测格式
  -> 上游 evaluator 或最小化 ToolVista scorer
  -> 上游原始结果
  -> ToolVista 统一结果协议
  -> 本地报告和 W&B
```

## 4. Thin Evaluation Bridge / 轻量评测桥接层

Upstream projects retain ownership of:

中文翻译：下列内容继续由上游项目负责：

- test data and split semantics;
- model-independent evaluator and scorer behavior;
- executable tool environments where required;
- metric definitions and original scales;
- benchmark-specific prediction formats.

中文翻译：

- 测试数据及其 split 语义；
- 与具体模型无关的 evaluator 和 scorer 行为；
- 需要时使用的可执行工具环境；
- 指标定义和原始指标量纲；
- benchmark 专用的预测格式。

ToolVista owns:

中文翻译：ToolVista 负责以下内容：

- checkpoint and API serving orchestration;
- test-manifest preparation and validation;
- prediction conversion into upstream formats;
- command construction and process lifecycle;
- parsing upstream outputs without changing metric semantics;
- a common result envelope and metric namespace;
- caching, resume, coverage, and failure accounting;
- retention, forgetting, efficiency, and multi-seed aggregation;
- local artifact storage and W&B upload.

中文翻译：

- checkpoint 和 API 服务编排；
- 测试 manifest 的准备与校验；
- 将预测转换成上游要求的格式；
- 构造命令并管理进程生命周期；
- 在不改变指标语义的前提下解析上游输出；
- 统一结果协议和指标命名空间；
- 缓存、断点恢复、coverage 和失败统计；
- retention、forgetting、效率和多 seed 聚合；
- 本地 artifact 保存与 W&B 上传。

The system has one command-line entry point, but it is not one large Python file. Benchmark-specific converters and parsers remain small, independently tested modules behind a registry.

中文翻译：系统只有一个统一命令行入口，但实现上不能写成一个巨型 Python 文件。不同 benchmark 的 converter 和 parser 应保持为 registry 后方的小型独立模块，并分别进行测试。

## 5. Training Topology / 训练拓扑

Replay ratios are measured by effective supervised tokens, not sample count.

中文翻译：replay 比例按照实际参与监督损失的有效 token 数计算，而不是按照样本条数计算。

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

中文对照：

| 阶段 | 父 checkpoint | 训练数据 | 初始 token 配比 | 必须执行的累计回测 |
|---|---|---|---:|---|
| C0 Base | Qwen3-VL | 无 | 无 | 所有当前可用基线测试 |
| C1 xLAM SFT | C0 | xLAM | 100% xLAM | xLAM 和通用能力保持 |
| C2 ToolVQA SFT | C1 | ToolVQA 加 replay | 70% ToolVQA，30% xLAM | xLAM、ToolVQA、通用能力保持 |
| C3 MLLM Router SFT | C2 | MLLM 加 replay | 60% MLLM，20% ToolVQA，20% xLAM | xLAM、ToolVQA、MLLM、通用能力保持 |
| C4 Unified SFT | C3 | 三种 SFT 数据源 | 35% xLAM，35% ToolVQA，30% MLLM | 完整 SFT 能力测试套件 |
| R1 GRPO | C4 | Visual-RFT | 共同 RL 数据 | 所有 SFT 测试、Visual-RFT、通用能力保持 |
| R2 BNPO | C4 | Visual-RFT | 与 R1 相同 | 与 R1 完全相同的评测 |
| R3 DAPO | C4 | Visual-RFT | 与 R1 相同 | 与 R1 完全相同的评测 |
| R4 MAT RL | 选定的 RL checkpoint | MAT 加 replay | 正式实验前冻结 | 所有历史测试和 MAT |

Every training run records planned and actual token ratios, source sample counts, supervised token counts, text and visual token counts, truncation counts, and total consumed training tokens. C4 is the single shared starting checkpoint for GRPO, BNPO, and DAPO.

中文翻译：每次训练都必须记录计划 token 比例、实际 token 比例、各来源样本数、监督 token 数、文本和视觉 token 数、截断样本/token 数，以及总消耗训练 token。GRPO、BNPO 和 DAPO 必须共同从同一个 C4 checkpoint 开始。

## 6. Evaluation Cadence / 评测频率

### 6.1 Online Metrics / 在线训练指标

Record loss, learning rate, gradient norm, throughput, GPU memory, reward components, KL, completion length, truncation rate, and generated tokens at every available logging step.

中文翻译：在每个可用 logging step 记录 loss、学习率、梯度范数、吞吐量、GPU 显存、各 reward 分量、KL、completion 长度、截断率以及生成 token 数。

### 6.2 Lightweight Cumulative Regression / 轻量累计回归测试

Run after every SFT stage and after each reviewed RL run or checkpoint. Each specialty suite uses a frozen manifest of approximately 100 to 500 samples where upstream data permits. It always covers the new capability, all historical capabilities, and general retention.

中文翻译：每个 SFT 阶段结束后，以及每个需要审查的 RL 运行或 checkpoint 之后，都执行轻量累计回测。在上游数据允许时，每个专项测试使用约 100 到 500 条样本组成的冻结 manifest。测试必须同时覆盖新能力、全部历史能力和通用能力保持情况。

### 6.3 Comprehensive Evaluation / 完整评测

Run full specialty and core evaluation at C0, C4, all nine final GRPO/BNPO/DAPO checkpoints, one representative checkpoint per RL algorithm, and the final MAT checkpoint.

中文翻译：在 C0、C4、九个正式 GRPO/BNPO/DAPO 最终 checkpoint、每种 RL 算法的一个代表 checkpoint，以及最终 MAT checkpoint 上执行完整专项与核心评测。

Expensive general suites run at C0, C4, representative GRPO, representative BNPO, representative DAPO, and the final model. VideoMME is comprehensive-only.

中文翻译：成本较高的通用评测只在 C0、C4、代表性的 GRPO/BNPO/DAPO checkpoint 和最终模型上运行。VideoMME 只属于完整评测，不进入轻量回测。

## 7. Required Regression by Checkpoint / 各 Checkpoint 必须回测的能力

| Checkpoint | Required specialty regression |
|---|---|
| C1 xLAM SFT | xLAM function-calling protocol |
| C2 ToolVQA SFT | xLAM and ToolVQA |
| C3 MLLM Router SFT | xLAM, ToolVQA, and MLLM-Tool Router |
| C4 Unified SFT | all three SFT suites |
| R1/R2/R3 RL | all SFT suites and Visual-RFT |
| R4 MAT RL | all previous suites and MAT |

中文对照：

| Checkpoint | 必须执行的专项回测 |
|---|---|
| C1 xLAM SFT | xLAM 函数调用协议 |
| C2 ToolVQA SFT | xLAM 和 ToolVQA |
| C3 MLLM Router SFT | xLAM、ToolVQA 和 MLLM-Tool Router |
| C4 Unified SFT | 全部三种 SFT 专项测试 |
| R1/R2/R3 RL | 全部 SFT 测试和 Visual-RFT |
| R4 MAT RL | 全部历史测试和 MAT |

## 8. Metrics / 指标

### 8.1 xLAM / xLAM 指标

JSON validity, tool-call schema validity, function-name accuracy, argument-key precision/recall/F1, argument-value accuracy, argument exact match, parallel-call exact match, and trajectory success where labels exist.

中文翻译：JSON 合法率、工具调用 schema 合法率、函数名称准确率、参数键 precision/recall/F1、参数值准确率、参数整体精确匹配率、并行工具调用精确匹配率，以及存在轨迹标签时的轨迹成功率。

### 8.2 ToolVQA / ToolVQA 指标

`every_with_gt` reports instruction alignment, tool accuracy, argument accuracy, answer accuracy, tool-call count, and tool-call error count.

中文翻译：`every_with_gt` 模式报告指令对齐率、工具准确率、参数准确率、答案准确率、工具调用次数和工具调用错误次数。

`every` reports answer accuracy, tool-call count, tool-call error count, and perception/operation/logic/creativity F1.

中文翻译：`every` 模式报告答案准确率、工具调用次数、工具调用错误次数，以及感知类、操作类、逻辑类和创造类工具的 F1。

### 8.3 MLLM-Tool Router / MLLM-Tool 路由指标

Output-format validity, top-1 accuracy, top-k accuracy, valid candidate-set hit rate, macro F1, per-modality accuracy, confusion matrix, and invalid-route rate. For multi-label samples, any listed valid tool is correct.

中文翻译：输出格式合法率、top-1 准确率、top-k 准确率、合法候选工具集合命中率、macro F1、各模态准确率、混淆矩阵和非法路由率。对于多标签样本，只要预测命中任意一个列出的合法工具，就判定为正确。

### 8.4 Visual-RFT / Visual-RFT 指标

Format accuracy, mean IoU, box precision/recall, matched-box IoU, no-object accuracy, COCO mAP/AP50/AP75/size AP, LISA global and cumulative IoU, fine-grained classification accuracy, reward distribution, zero-reward rate, and completion length.

中文翻译：格式准确率、平均 IoU、边界框 precision/recall、匹配框 IoU、无目标样本准确率、COCO mAP/AP50/AP75/不同尺寸 AP、LISA 全局 IoU 和累计 IoU、细粒度分类准确率、reward 分布、零 reward 比例和 completion 长度。

### 8.5 MAT and BFCL / MAT 与 BFCL 指标

MAT-Coding reports simple, hard, and overall F1. MAT-Search reports simple, hard, and overall F1 and exact match, plus tool count, exceptions, media validity, and latency. BFCL v3 reports AST accuracy, executable accuracy, relevance detection, and multi-turn accuracy.

中文翻译：MAT-Coding 报告简单样本、困难样本和总体 F1；MAT-Search 报告简单、困难和总体的 F1 与精确匹配率，并记录工具调用次数、异常率、媒体有效性和延迟。BFCL v3 报告 AST 准确率、可执行准确率、相关性检测和多轮准确率。

## 9. General Retention / 通用能力保持

The lightweight suite uses RealWorldQA, MMMU, MathVision Mini, OCRBench, GSM8K, IFEval, and BFCL subsets.

中文翻译：轻量通用能力测试使用 RealWorldQA、MMMU、MathVision Mini、OCRBench、GSM8K、IFEval 和 BFCL 的冻结子集。

The comprehensive suite may include MMBench, MMStar, MMVet, OCRBench, DocVQA, ChartQA, MathVista, HallusionBench, MMLU-Pro, GSM8K, IFEval, BFCL v3, and VideoMME.

中文翻译：完整通用评测可包括 MMBench、MMStar、MMVet、OCRBench、DocVQA、ChartQA、MathVista、HallusionBench、MMLU-Pro、GSM8K、IFEval、BFCL v3 和 VideoMME。

Qwen3-VL official reproduction code is the preferred reference for MMMU, MathVision, ODinW-13, RealWorldQA, and VideoMME when available. ms-swift's EvalScope, OpenCompass, and VLMEvalKit backends are the default execution path for standard suites.

中文翻译：如果部署环境中具备 Qwen3-VL 官方复现代码，则 MMMU、MathVision、ODinW-13、RealWorldQA 和 VideoMME 优先参考官方实现。标准测试默认通过 ms-swift 已封装的 EvalScope、OpenCompass 和 VLMEvalKit 后端执行。

## 10. Upstream Inventory / 上游项目盘点

### 10.1 ms-swift / ms-swift 评测能力

```text
Repository: F:\code\ms-swift
Commit: 0a4f08e68d936c7d5d803c687b26f9d103dd0161
Documentation: docs/source/Instruction/Evaluation.md
Execution: command
Primary command: swift eval
Backends: Native, OpenCompass, VLMEvalKit
```

中文翻译：本地 ms-swift 仓库位于 `F:\code\ms-swift`，盘点时 commit 为 `0a4f08e...`。正式评测文档位于 `docs/source/Instruction/Evaluation.md`，执行方式是上游命令 `swift eval`，可使用 Native、OpenCompass 和 VLMEvalKit 三种后端。

Light evaluation sets `--eval_limit`; full evaluation uses the same frozen generation configuration without a limit.

中文翻译：轻量评测通过 `--eval_limit` 限制样本量；完整评测使用完全相同的冻结生成配置，但取消样本量限制。

Upstream command template:

中文翻译：上游命令模板如下，代码和参数名保持英文原样：

```bash
swift eval \
  --model /root/autodl-tmp/checkpoints/<checkpoint> \
  --eval_backend VLMEvalKit \
  --infer_backend vllm \
  --eval_dataset RealWorldQA MathVision_MINI MMMU_DEV_VAL OCRBench \
  --eval_limit 200 \
  --eval_generation_config '{"max_tokens":512,"temperature":0}'
```

### 10.2 xLAM / xLAM 项目

```text
Repository: F:\code\xLAM
Commit: a88aa3aeddbc2d7d6aa7a87687d1a085c34e2aec
Training data: actionstudio/datasets/unified_data/actionstudio-98k
Upstream benchmark reference: BFCL v3
Repository-local held-out scorer: absent
Execution: toolvista_native
```

中文翻译：本地 xLAM 仓库位于 `F:\code\xLAM`，训练数据位于 `actionstudio/datasets/unified_data/actionstudio-98k`。xLAM 官方使用 BFCL v3 展示函数调用能力，但仓库内没有针对本地 held-out 数据的标准 scorer，因此当前专项测试采用 `toolvista_native` 模式。

ToolVista implements only the minimal held-out function-call scorer. Official BFCL is integrated separately when installed.

中文翻译：ToolVista 只实现最小化的 held-out 函数调用 scorer；安装官方 BFCL 后，再将 BFCL 作为独立的上游命令集成。

### 10.3 MLLM-Tool / MLLM-Tool 项目

```text
Repository: F:\code\MLLM-Tool
Commit: 8b74aad63ac511f6cbd7a74fb12aceee33975cf8
Legacy inference: code/inference.py
Evaluation candidate: data/IT_data_ins/T+X-T_data/combined_data.json
Repository-local scorer: absent
Execution: toolvista_native
```

中文翻译：本地 MLLM-Tool 仓库位于 `F:\code\MLLM-Tool`。其旧推理入口是 `code/inference.py`，可用于冻结 Router 评测的候选数据是 `combined_data.json`，但仓库没有 scorer，因此使用 `toolvista_native` 模式。

The legacy script is tied to OpenLLaMA/Vicuna, ImageBind, and hard-coded paths and is not used for Qwen3-VL inference. `combined_data.json` contains 2,397 rows and 2,350 unique question-label pairs. An exact question-label comparison found no overlap with the four local source files. The final manifest must still apply normalized text and media-hash deduplication.

中文翻译：旧脚本绑定 OpenLLaMA/Vicuna、ImageBind 和硬编码路径，不用于 Qwen3-VL 推理。`combined_data.json` 共 2397 条，其中 2350 个“问题-标签”组合唯一；按问题和标签精确比较，与本地四个来源文件没有重合。不过在冻结正式 manifest 前，仍然必须做标准化文本哈希和媒体哈希去重。

### 10.4 ToolVQA / ToolVQA 项目

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

中文翻译：本地 ToolVQA 仓库位于 `F:\code\ToolVQA-release`。评测通过仓库自带 OpenCompass 的 `run.py` 启动；数据集配置、数据加载器/scorer、测试标注和工具元数据路径如上，执行模式为 `command`，即调用上游命令。

The local OpenCompass dataset has 1,622 examples. All media references point to missing author-machine paths. The scorer and annotations are usable after media download and path remapping.

中文翻译：本地 OpenCompass 测试文件共有 1622 条样本，但所有媒体引用仍指向作者机器上的缺失路径。下载媒体并完成路径重映射后，现有标注和 scorer 可以继续使用。

ToolVista dynamically generates an OpenCompass config instead of using `configs/eval.py`, which contains author-machine API and tool-server addresses. Step evaluation uses `every_with_gt`; end-to-end evaluation uses `every` and starts AgentLego.

中文翻译：ToolVista 动态生成 OpenCompass 配置，不直接使用包含作者机器 API 地址和工具服务器地址的 `configs/eval.py`。逐步评测使用 `every_with_gt`；端到端评测使用 `every`，并启动 AgentLego 工具服务器。

Upstream command template after ToolVista generates the config:

中文翻译：ToolVista 生成配置后的上游启动命令模板如下：

```bash
cd /root/autodl-tmp/ToolVQA-release/opencompass
python run.py /root/autodl-tmp/ToolVista/runs/<run_id>/eval/toolvqa/generated_config.py \
  -p llmit \
  -q auto \
  --max-num-workers 32 \
  --debug \
  --eval_mode toolvista
```

### 10.5 Visual-RFT and Visual-ARFT / Visual-RFT 与 Visual-ARFT 项目

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

中文翻译：本地 Visual-RFT 仓库位于 `F:\code\Visual-RFT`。COCO scorer、LISA scorer 链、分类验证标注、MAT-Coding 和 MAT-Search 的本地路径如上。根据 evaluator 形式，使用 `python_callable` 或 `command` 模式复用。

The classification annotations contain 3,333 FGVC-Aircraft, 2,463 Oxford Flowers, 3,669 Oxford Pets, and 8,041 Stanford Cars validation rows. Their image paths are missing author-machine paths, so ToolVista must create a remapped immutable manifest after downloading official images.

中文翻译：分类标注包含 3333 条 FGVC-Aircraft、2463 条 Oxford Flowers、3669 条 Oxford Pets 和 8041 条 Stanford Cars 验证样本。标注中的图片路径指向作者机器且本地不存在，因此下载官方图片后，需要由 ToolVista 生成重映射且不可变的 manifest。

COCO, LVIS, and LISA require external image and annotation packages. For MAT, prefer current VLMEvalKit MAT-Bench support; the repository prediction formats and notebook formulas are compatibility fallbacks because the old scripts hard-code Qwen2.5-VL classes and paths.

中文翻译：COCO、LVIS 和 LISA 还需要外部图片与标注包。MAT 优先使用当前 VLMEvalKit 的 MAT-Bench 支持；仓库原有预测格式和 notebook 评分公式只作为兼容后备，因为旧脚本硬编码了 Qwen2.5-VL 类和路径。

### 10.6 Specialty Evaluation Launch Matrix / 专项评测启动矩阵

The paths below are the exact local upstream locations inspected for this design. Commands marked `legacy` reproduce the upstream entry point but are not the final ToolVista interface. Angle-bracket values in command templates are explicit operator inputs.

中文翻译：下表路径是本次设计实际检查过的本地上游位置。标记为 `legacy` 的命令只能复现上游旧入口，并不是最终 ToolVista 接口。命令中的尖括号值表示需要由运行者或编排器显式提供的参数。

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

中文对照：

| 能力 | 上游位置 | 当前上游启动方式 | 当前可用状态 |
|---|---|---|---|
| xLAM 函数调用 | `F:\code\xLAM` | 仓库内没有 held-out evaluator | 需要 Phase 1 的 ToolVista scorer |
| MLLM Router | `F:\code\MLLM-Tool\code\inference.py` | 进入目录后运行 `python inference.py` | 仅支持旧 Vicuna/ImageBind 推理，没有 scorer |
| ToolVQA 逐步评测 | `F:\code\ToolVQA-release\opencompass` | 用动态配置运行 `python run.py ...` | evaluator 已具备，仍需媒体重映射 |
| ToolVQA 端到端 | 同上，并使用 `agentlego` | 启动 `agentlego-server` 后运行 OpenCompass | 需要媒体、工具依赖和外部凭据 |
| Visual-RFT 分类 | `classification/Qwen2_VL_classification_infere.py` | 运行旧分类推理脚本 | Qwen2-VL 路径硬编码，官方图片缺失 |
| Visual-RFT COCO | `coco_evaluation` | 旧预测脚本加上游 COCO evaluator/notebook | COCO 图片和标注缺失 |
| Visual-RFT LISA | `lisa_evaluation` | 推理后运行 `box2mask.py` 和 `mask_iou.py` | 缺 LISA、SAM 数据/模型，需要注入路径 |
| MAT-Coding | `Visual-ARFT/evaluation_coding` | 运行 Python 推理脚本，再运行 notebook | 旧路径和类硬编码，优先 VLMEvalKit |
| MAT-Search | `Visual-ARFT/evaluation_search/MAT-Search-Benchmark` | 运行 Python 推理脚本，再运行 notebook | 需要 MAT 媒体和搜索环境，优先 VLMEvalKit |
| 通用多模态 | `F:\code\ms-swift` | `swift eval ... --eval_backend VLMEvalKit` | 安装评测依赖后可直接使用 |
| BFCL v3 | 外部官方 BFCL 仓库 | 官方生成与评分命令 | 本地尚未安装 |

After the corresponding implementation phase, all rows are launched through the target interface:

中文翻译：完成相应实施阶段后，所有专项测试统一通过以下目标接口启动：

```bash
python scripts/evaluate.py \
  --checkpoint <checkpoint-path> \
  --benchmark <benchmark-id> \
  --profile light
```

The orchestrator records the actual expanded upstream command in `task.json`; reports must never show only the wrapper command.

中文翻译：编排器必须把实际展开后的上游命令记录到 `task.json`，报告中不能只显示 ToolVista 外层 wrapper 命令。

## 11. Execution Modes / 执行模式

`command` generates configuration and invokes an upstream CLI. It is used for ms-swift/EvalScope, VLMEvalKit, ToolVQA OpenCompass, BFCL, and MAT-Bench.

中文翻译：`command` 模式由 ToolVista 生成配置，然后调用上游 CLI。适用于 ms-swift/EvalScope、VLMEvalKit、ToolVQA OpenCompass、BFCL 和 MAT-Bench。

`python_callable` imports a model-independent upstream scorer and supplies upstream-compatible predictions. It is used for Visual-RFT COCO and LISA scoring.

中文翻译：`python_callable` 模式直接导入与模型无关的上游 scorer，并向它提供与上游格式兼容的预测。适用于 Visual-RFT 的 COCO 和 LISA 评分。

`toolvista_native` is permitted only when upstream scoring is absent. Initial native scorers are xLAM function calling, MLLM Router, and retention/forgetting/efficiency aggregation.

中文翻译：只有上游没有可用评分器时，才允许使用 `toolvista_native`。第一批原生 scorer 仅包括 xLAM 函数调用、MLLM Router，以及 retention/forgetting/efficiency 聚合器。

## 12. Registry and Suite Contracts / Registry 与 Suite 契约

Every benchmark registry entry declares benchmark ID/version, upstream project/root/commit, working directory, execution mode, runtime type, data manifest/hash, converter, command or callable, parser, metric mapping/scaling, environment, resources, timeout/retry policy, services, and credentials.

中文翻译：每个 benchmark registry 条目都必须声明 benchmark ID/版本、上游项目/根目录/commit、工作目录、执行模式、runtime 类型、数据 manifest/hash、converter、上游命令或 callable、parser、指标映射与缩放规则、环境、资源、超时/重试策略、所需服务和凭据。

Suites describe what to evaluate, not how. The registry determines conversion, execution, scoring, and parsing.

中文翻译：suite 只描述“评测什么”，不描述“如何评测”。具体转换、执行、评分和解析方式由 registry 决定。

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

## 13. Unified Command / 统一命令

The following is the target ToolVista CLI contract. It does not exist in the current repository until Phase 1 is implemented.

中文翻译：以下命令是目标 ToolVista CLI 契约。在 Phase 1 实现之前，当前仓库中还不存在这个命令。

```bash
python scripts/evaluate.py \
  --checkpoint /root/autodl-tmp/checkpoints/c4_unified_sft \
  --suite c4_full \
  --run-id c4_unified_sft_seed42 \
  --mode full
```

Focused operations support one benchmark, `--predict-only`, `--score-only`, and `--resume`.

中文翻译：统一命令还必须支持仅执行一个 benchmark，以及 `--predict-only`、`--score-only` 和 `--resume` 等聚焦操作。

## 14. Data Flow / 数据流

中文翻译：完整数据流如下；代码块中同时保留英文步骤，便于与实现中的函数和状态名称对应。

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

```text
读取 suite 和 registry
  -> 创建 evaluation manifest
  -> 校验 checkpoint、数据、媒体、commit、依赖和凭据
  -> 创建执行计划
  -> 启动或复用模型服务与工具服务
  -> 生成或复用预测
  -> 将预测转换成上游格式
  -> 调用上游 evaluator 或最小化本地 scorer
  -> 保留上游原始输出
  -> 解析成 ToolVista 统一结果协议
  -> 校验 coverage 和可比性
  -> 计算 retention、forgetting 和效率
  -> 原子写入本地 artifact
  -> 将已经完成的本地结果上传到 W&B
```

## 15. Model Runtime / 模型运行方式

OpenAI-compatible API mode serves xLAM, MLLM Router, ToolVQA, BFCL, and compatible upstream evaluators. The bridge starts one `swift deploy` service, waits for health, and shares it across compatible benchmarks.

中文翻译：OpenAI-compatible API 模式用于 xLAM、MLLM Router、ToolVQA、BFCL 和其他兼容的上游 evaluator。桥接器启动一个 `swift deploy` 服务，等待健康检查通过后，让兼容的 benchmark 共享该服务。

Direct checkpoint loading is used by `swift eval`, native VLMEvalKit integrations, Visual-RFT grounding/detection, and MAT evaluators that need model internals or a local processor.

中文翻译：`swift eval`、VLMEvalKit 原生集成、Visual-RFT grounding/detection，以及需要访问模型内部或本地 processor 的 MAT evaluator，使用直接加载 checkpoint 的方式。

## 16. Frozen Inference Configuration / 冻结的推理配置

The comparability contract includes chat template, system prompt, temperature, top-p/top-k, maximum tokens, image preprocessing, tool order/schema, stop words, seed, timeout, retry policy, concurrency, and reasoning visibility. Changing any value invalidates the prediction cache.

中文翻译：可比性契约包含 chat template、system prompt、temperature、top-p/top-k、最大生成 token、图片预处理、工具顺序/schema、stop words、seed、timeout、重试策略、并发数和 reasoning 可见性。任意一项发生变化，都必须使预测缓存失效。

## 17. Prediction, Scoring, and Cache / 预测、评分与缓存

Each specialty benchmark separates prediction from scoring:

中文翻译：每个专项 benchmark 都必须将预测与评分分离：

```text
predict: manifest -> predictions.jsonl or trajectories.jsonl
score: predictions -> upstream evaluator -> upstream result
```

```text
predict：manifest -> predictions.jsonl 或 trajectories.jsonl
score：predictions -> 上游 evaluator -> 上游结果
```

The cache key includes checkpoint fingerprint, benchmark manifest hash, prompt/template hash, generation configuration hash, media preprocessing hash, tool-schema hash, bridge version, and upstream commit.

中文翻译：缓存键包含 checkpoint 指纹、benchmark manifest hash、prompt/template hash、生成配置 hash、媒体预处理 hash、tool-schema hash、bridge 版本和上游 commit。

Changing the checkpoint, prompt, generation settings, data, or media preprocessing requires new predictions. Changing only the result parser or upstream evaluator permits prediction reuse but requires re-scoring. Changing only W&B upload logic requires neither.

中文翻译：checkpoint、prompt、生成设置、数据或媒体预处理发生变化时，必须重新生成预测。只修改结果 parser 或上游 evaluator 时可以复用预测，但必须重新评分。只修改 W&B 上传逻辑时，不需要重新预测或评分。

Every result records `fresh_prediction`, `cached_prediction`, or `rescored_prediction`.

中文翻译：每份结果都必须记录预测来源状态：`fresh_prediction` 表示本次新生成，`cached_prediction` 表示直接使用缓存，`rescored_prediction` 表示复用预测后重新评分。

## 18. Tool Execution Isolation / 工具执行隔离

### 18.1 ToolVQA / ToolVQA 隔离方式

`every_with_gt` uses ground-truth historical context, does not start a real tool server, measures per-step behavior, and is the default lightweight evaluation.

中文翻译：`every_with_gt` 使用真实历史轨迹作为上下文，不启动真实工具服务器，用于测量逐步行为，是默认的轻量评测模式。

`every` starts the AgentLego tool server, lets the model complete the trajectory, and is used at major checkpoints.

中文翻译：`every` 启动 AgentLego 工具服务器，让模型自主完成整条轨迹，只在关键 checkpoint 上运行。

### 18.2 MAT-Coding / MAT 代码执行隔离

Each sample executes generated code in an isolated temporary directory. Preserve generated code, stdout, stderr, exit status, timeout status, generated media, and final answer. Apply explicit CPU, memory, disk, and wall-clock limits.

中文翻译：每条样本都在独立临时目录中执行模型生成的代码。必须保存生成代码、stdout、stderr、退出状态、超时状态、生成媒体和最终答案，并设置明确的 CPU、内存、磁盘和运行时长限制。

A model-generated code timeout is a task result. A bridge process crash is an infrastructure failure.

中文翻译：模型生成代码发生超时属于模型任务结果；桥接器进程崩溃才属于基础设施失败。

### 18.3 MAT-Search / MAT 搜索环境

Support two modes:

中文翻译：支持两种搜索模式：

```text
frozen
  cache search responses by normalized query
  use identical responses across algorithms and seeds
  default for formal RL comparison

live
  query the current search service
  use for final real-environment evaluation
```

```text
frozen（冻结模式）
  按标准化 query 缓存搜索响应
  不同算法和 seed 使用完全相同的搜索结果
  作为正式 RL 对比的默认模式

live（实时模式）
  调用当前实时搜索服务
  仅用于最终真实环境能力评测
```

Frozen and live scores remain separate and never appear in the same primary comparison table.

中文翻译：frozen 和 live 的得分必须分开报告，不能混入同一张主对比表。

## 19. Resource Scheduling / 资源调度

Each benchmark declares GPU count, expected GPU memory, CPU count, and whether GPU access is exclusive.

中文翻译：每个 benchmark 都要声明 GPU 数量、预计显存、CPU 数量，以及是否需要独占 GPU。

First-version scheduling rules are:

中文翻译：第一版调度规则如下：

1. Direct-load GPU benchmarks run sequentially unless explicitly shareable.
2. API benchmarks share one `swift deploy` service and use bounded concurrency.
3. CPU-only scorers may run in parallel.
4. Tool servers have explicit start, health-check, and stop lifecycles.
5. Training distribution remains the responsibility of ms-swift and `torchrun`.

中文翻译：

1. 直接加载模型的 GPU benchmark 默认顺序执行，除非明确声明可以共享。
2. API benchmark 共享一个 `swift deploy` 服务，并使用有限并发。
3. 纯 CPU scorer 可以并行执行。
4. 工具服务器必须具备明确的启动、健康检查和停止生命周期。
5. 训练分布式调度仍然由 ms-swift 和 `torchrun` 负责。

The first version is a single-machine AutoDL orchestrator, not a cluster scheduler.

中文翻译：第一版只实现单机 AutoDL 编排器，不实现集群调度器。

## 20. Run Artifacts / 运行产物

中文翻译：每次运行的目录结构如下，文件名保持英文以便与代码字段一致：

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

中文翻译：上游原始输出在解析完成后也不能删除。

`manifest.json` records run/checkpoint IDs, parent checkpoint, stage, algorithm, seed, ToolVista commit, ms-swift and upstream commits, base-model revision, dataset versions and hashes, training-token budget, replay ratios, inference settings, hardware/software environment, and timestamps.

中文翻译：`manifest.json` 记录 run/checkpoint ID、父 checkpoint、训练阶段、算法、seed、ToolVista commit、ms-swift 和各上游 commit、基座模型 revision、数据集版本/hash、训练 token 预算、replay 比例、推理设置、软硬件环境和时间戳。

## 21. Result Protocol / 统一结果协议

Each benchmark writes `result.json`:

中文翻译：每个 benchmark 都写出一个 `result.json`，示例如下。JSON 字段名保持英文，避免实现和上游工具之间产生二义性。

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

中文翻译：`upstream_metrics` 保留上游原始指标名称和量纲；`metrics` 存放明确映射后的 ToolVista 指标名称。百分数转比例等量纲转换必须在 registry 中声明，parser 不能自行猜测。

主要字段含义：

- `schema_version`：ToolVista 结果协议版本。
- `run_id`：本次训练或评测运行 ID。
- `checkpoint_id`：被评测 checkpoint 的稳定标识。
- `benchmark` / `benchmark_version`：测试名称及其上游版本。
- `profile`：`light` 或 `full` 等评测规模。
- `status`：本项 benchmark 的运行状态。
- `prediction_provenance`：预测是新生成、缓存复用还是重新评分。
- `comparability_key`：判断不同 checkpoint 结果能否直接比较的哈希。
- `coverage`：计划、尝试、完成、基础设施失败和非法预测统计。
- `artifacts`：预测、失败记录、上游结果和日志文件路径。

## 22. Coverage Semantics / Coverage 语义

An invalid model prediction counts as a completed sample and receives the benchmark-defined failure score, normally zero. It does not lower coverage.

中文翻译：模型输出非法预测仍然算作已完成样本，并按照 benchmark 规则计失败分，通常为 0 分。它不会降低 coverage。

Coverage decreases only for infrastructure failures such as an unfinished request, unreadable media, unavailable tool service, or an upstream process losing the sample.

中文翻译：只有请求未完成、媒体无法读取、工具服务不可用、上游进程丢失样本等基础设施失败，才会降低 coverage。

Formal reports require 100% coverage. Lightweight trend reports require at least 98% infrastructure coverage and may compare checkpoints only on an identical completed sample-ID set.

中文翻译：正式报告要求 100% coverage。轻量趋势报告至少要求 98% 的基础设施 coverage，并且只有不同 checkpoint 的已完成 sample ID 集合完全一致时才允许比较。

## 23. Benchmark Status / Benchmark 状态

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

中文翻译：

- `pending`：等待执行。
- `running`：正在执行。
- `passed`：全部计划样本完成，且结果协议校验通过。
- `partial`：评分成功，但基础设施失败导致 coverage 不完整。
- `failed`：预测、evaluator、解析或结果校验失败。
- `blocked`：缺少 checkpoint、数据、媒体、依赖、凭据或服务。
- `skipped`：suite 明确跳过该 benchmark，并不表示测试通过。

Independent benchmarks continue after another benchmark fails. Suite status is the most severe member status.

中文翻译：某个 benchmark 失败后，其他相互独立的 benchmark 继续执行。suite 总状态取所有成员中最严重的状态。

## 24. Failure and Retry Policy / 失败与重试策略

Only transient infrastructure failures are retried:

中文翻译：只有瞬时基础设施故障允许重试：

```text
API requests: up to 3 retries with 2s, 8s, and 30s delays
tool calls: up to 2 retries for transport failure only
model service: one normal start and one retry on a new port
W&B upload: up to 5 retries, then queue for later upload
upstream evaluator process: no automatic restart by default
```

```text
API 请求：最多重试 3 次，等待 2 秒、8 秒、30 秒
工具调用：仅传输故障最多重试 2 次
模型服务：正常启动 1 次，失败后换端口再尝试 1 次
W&B 上传：最多重试 5 次，之后进入待上传队列
上游 evaluator 进程：默认不自动重启
```

Invalid JSON, unknown tool names, invalid arguments, and incorrect answers are model results and are never retried.

中文翻译：非法 JSON、未知工具名、非法参数和错误答案属于模型能力结果，绝不能通过重试掩盖。

Configuration and data failures block execution during preflight. Evaluator crashes produce `failed`, preserve stdout and stderr, and do not create a synthetic score. W&B failure never changes a locally completed benchmark result.

中文翻译：配置或数据问题在 preflight 阶段直接阻止运行。evaluator 崩溃时状态记为 `failed`，保留 stdout/stderr，不能伪造分数。W&B 上传失败不能改变已经在本地完成的 benchmark 结果。

Final JSON and JSONL indexes are written to temporary files and atomically replaced.

中文翻译：最终 JSON 和 JSONL 索引先写入临时文件，完成后再进行原子替换，避免中断产生半截文件。

## 25. Resume Rules / 断点恢复规则

中文翻译：恢复逻辑按照已有 artifact 的完整程度决定：

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

```text
存在合法 result.json
  -> 跳过该 benchmark

存在完整 predictions
  -> 只重新评分

存在部分 predictions
  -> 只生成缺失 sample ID，再评分

只有上游原始结果
  -> 重新解析

只有 W&B 尚未上传
  -> 只重新上传

缓存键或可比性键不同
  -> 拒绝复用，重新生成预测
```

## 26. Comparability Contract / 可比性契约

Formal checkpoint comparison requires equality of benchmark ID/version, manifest hash, sample-ID set, prompt/template hash, generation configuration, media preprocessing, tool schema/order, upstream evaluator version, ToolVista parser version, and required coverage.

中文翻译：正式 checkpoint 对比要求 benchmark ID/版本、manifest hash、sample ID 集合、prompt/template hash、生成配置、媒体预处理、工具 schema/顺序、上游 evaluator 版本、ToolVista parser 版本以及 coverage 要求全部一致。

If any requirement differs, derived comparison metrics are `null` with an explicit reason. They are never replaced with zero.

中文翻译：任一条件不同，派生对比指标必须写为 `null` 并说明原因，不能使用 0 代替。

## 27. Data Leakage Prevention / 数据泄漏防护

The split policy is:

中文翻译：数据划分原则如下：

```text
official train -> training only
official validation -> tuning and checkpoint selection
official test -> final evaluation only
```

```text
官方 train -> 只用于训练
官方 validation -> 用于调参和 checkpoint 选择
官方 test -> 只用于最终评测
```

When no official split exists, ToolVista creates a stable hash-based train/dev/test partition.

中文翻译：如果上游没有官方 split，ToolVista 使用稳定哈希创建 train/dev/test 划分。

Every evaluation manifest stores manifest version, source project/commit/split, stable sample IDs, normalized text hashes, media hashes, tool-schema hashes, and a manifest hash.

中文翻译：每个 evaluation manifest 都保存 manifest 版本、来源项目/commit/split、稳定 sample ID、标准化文本 hash、媒体 hash、tool-schema hash 和整个 manifest 的 hash。

Cross-source deduplication checks stable IDs, normalized text, media hashes, and tool/answer content. Test solutions, tool outputs, and reward labels are not exposed to training prompts.

中文翻译：跨数据源去重检查稳定 ID、标准化文本、媒体 hash 和工具/答案内容。测试答案、工具输出和 reward 标签不能暴露给训练 prompt。

## 28. Checkpoint Selection / Checkpoint 选择

Checkpoint selection does not use training loss alone and does not select directly from online RL reward.

中文翻译：checkpoint 不能只根据训练 loss 选择，也不能直接根据在线 RL reward 选择。

Selection order:

中文翻译：选择顺序如下：

1. Enforce historical capability and general-retention constraints on validation manifests.
2. Among eligible checkpoints, maximize the current stage's primary validation metric.
3. Use test manifests only for final reporting.

中文翻译：

1. 先在 validation manifest 上满足历史能力和通用能力保持约束。
2. 在满足约束的 checkpoint 中，最大化当前训练阶段的主验证指标。
3. test manifest 只用于最终报告，不参与选择。

Initial configurable constraints are:

中文翻译：初始可配置约束如下，字段名保持英文：

```text
general_retention >= 0.95
xlam_forgetting <= 0.03
toolvqa_forgetting <= 0.03
mllm_router_forgetting <= 0.03
```

## 29. Derived Metrics / 派生指标

中文翻译：派生指标公式如下，变量名保持英文以便与实现一致：

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

中文翻译：每个 registry 条目都声明指标优化方向和主指标。除非显式定义了有名称、有版本的复合指标，否则聚合器不能把互不相关的指标平均成一个含义不透明的总分。

## 30. RL A/B Protocol / RL A/B 测试协议

Smoke runs use GRPO, BNPO, and DAPO with seed 42. They validate data loading, rollout, reward registration, gradient updates, logs, checkpoint saving, evaluation, and resume, but do not rank algorithms.

中文翻译：冒烟阶段分别使用 GRPO、BNPO 和 DAPO，seed 均为 42。它们只验证数据加载、rollout、reward 注册、梯度更新、日志、checkpoint 保存、评测和恢复流程，不用于算法排名。

Formal runs use all three algorithms with seeds 42, 1234, and 2026, for nine total runs.

中文翻译：正式实验使用三种算法和 42、1234、2026 三个 seed，共 9 次运行。

Every formal run uses the same C4 checkpoint, Visual-RFT manifest, reward functions, prompt/template, generation parameters, effective training/generated-token budget, evaluation manifests, and seed set.

中文翻译：所有正式运行必须使用相同的 C4 checkpoint、Visual-RFT manifest、reward 函数、prompt/template、生成参数、有效训练/生成 token 预算、评测 manifest 和 seed 集合。

Compare results against training step, cumulative generated tokens, and cumulative GPU hours. Report every seed and `mean +/- std`.

中文翻译：结果必须分别以训练 step、累计生成 token 和累计 GPU 小时为横轴进行比较，并报告每个 seed 的结果以及 `均值 +/- 标准差`。

## 31. W&B and Local Storage / W&B 与本地存储

Local JSON and JSONL remain the source of truth. W&B provides training curves, reward distributions, checkpoint tables, algorithm/seed comparisons, retention/forgetting curves, efficiency metrics, artifact references, and manifest hashes.

中文翻译：本地 JSON 和 JSONL 始终是事实来源。W&B 用于展示训练曲线、reward 分布、checkpoint 表格、算法/seed 对比、retention/forgetting 曲线、效率指标、artifact 引用和 manifest hash。

Recommended W&B project:

中文翻译：推荐使用以下 W&B project 名称：

```text
toolvista-qwen3vl
```

Groups are `c1-xlam-sft`, `c2-toolvqa-sft`, `c3-mllm-router-sft`, `c4-unified-sft`, `r1-grpo`, `r2-bnpo`, `r3-dapo`, and `r4-mat`.

中文翻译：W&B group 按训练阶段划分为 `c1-xlam-sft`、`c2-toolvqa-sft`、`c3-mllm-router-sft`、`c4-unified-sft`、`r1-grpo`、`r2-bnpo`、`r3-dapo` 和 `r4-mat`。

Only locally written and validated results are uploaded. W&B can be rebuilt from local artifacts without rerunning inference.

中文翻译：只有已经在本地写入并通过校验的结果才能上传。即使 W&B 数据丢失，也必须能够只依靠本地 artifact 重建，而不需要重新推理。

## 32. Testing Strategy / 测试策略

### 32.1 Unit Tests / 单元测试

Cover registry validation, suite expansion, converters, parsers, cache/comparability keys, status and coverage calculation, metric scaling, and resume planning.

中文翻译：覆盖 registry 校验、suite 展开、converter、parser、缓存键/可比性键、状态与 coverage 计算、指标缩放和恢复计划。

### 32.2 Contract Tests / 契约测试

Every upstream integration has a small frozen prediction fixture and expected upstream result. Contract tests validate the bridge without loading a model.

中文翻译：每个上游集成都要准备一份小型冻结 prediction fixture 和预期上游结果。契约测试在不加载模型的情况下验证格式桥接层。

### 32.3 Evaluator Parity Tests / Evaluator 一致性测试

For the same frozen prediction, the upstream evaluator score must equal ToolVista `upstream_metrics`, except for a declared floating-point serialization tolerance.

中文翻译：对于同一份冻结预测，上游 evaluator 的分数必须与 ToolVista `upstream_metrics` 完全一致，只允许事先声明的浮点序列化误差。

### 32.4 Smoke Tests / 冒烟测试

Every benchmark runs 1 to 10 samples through preflight, prediction, conversion, scoring, result validation, and artifact writing.

中文翻译：每个 benchmark 使用 1 到 10 条样本贯通 preflight、预测、转换、评分、结果校验和 artifact 写入。

### 32.5 Recovery Tests / 恢复测试

Simulate API timeout, HTTP 429/5xx, partial predictions, evaluator crash, missing result files, W&B failure, and tool-server failure.

中文翻译：模拟 API 超时、HTTP 429/5xx、部分预测、evaluator 崩溃、结果文件缺失、W&B 失败和工具服务器失败。

### 32.6 Leakage and Cache Tests / 泄漏与缓存测试

Training/evaluation manifests must not overlap under the configured ID, text, media, schema, answer, and tool-output hash rules. Changing model, prompt, generation settings, data, preprocessing, tool schema, or evaluator version must invalidate the correct cache layer.

中文翻译：训练和评测 manifest 按配置的 ID、文本、媒体、schema、答案和工具输出 hash 规则不能重合。模型、prompt、生成设置、数据、预处理、工具 schema 或 evaluator 版本发生变化时，必须使正确的缓存层失效。

## 33. Implementation Phases / 实施阶段

### Phase 1: Core and Immediately Usable Evaluation / 核心与可立即使用的评测

中文翻译：Phase 1 实现统一 CLI、registry/suite loader、运行 manifest 与目录、`command` 和本地执行模式、`swift eval`/VLMEvalKit 集成、xLAM scorer、MLLM Router scorer、结果协议、retention/forgetting 聚合和 W&B 上传。

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

### Phase 2: ToolVQA Step Evaluation / ToolVQA 逐步评测

中文翻译：Phase 2 实现 ToolVQA manifest、动态 OpenCompass 配置、OpenAI-compatible 模型 adapter、GTABenchEvaluator 结果 parser，以及 `every_with_gt` parity 和冒烟测试。

```text
ToolVQA manifest preparation
dynamic OpenCompass configuration
OpenAI-compatible model adapter
GTABenchEvaluator result parser
every_with_gt parity and smoke tests
```

### Phase 3: Media and Tool Execution / 媒体与真实工具执行

中文翻译：Phase 3 实现 ToolVQA 媒体下载和路径重映射、端到端 AgentLego 编排、Visual-RFT 分类 manifest，以及 COCO/LISA scorer bridge。

```text
ToolVQA media download/remapping workflow
ToolVQA end-to-end AgentLego orchestration
Visual-RFT classification manifests
COCO and LISA scorer bridges
```

### Phase 4: External and Agentic Benchmarks / 外部与 Agentic Benchmark

中文翻译：Phase 4 集成官方 BFCL v3、通过 VLMEvalKit 接入 MAT-Bench、实现 MAT-Coding 隔离执行、MAT-Search frozen/live 环境，以及 GRPO/BNPO/DAPO 多 seed 聚合。

```text
official BFCL v3 integration
MAT-Bench through VLMEvalKit
MAT-Coding isolated execution
MAT-Search frozen and live environments
GRPO / BNPO / DAPO multi-seed aggregation
```

## 34. Acceptance Criteria / 验收标准

Phase 1 is accepted when:

中文翻译：Phase 1 满足以下条件时可以验收：

1. One command evaluates a checkpoint with xLAM, MLLM Router, and a standard ms-swift/VLMEvalKit suite.
2. A stopped run resumes without regenerating completed predictions.
3. Model, prompt, data, and evaluator changes invalidate the correct cache layer.
4. Raw upstream artifacts and ToolVista result envelopes are both preserved.
5. W&B can be reconstructed from local artifacts without inference.
6. Retention and forgetting are computed only for comparable results.
7. Evaluator parity tests pass for every implemented bridge.

中文翻译：

1. 一个命令可以对同一 checkpoint 运行 xLAM、MLLM Router 和标准 ms-swift/VLMEvalKit 测试。
2. 中断后的运行可以恢复，且不会重新生成已完成的预测。
3. 模型、prompt、数据或 evaluator 发生变化时，正确的缓存层会失效。
4. 上游原始 artifact 和 ToolVista 统一结果都得到保留。
5. 不重新推理即可从本地 artifact 重建 W&B。
6. retention 和 forgetting 只在结果具备可比性时计算。
7. 每个已经实现的 bridge 都通过 evaluator parity 测试。

The complete system is accepted when every training stage runs cumulative lightweight regression; major checkpoints run comprehensive evaluation; ToolVQA, Visual-RFT, and MAT preserve upstream semantics; nine formal RL runs are aggregated per seed and as mean/std; and all formal rows have immutable manifests, required coverage, complete provenance, and valid comparability keys.

中文翻译：完整系统的验收条件是：每个训练阶段都会运行累计轻量回测；关键 checkpoint 会运行完整评测；ToolVQA、Visual-RFT 和 MAT 保持上游指标语义；9 个正式 RL 运行按单 seed 和均值/标准差聚合；所有正式对比行都具备不可变 manifest、所需 coverage、完整 provenance 和合法 comparability key。

## 35. Final Boundary / 最终职责边界

ToolVista is the control plane for training-data conversion and evaluation orchestration. It is not a replacement for ms-swift, EvalScope, OpenCompass, VLMEvalKit, BFCL, AgentLego, COCO, LISA, Visual-RFT, or Visual-ARFT.

中文翻译：ToolVista 是训练数据转换和评测编排的控制层，不替代 ms-swift、EvalScope、OpenCompass、VLMEvalKit、BFCL、AgentLego、COCO、LISA、Visual-RFT 或 Visual-ARFT。

The implementation preference is:

中文翻译：实现时的优先级如下：

```text
reuse upstream command
  -> import upstream scorer
  -> add a minimal ToolVista scorer only when upstream scoring is absent
```

```text
优先复用上游命令
  -> 其次直接导入上游 scorer
  -> 只有上游没有评分能力时，才增加最小化 ToolVista scorer
```

This boundary keeps results comparable to the original projects while giving ToolVista one reproducible workflow for staged SFT, RL algorithm comparison, regression detection, and experiment reporting.

中文翻译：这个边界既能保持结果与原始项目可比较，又能让 ToolVista 为分阶段 SFT、RL 算法对比、回归检测和实验报告提供一条统一、可复现的工作流。
