# verl BrowserGym AgentRL Implementation Plan

> **For agentic workers:** Execute this approved continuation inline, task by task. The superpowers execution sub-skills are not installed in this workspace; use the repository's unittest workflow and record verification here.

**Goal:** Add an isolated, gated MiniWoB environment-feedback GRPO path that starts from a verified merged ToolVista checkpoint.

**Architecture:** A dependency-free action/reward/episode core is shared by cold-start evaluation and a version-pinned verl AgentLoop. BrowserGym runs in a dedicated subprocess per episode because its synchronous Playwright API cannot run on verl's asyncio thread. Model observations contain only screenshots, task text, and action feedback. Checkpoint export and launch preflight fail closed.

**Tech Stack:** Python 3.11 on AutoDL, verl 0.7.0, BrowserGym 0.13.3, vLLM 0.11.2, Transformers 4.57.3, Bash, unittest. Lightweight tests also run on Windows without the optional GPU/browser packages.

## Task 1: Action and episode contract

- [x] Create `toolvista/agentrl/actions.py`, `rewards.py`, `rollout.py`, and `prompts.py`.
- [x] Implement strict JSON actions: click(x,y) in [0,1], type(text), scroll(dx,dy) in [-1000,1000], press(key) from an allowlist, finish(status=success|failure). Distinguish JSON parse errors from schema/range errors. A self-declared finish cannot grant success.
- [x] Implement a bounded async episode, screenshot/action records, pure reward calculation, failure reasons, and unconditional close.
- [x] Verify fake-environment success, invalid actions, timeouts, loops, and close-on-error with `python -m unittest tests.test_agentrl_core -v`.

## Task 2: BrowserGym and verl boundaries

- [x] Create `toolvista/agentrl/miniwob_env.py`, `dependencies.py`, and `verl_agent_loop.py`.
- [x] Translate validated actions through BrowserGym's coordinate action set; persist screenshot hashes and environment diagnostics. Require local MiniWoB assets and seeded task initialization. Isolate sync Playwright in a spawned process with a bounded IPC timeout.
- [x] Load the actual verl 0.7.0 AgentLoopBase lazily through a factory registered by Hydra `_target_`. Fail clearly on version/signature mismatch.
- [x] Preserve sampled token IDs and sampled log probabilities; append observation tokens with mask 0 and generated action tokens with mask 1. Enforce prompt/response budgets before adding images. Return terminal environment score directly via AgentLoopOutput.reward_score.
- [x] Verify token/mask/image alignment and incompatible APIs with `python -m unittest tests.test_agentrl_verl -v`.

## Task 3: Checkpoint export and provenance

- [x] Create `toolvista/agentrl/checkpoints.py` and `scripts/prepare_verl_checkpoint.py`.
- [x] Validate local Qwen3-VL config/weights and LoRA config/weights; validate supplied completed trainer state, exact-checkpoint retention gate, and source manifest. Export using an argument-list Swift subprocess into a staging directory; publish only after inspecting full model artifacts.
- [x] Hash adapter, base, merged model, and supplied evidence in a manifest. Refuse existing destinations unless explicit overwrite, and always reject source/output overlap.
- [x] Verify hashing/tamper detection, invalid base/adapter, export command, and overwrite behavior with `python -m unittest tests.test_agentrl_checkpoint -v`.

## Task 4: Cold-start gate, data, and optional SFT bridge

- [x] Create `toolvista/agentrl/gate.py`, `scripts/agentrl_smoke.py`, `scripts/prepare_agentrl_data.py`, and `scripts/prepare_gui_sft.py`.
- [x] Use a fixed 30-episode suite (click-test/click-button/enter-text, seeds 0..9). Compute JSON parse >=95%, schema validity among parsed >=95%, and at least one environment success; infrastructure errors block separately. Bind reports to the merged manifest and exact suite.
- [x] Generate disjoint seeded train/validation parquet rows containing agent_name and extra_info. Generate GUI state-action SFT only from verified successful trajectories, deduplicate by image/task/history/action, and require existing ToolVista replay.
- [x] Verify gate edge cases, deterministic data rows, deduplication, rejected unverified trajectories, and replay with `python -m unittest tests.test_agentrl_gate_data -v`.

## Task 5: AutoDL launch and documentation

- [x] Create `configs/verl/miniwob_grpo.yaml`, `miniwob_agent.yaml`, `requirements-agentrl.txt`, and `scripts/autodl/train_agent_rl.sh`; add opt-in isolated AgentRL setup to `setup_env.sh`.
- [x] Launcher supports dependency/path preflight, dry run, one-episode smoke, fixed-suite gate, and formal training. Training reruns the checkpoint-specific gate before launching GRPO with a fresh LoRA and a new output directory.
- [x] Add `docs/agentrl_verl_usage.md` with AutoDL commands, environment setup, checkpoint provenance limitations, one-generation rollout, and the conditional SFT decision.
- [x] Run `python -m unittest discover -s tests`, compile new Python modules, and Bash syntax checks where Bash is available. Optional real BrowserGym smoke must skip explicitly when unavailable. Do not claim GPU validation from fake tests.

## Upstream contracts inspected

- `https://github.com/verl-project/verl/tree/v0.7.0/verl/experimental/agent_loop`
- `https://github.com/ServiceNow/BrowserGym/tree/v0.13.3`
- `https://github.com/modelscope/ms-swift/tree/v3.11.3/swift/llm/export`

## Verification record

2026-09-08: implementation completed locally. New tests: 35 run, 34 passed, 1 optional real MiniWoB test skipped. Python compile checks, Bash syntax checks, and Hydra composition/resolution against the upstream verl 0.7.0 config tree passed. The PyPI verl 0.7.0 wheel AgentLoop, trainer config, and FSDP implementation were compared with the inspected Git tag sources and matched byte-for-byte.

Full regression: the last full run executed 387 tests (before adding the final two passing launcher tests), with 26 failures, 36 errors, and 12 skips. A clean HEAD archive under the same interpreter executed 354 tests with the identical 26 failures and 36 errors (11 skips). Failure identifiers were compared: no new failures. Existing failures include missing Swift/pandas/ijson, Windows symlink permissions, and shell/path assumptions. Logs are in logs/agentrl-local-verification/ (ignored local artifacts).

Windows verification used Python 3.14 with Pillow/PyYAML/Hydra installed in the ignored 	ests/tmp-agentrl-py314 directory and supplied through PYTHONPATH. Production remains Linux/Python 3.11 with the pinned optional requirements.

Remaining target-machine validation: real Swift merge, BrowserGym smoke, one-episode verl rollout, fixed checkpoint cold-start gate, and GPU GRPO. No target checkpoint, CUDA runtime, verl or BrowserGym was available in this workspace, so no claim is made that GPU training or the current checkpoint gate has passed.
