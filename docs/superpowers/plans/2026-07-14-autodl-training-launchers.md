# AutoDL Training Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple, directly executable AutoDL pipeline for deterministic SFT replay, serial C1/C2/C3 SFT, and four comparable Visual-RFT RL branches.

**Architecture:** A Python CLI validates and combines already-exported ms-swift JSONL files. Thin numbered shell launchers source one shared SFT runner or one shared RL runner, leaving all training, rollout, reward registration, checkpointing, and W&B logging to ms-swift.

**Tech Stack:** Python 3.10+, JSONL, Bash, ms-swift `swift sft`, ms-swift `swift rlhf`, unittest.

---

### Task 1: Deterministic replay dataset builder

**Files:**
- Create: `scripts/build_sft_replay.py`
- Create: `tests/test_build_sft_replay.py`

- [ ] Write failing tests for JSONL validation, deterministic selection, anchor-based proportions, cycling small replay sources, report hashes, overwrite protection, and transactional output.
- [ ] Run `python -m unittest tests.test_build_sft_replay -v` and verify the missing module failure.
- [ ] Implement `--source NAME=PATH`, `--weight NAME=NUMBER`, `--anchor NAME`, `--seed`, `--output`, `--report`, and `--overwrite`.
- [ ] Preserve every row from the anchor source, calculate other target counts from weights, cycle deterministically when a source is smaller than its target, shuffle the merged rows, and write source provenance only to the report.
- [ ] Atomically promote JSONL and report together, restoring any previous files if promotion fails.
- [ ] Run `python -m unittest tests.test_build_sft_replay -v` and verify all tests pass.

### Task 2: Shared SFT runner and serial stage wrappers

**Files:**
- Create: `scripts/autodl/sft_common.sh`
- Create: `scripts/autodl/05_train_xlam_sft.sh`
- Create: `scripts/autodl/06_build_and_train_toolvqa_replay_sft.sh`
- Create: `scripts/autodl/07_build_and_train_mllm_router_replay_sft.sh`
- Create: `tests/test_autodl_launchers.py`

- [ ] Write launcher contract tests using a temporary fake `swift` executable and small JSONL fixtures.
- [ ] Verify C1 invokes `swift sft` from the base model without resume flags.
- [ ] Verify C2 builds an 80/20 ToolVQA/xLAM dataset and passes `C1_CHECKPOINT` through `--resume_from_checkpoint ... --resume_only_model true`.
- [ ] Verify C3 builds a 60/20/20 MLLM/ToolVQA/xLAM dataset and passes `C2_CHECKPOINT` the same way.
- [ ] Implement fail-fast checks for the ms-swift root, base model, non-empty dataset, and required previous checkpoint.
- [ ] Keep SFT hyperparameters, W&B naming, validation, output directories, and optional `MAX_STEPS` in `sft_common.sh`.
- [ ] Run the focused launcher tests.

### Task 3: Shared RL runner and four A/B wrappers

**Files:**
- Create: `scripts/autodl/rl_common.sh`
- Create: `scripts/autodl/08_train_visual_rft_grpo.sh`
- Create: `scripts/autodl/09_train_visual_rft_bnpo.sh`
- Create: `scripts/autodl/10_train_visual_rft_dapo.sh`
- Create: `scripts/autodl/11_train_visual_rft_drgrpo.sh`
- Modify: `tests/test_autodl_launchers.py`

- [ ] Add contract tests asserting every branch uses the same `C3_CHECKPOINT`, dataset, reward plugin, seed, generation settings, and common hyperparameters.
- [ ] Assert `loss_type` is respectively `grpo`, `bnpo`, `dapo`, and `dr_grpo`.
- [ ] Assert DrGRPO disables reward standard-deviation scaling and DAPO alone enables asymmetric clipping, dynamic sampling, resampling, and overlong filtering.
- [ ] Implement `swift rlhf --rlhf_type grpo` with `--adapters C3_CHECKPOINT --ref_adapters C3_CHECKPOINT`, the Visual-RFT external reward plugin, vLLM colocate, LoRA, W&B, and diagnostic metrics.
- [ ] Keep the Visual-RFT reward identical across all four branches; do not add DAPO `soft_overlong` to this A/B.
- [ ] Run the focused launcher tests.

### Task 4: Evaluation launcher and AutoDL handoff

**Files:**
- Create: `scripts/autodl/12_run_light_eval.sh`
- Create: `scripts/autodl/README.md`
- Modify: `tests/test_autodl_launchers.py`

- [ ] Add a contract test that maps checkpoint, run ID, stage, suite, API base, API model, seed, baseline summary, history summary, and W&B settings to `scripts/evaluate.py`.
- [ ] Implement the evaluation wrapper without starting or stopping the model service.
- [ ] Document sibling repository layout, raw export commands, evaluation manifest preparation, C0 baseline, C1/C2/C3 commands, explicit checkpoint selection, four RL branches, W&B grouping, smoke overrides, and the exact benchmark sources.
- [ ] Run the focused launcher tests.

### Task 5: Repository verification

**Files:**
- Modify only files from Tasks 1 through 4 if verification exposes a defect.

- [ ] Run `python -m unittest tests.test_build_sft_replay tests.test_autodl_launchers -v`.
- [ ] Run the complete repository suite with `python -m unittest discover -s tests -v`.
- [ ] Run `python -m py_compile scripts/build_sft_replay.py`.
- [ ] Run Bash syntax checks with Git Bash when available.
- [ ] Inspect `git diff --check`, `git status --short`, and the final file list without touching the existing `.gitignore` or `data/` changes.
- [ ] Report that local contract tests do not replace AutoDL CUDA/vLLM smoke runs.
