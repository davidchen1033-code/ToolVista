# verl BrowserGym AgentRL Design

**Date:** 2026-09-07

**Status:** Implemented locally on 2026-09-08; target AutoDL runtime verification pending. See ../plans/2026-09-08-verl-browsergym-agentrl.md and ../../agentrl_verl_usage.md.

## Goal

Add an isolated multi-turn GUI AgentRL path to ToolVista using verl Agent Loop and BrowserGym MiniWoB, while preserving the existing ms-swift SFT and static Visual-RFT GRPO pipelines. The new path must be able to start from a completed ToolVista LoRA checkpoint after explicit validation and export, run a deterministic local browser environment, and expose a smoke-testable rollout before GPU training.

## Non-goals

- Do not replace `scripts/autodl/train_rl.sh` or the current Visual-RFT reward plugins.
- Do not support Android emulators, OSWorld VMs, or arbitrary public websites in the first implementation.
- Do not add a learned reward model to the first MiniWoB path; use the environment's verifiable reward and small action/format penalties.
- Do not claim that the current static GRPO checkpoint is automatically an environment agent policy. It must pass an action-format and cold-start gate first.

## Architecture

The repository keeps two independent training paths:

```text
Existing path:
ToolVista dataset -> ms-swift SFT/GRPO -> static visual evaluator

New path:
task prompt -> verl AgentLoop -> BrowserGym MiniWoB/Playwright -> screenshot observation
           -> Qwen3-VL action generation -> programmatic terminal reward -> verl GRPO
```

The new path consists of four boundaries:

1. `toolvista/agentrl/actions.py` defines a small JSON action protocol: `click`, `type`, `scroll`, `press`, and `finish`. It validates normalized coordinates, bounded scroll values, and terminal status before an environment call.
2. `toolvista/agentrl/miniwob_env.py` adapts BrowserGym/Gymnasium to a stable internal environment interface. It owns reset, screenshot capture, action execution, step limits, and terminal reward extraction. The actor receives screenshots and task/history text only; DOM/accessibility data is retained for diagnostics and evaluator logic.
3. `toolvista/agentrl/verl_agent_loop.py` implements the version-pinned verl `AgentLoopBase` contract. Each model turn is followed by action parsing and environment stepping. The loop returns generated token ids and a response mask that excludes tool/environment observations from policy loss. The implementation must fail clearly when the installed verl API is incompatible instead of silently falling back to static GRPO.
4. `scripts/autodl/train_agent_rl.sh` and `configs/verl/miniwob_grpo.yaml` provide AutoDL preflight, dependency checks, checkpoint authorization, smoke rollout, and formal GRPO launch. Existing launchers remain unchanged.

## Checkpoint transition

The current ms-swift checkpoint is normally a LoRA adapter directory. The new launcher will not pass that directory directly to verl. `scripts/prepare_verl_checkpoint.py` will:

1. Require a base Qwen3-VL model and an adapter containing `adapter_model.safetensors`.
2. Validate optional `trainer_state.json`, retention gate, and source provenance when supplied.
3. Use the installed Swift exporter to merge the adapter into a fresh output directory with safe serialization.
4. Write a manifest containing absolute paths, SHA-256 hashes, base model identity, adapter identity, and merge command metadata.
5. Refuse to overwrite an existing output or manifest unless `--overwrite` is explicit.

The AgentRL run then attaches a new LoRA adapter to the merged policy. This prevents cross-framework optimizer-state assumptions and preserves an exact pre-RL model for retention comparisons.

## Rollout and reward contract

Each dataset row contains a task prompt plus optional environment configuration. A rollout records:

```text
episode_id
task_id
step_index
observation image path/hash
assistant action text
parsed action JSON
environment result
terminal/success flags
episode reward
```

The first reward contract is deterministic:

- terminal task success: `+1.0`;
- terminal failure or timeout: `0.0`;
- invalid action or malformed action JSON: `-0.02`;
- no-op/repeated state: `-0.02` per bounded event;
- loop detection: at most `-0.05` per episode;
- process shaping: disabled by default unless BrowserGym supplies a verifiable intermediate reward, and capped at `0.2` total.

The reward implementation must be pure with respect to a rollout record and must not call a second VLM. A future learned judge can be added as a separate bounded component after offline calibration.

## GUI SFT gate

No mandatory full SFT continuation is required if the current checkpoint already emits valid GUI actions and has non-trivial MiniWoB cold-start success. Before AgentRL, the launcher runs a gate on a fixed smoke set:

- action JSON parse rate must be at least `95%`;
- coordinate/range validation must be at least `95%` on parsed actions;
- terminal success need not be high, but a zero-success policy is reported as a blocking cold-start failure.

If the gate fails, `scripts/prepare_gui_sft.py` can convert verified GUI trajectory state-action pairs into a short continuation SFT dataset. The continuation is intended to be `5K-20K` deduplicated state-action examples, `0.5-1` epoch, with replay from the existing ToolVista data to reduce router/visual regression. The SFT output must be evaluated by the existing retention suites before it becomes the AgentRL starting checkpoint.

## Failure handling

- Missing optional dependencies produce an actionable preflight error and never alter existing training artifacts.
- Browser crashes, page load timeouts, malformed model output, and evaluator exceptions terminate only the current episode and are logged with a typed failure reason.
- A rollout worker must always close its BrowserGym context in a `finally` path.
- Dataset/task identifiers and environment versions are recorded so a failed run can be reproduced.
- Formal training refuses an unverified merged checkpoint, a missing environment package, an invalid action schema, or an existing output directory.

## Testing and verification

The implementation will include:

- pure action parser tests for valid actions, malformed JSON, coordinate bounds, and finish handling;
- fake-environment rollout tests that verify reset/step/close and reward propagation without BrowserGym;
- checkpoint manifest tests for hashing, overwrite refusal, and adapter/base validation;
- launcher dry-run tests covering dependency and path contracts;
- an optional MiniWoB smoke test, skipped with a clear message when BrowserGym/Chromium is unavailable;
- compile/import checks that do not require installing verl on the development machine.

The first real AutoDL verification is a one-episode, one-generation rollout. Only after that passes should the formal GRPO command be used.

## Dependency policy

The new Python modules use lazy imports for `verl`, `browsergym`, `gymnasium`, and `playwright`. The base ToolVista test suite remains runnable without these packages. AutoDL setup gains an explicit opt-in `INSTALL_AGENTRL=true` path that installs pinned compatible versions and verifies the AgentLoop import.
