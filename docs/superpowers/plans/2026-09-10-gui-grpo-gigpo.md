# GUI GRPO / GiGPO implementation plan

> Execute the user-approved continuation inline. Do not download model/data/browser resources; validate locally with the available runtime.

**Goal:** Two minimal, isolated GUI training routes on the same 12 MiniWoB++ tasks and disjoint seeds (1,200 train / 120 validation).

**Architecture:** Keep verl 0.7.0 and the existing screenshot AgentLoop. Transport per-action token spans, state hashes and terminal-only rewards through AgentLoopOutput.extra_fields. A dedicated GiGPO TaskRunner temporarily wraps compute_advantage in its own Ray process because the upstream registry does not receive non-tensor metadata. Never edit or install over upstream verl. Both routes use the same trajectory-level episode baseline (one vote per trajectory), masking and loss reduction; GiGPO adds normalized same-state discounted-return advantages.

**Tech stack:** Existing Python 3.11 / BrowserGym 0.13.3 / verl 0.7.0 pins; dependency-free numerical core and unittest on the development machine.

- [x] Add compact GiGPO numerical core, conservative browser state signature, and token-span validation. Tests cover terminal rewards, group isolation, singleton groups, repeated states, batch reorder, and zero step weight equivalence to baseline.
- [x] Extend the existing AgentLoop to return terminal-only reward and in-memory per-step metadata. Fail on infrastructure errors instead of training on them. Keep legacy episode scoring available to existing smoke/gate callers.
- [x] Add the isolated TaskRunner bridge and inherited GiGPO config. Use shared GRPO normalization; retain generated token IDs and mask all observations.
- [x] Add algorithm selection to existing launcher and one thin shell wrapper; retire the unfinished fork launcher/setup in favor of compatibility forwarding to the pinned runtime.
- [x] Document the 12 task capabilities, data size vs rollout size, seed split, source revision, launch commands, and limitations. Prepare metadata only, no source resources.
- [x] Run unit tests, Python compilation, shell syntax and pinned upstream contract checks. Record remaining live browser/CUDA validation honestly.

**Design choices:** Both routes use terminal success 1 / failure 0, no invalid/no-op shaping. gamma=1, step weight=1 initially, group sample-standard-deviation normalization enabled (same as GRPO). These are explicit experimental defaults, not claimed optimal hyperparameters. The episode baseline is per trajectory, differing from the fork's default step-weighted baseline. State grouping uses rollout uid (same task/seed) plus exact state hash; no fuzzy visual matching. Page state includes live controls, focus, scroll, stable DOM and screenshot, but cannot guarantee equality of arbitrary JavaScript hidden state or policy histories. Log matched and nonzero-step-advantage fractions.


**Verification (2026-09-10):** 47 AgentRL tests ran: 45 passed, 2 skipped (real BrowserGym smoke; real verl/torch advantage integration). Python compilation and Bash syntax checks passed. Both configurations were composed and resolved against 20 upstream verl v0.7.0 YAML source paths: only the estimator, GiGPO step weight, and experiment name differ. Source contracts for TaskRunner injection, compute_advantage and extra_fields transport were checked. The Hydra primary-config searchpath restriction was reproduced and fixed by extracting miniwob_common.yaml; a composition regression test now covers it.

The data preparation dry run confirmed 1,200 train / 120 validation rows with pinned environment metadata; no Parquet files were generated because pyarrow is unavailable locally. No model, dataset, browser or environment package was downloaded/installed. Real browser interaction, actual Torch tensor integration and GPU optimization remain target-machine checks; the optional integration test is included for that runtime.
