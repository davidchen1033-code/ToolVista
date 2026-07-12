# MLLM-Tool Export Entry Point Implementation Plan

> **For agentic workers:** Execute these steps inline in order. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a command-line entry point that converts MLLM-Tool annotations through `MLLMToolAdapter` and writes ms-swift JSONL, report JSON, and error JSONL files.

**Architecture:** Keep source interpretation inside `toolvista/adapters/mllm_tool.py` and framework lowering inside `toolvista/exporters/ms_swift.py`. The new script only parses MLLM-specific file-key options, streams `AdaptResult` records through the existing exporter, and atomically writes output and reports.

**Tech Stack:** Python 3.12, `unittest`, ToolVista canonical schema, ToolVista ms-swift exporter.

---

### Task 1: Add Export Contract Tests

**Files:**
- Create: `examples/data/mllm_tool/data/IT_data_ins/T+X-T_data/*.json`
- Create: `tests/test_export_mllm_tool.py`

- [ ] Add one text sample and one image sample in the same structure as the local MLLM-Tool repository.
- [ ] Assert comma-separated file keys are normalized and invalid keys are rejected.
- [ ] Assert adapter results are written as ms-swift rows with text and image media preserved.
- [ ] Assert warnings and failures are written to the error JSONL and counted in the report statistics.
- [ ] Run the test and confirm it fails because `scripts.export_mllm_tool` does not exist.

### Task 2: Implement The Export Script

**Files:**
- Create: `scripts/export_mllm_tool.py`

- [ ] Accept config, file keys, output/report/error paths, maximum items, media-validation override, and metadata inclusion.
- [ ] Default to `image,video,audio,text`; require `combined` to be selected explicitly.
- [ ] Build `MLLMToolAdapter`, consume `adapter.iter_results()`, and pass successful samples to `canonical_to_ms_swift_row()`.
- [ ] Write output and error streams through temporary files, then atomically replace the requested paths.
- [ ] Include selected file keys, media-validation mode, and conversion counts in the report JSON.

### Task 3: Verify Local And Real Data

**Files:**
- No additional files expected.

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m py_compile scripts/export_mllm_tool.py tests/test_export_mllm_tool.py`.
- [ ] Export the local sample with media validation disabled and inspect the JSONL roles/media fields.
- [ ] Run a one-item smoke conversion for each local MLLM-Tool file key using the real repository and inspect errors without writing a full dataset.

