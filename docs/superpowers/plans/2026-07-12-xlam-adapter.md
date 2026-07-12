# xLAM Adapter Implementation Plan

> **For agentic workers:** Execute these steps inline in order. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert xLAM ActionStudio unified trajectories into ToolVista canonical SFT samples that export directly to ms-swift.

**Architecture:** Stream the large JSON array with `ijson`, then split each trajectory at every valid assistant turn. Convert historical messages with loss disabled and convert the selected assistant turn into either one text target or one-or-more consecutive canonical `tool_call` targets.

**Tech Stack:** Python 3.12, `ijson`, `unittest`, ToolVista canonical schema, ToolVista ms-swift exporter.

---

### Task 1: Fix The Source Contract With Tests

**Files:**
- Create: `examples/data/xlam/train_sample.json`
- Create: `tests/test_xlam_adapter.py`

- [ ] Add a small JSON array containing one multi-turn single-call trajectory, one parallel-call trajectory, and one `tool_calls: []` text trajectory.
- [ ] Assert that each assistant turn becomes a separate sample with the xLAM-compatible `trajectory_id---message_index` identifier.
- [ ] Assert that historical assistant/tool-call messages have `loss=False`, while target messages retain the default loss behavior.
- [ ] Assert that parallel calls export as consecutive `tool_call` messages and retain argument objects.
- [ ] Assert that dictionary/list tool responses become JSON strings and empty call lists with text become assistant text targets.
- [ ] Run `python -m unittest discover -s tests -p test_xlam_adapter.py -v` and confirm it fails because `toolvista.adapters.xlam` does not exist.

### Task 2: Implement The Streaming Adapter

**Files:**
- Create: `toolvista/adapters/xlam.py`

- [ ] Implement `_annotation_path()` using `unified_data_dir`, subset, split, and optional annotation-file override.
- [ ] Implement `iter_raw()` with `ijson.items(handle, "item")`, source indices, and `max_trajectories` limiting.
- [ ] Validate trajectory identifiers, tools, conversations, roles, tool schemas, and function call payloads into `AdaptResult` errors instead of uncaught exceptions.
- [ ] Convert `tool_calls[*].function` into compact JSON `{name, arguments}` canonical tool-call messages.
- [ ] Serialize non-string tool responses with `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`.
- [ ] Emit one canonical sample per valid assistant turn and record target kind, target calls, call IDs, source format instruction, and source indices in labels/metadata.
- [ ] Run the xLAM tests and confirm they pass.

### Task 3: Add A Streaming ms-swift Entry Point

**Files:**
- Create: `scripts/export_xlam.py`

- [ ] Reuse `canonical_to_ms_swift_row()` and atomic temporary-file replacement.
- [ ] Accept config, subset, split, annotation override, output/report/error paths, and maximum trajectory count.
- [ ] Stream adapter results so the 558 MB source file is never loaded into memory as one Python list.
- [ ] Add a test that exports the sample file and verifies single, parallel, text, and historical-loss fields in JSONL.

### Task 4: Verify End To End

**Files:**
- Modify only if required by verification: `configs/autodl.yaml`

- [ ] Run `python -m unittest discover -s tests -v` and require zero failures.
- [ ] Run `python -m py_compile toolvista/adapters/xlam.py tests/test_xlam_adapter.py scripts/export_xlam.py`.
- [ ] Export the small sample through `scripts/export_xlam.py` and inspect message roles, tools, target call JSON, and row count.
- [ ] Run the adapter against the first two trajectories of the local 558 MB ActionStudio file and confirm streaming conversion completes without loading the complete dataset.

