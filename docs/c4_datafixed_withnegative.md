# C4 DAPO `datafixed-withnegative`

This track tests whether adding verified empty-target examples and an empty/false-positive-aware reward fixes the
regression seen after the original Visual-RFT RL run. It is deliberately separate from both the legacy RL runs and the
first C3-datafixed DAPO run. No control script or artifact is overwritten.

## Experiment Contract

The primary comparison is the final `step6000` model from each row below.

| Setting | C3 start | Updates | DAPO/LR/beta | Dataset and reward |
|---|---|---:|---|---|
| `datafixed` control | same passed C3 datafixed checkpoint | 6000 | `dapo`, `1e-6`, `0.04` | original positive-only base65 data and `visual_rft` |
| `datafixed-withnegative` | same passed C3 datafixed checkpoint | 6000 | `dapo`, `1e-6`, `0.04` | audited withnegative data and `visual_rft_withnegative` |

The intervention intentionally changes two coupled pieces: the training distribution and the reward that defines a
correct empty response and penalizes false positives/confidence errors. The preparation report is the source of truth
for row counts, positive/negative ratio, category balance, seed, provenance and hashes. Do not describe this run as a
pure DAPO algorithm ablation.

The launcher keeps `MAX_STEPS=6000`, `LEARNING_RATE=1e-6`, `BETA=0.04`, seed 42 and the original DAPO switches. This is
an update-matched comparison with the first datafixed DAPO run. Changing any of those values creates a follow-up
experiment and must use a new run tag. In particular, stronger KL or fewer updates may improve retention, but changing
them in the first withnegative run would make the cause of any improvement ambiguous.

Checkpoints are saved every 500 updates and `SAVE_TOTAL_LIMIT=13` retains the complete 500..6000 curve. Until an
independent frozen validation split is available, `step6000` remains the predeclared primary result.
Selecting the best checkpoint repeatedly on `visual_rft_coco_light` is test-set selection; such a result may be shown
only as post-hoc diagnostics, not as an unbiased improvement. Light is a subset of the full benchmark, so evaluating
full after selecting on light does not remove that bias.

## Prepare And Verify

Run preparation before allocating the GPU. The builder refuses unverified negatives and writes a provenance report.

```bash
cd /root/autodl-tmp/ToolVista

PYTHON_BIN=/root/autodl-tmp/venvs/toolvista/bin/python
DATASET=/root/autodl-tmp/ToolVista/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_train.jsonl
DATASET_REPORT=/root/autodl-tmp/ToolVista/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_train.report.json

"$PYTHON_BIN" scripts/prepare_visual_rft_datafixed_withnegative.py \
  --output "$DATASET" \
  --report "$DATASET_REPORT"

"$PYTHON_BIN" scripts/prepare_visual_rft_datafixed_withnegative.py \
  --output "$DATASET" \
  --report "$DATASET_REPORT" \
  --verify-only
```

The training launcher runs the same `--verify-only` check again immediately before delegating to the proven DAPO
launcher. Negative rows come from the 3,156 COCO `val2017` images left after excluding every one of the 1,844 image IDs
in the frozen full novel-15 manifest. The selected 2,690 negative images are globally unique and each target absence is
checked against official annotations. This is exact-image-disjoint from the local benchmark, but it is not an official
COCO-train split; the report must disclose that distinction when results are presented.

### Media Overlap And Leakage Audit

The 1,844-ID exclusion proves exact-image disjointness only for the generated negative rows, whose official COCO image
IDs remain recoverable. The upstream positive artifact exposes only materialized identifiers such as
`visual-rft-base65:000414`; its official COCO image IDs cannot be recovered, so the same ID-based proof is not available
for positives. A read-only audit of all 6,000 source-positive PNGs against the 1,844 full-evaluation images found no
decoded-RGB exact match, but did confirm these two same-scene, adjacent-frame pairs. Both positive rows were selected by
the earlier 3,310-positive preparation.

| Positive source and target | Full-evaluation source and target | 64-bit hashes | SIFT/RANSAC confirmation |
|---|---|---:|---:|
| `visual-rft-base65:000414` (`images/000414.png`, `sink`) | `coco-val2017-508730-70` (`000000508730.jpg`, `toilet`, GT present) | pHash 12, dHash 5 | 159 ratio matches, 116 inliers (72.96%) |
| `visual-rft-base65:003442` (`images/003442.png`, `chair`) | `coco-val2017-439525-48` (`000000439525.jpg`, `fork`, GT absent) | pHash 16, dHash 13 | 634 ratio matches, 489 inliers (77.13%) |

The source-positive set has 5,818 unique decoded images: its remaining 182 rows are repeated content under different
target categories. The hash audit treated those repeats as one image before candidate matching, then restored row
weights. Joint pHash/dHash Hamming thresholds at or below 8 and 10 produced no candidates; the wider threshold search
through 18 was geometrically checked so that similar layouts were not counted as leakage.

Canonical v2 therefore deny-lists `visual-rft-base65:000414` and `visual-rft-base65:003442` before deterministic
per-category ranking and the 60-row cap. The next eligible `sink` and `chair` rows backfill the two slots, preserving the
3,310 positive / 2,690 negative contract. Verification must reject either denied identifier in canonical output and
record the exclusions and replacements in its report. This rule does not rewrite the positive source, the legacy
datafixed-withnegative dataset, or any control artifact.

Results may state that the negative rows are exact-image-disjoint from the frozen full benchmark. They must not state
that the original positive source or the earlier whole training dataset was scene-disjoint. For canonical v2, the
accurate claim is narrower: the two confirmed positive/evaluation near-duplicate scenes are excluded; absence of
recoverable official positive IDs prevents an exhaustive ID-level proof for every positive image.

## Start In Tmux

Start this only after the current DAPO/evaluation services have exited and `nvidia-smi` shows the GPU allocation has
been released. The command creates the training, log and GPU windows, selects the live training output, and immediately
attaches to tmux.

```bash
cd /root/autodl-tmp/ToolVista

export SESSION=c4fixed-withnegative-dapo
export RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
export C3_DATAFIXED_CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/c3_mllm_router_datafixed_sft/v1-20260721-203434/checkpoint-1063
export DATASET=/root/autodl-tmp/ToolVista/data/exports/msswift_grpo/visual_rft_coco_base65_datafixed-withnegative_train.jsonl
export DATASET_REPORT=/root/autodl-tmp/ToolVista/data/reports/c4/visual_rft_coco_base65_datafixed-withnegative_train.report.json
export OUTPUT_DIR=/root/autodl-tmp/ToolVista/outputs/rl/dapo_datafixed-withnegative_seed42_run${RUN_TAG}
export TRAIN_LOG=/root/autodl-tmp/ToolVista/logs/c4/c4-dapo-datafixed-withnegative-${RUN_TAG}-seed42.train.log

tmux new-session -d -s "$SESSION" -n train \
  "cd /root/autodl-tmp/ToolVista && env C3_DATAFIXED_CHECKPOINT='$C3_DATAFIXED_CHECKPOINT' DATASET='$DATASET' DATASET_REPORT='$DATASET_REPORT' OUTPUT_DIR='$OUTPUT_DIR' TRAIN_LOG='$TRAIN_LOG' RUN_TAG='$RUN_TAG' SEED=42 WANDB_PROJECT=toolvista-qwen3vl bash scripts/autodl/train_rl.sh --variant datafixed-withnegative --algorithm dapo --foreground; status=\$?; echo TRAIN_EXIT=\$status; exec bash" &&
tmux new-window -t "$SESSION" -n log \
  "bash -lc 'tail --retry -n 100 -F \"$TRAIN_LOG\"; exec bash'" &&
tmux new-window -t "$SESSION" -n gpu \
  "bash -lc 'watch -n 10 nvidia-smi; exec bash'" &&
tmux select-window -t "$SESSION":train &&
tmux attach-session -t "$SESSION"
```

Inside tmux, `Ctrl-b 0` shows training, `Ctrl-b 1` follows the complete log, `Ctrl-b 2` shows GPU state, and `Ctrl-b d`
detaches without stopping the run. To stop deliberately, attach to the training window and press `Ctrl-c`; do not kill
unrelated tmux sessions or inference services.

The launcher refuses an existing output directory or log file. A retry therefore needs a new `RUN_TAG`, unless it is
an intentional resume workflow implemented and recorded separately.

## Compare Results

For the primary table, evaluate these three checkpoints with the same candidate deployment settings, fixed judge,
`c4_datafixed_light` suite and C3 datafixed history summary:

1. C3 datafixed `checkpoint-1063` before RL.
2. Original datafixed DAPO `step6000` control.
3. Datafixed-withnegative DAPO `step6000` intervention.

Report at least Novel15 mAP, empty-query correctness/false positives, positive-query localization, Router, RealWorldQA,
MathVision, MMMU, OCRBench, xLAM, ToolVQA and BFCL. Preserve the run summary, manifest, dataset report, ms-swift
`args.json`, W&B run ID and exact adapter checkpoint together. Single-seed differences near the normal light-set
variance are candidates for a full evaluation or additional seeds, not a stable ranking.

For the diagnostic learning curve, merge and evaluate the retained 500-step checkpoints under distinct run IDs. Keep
the final `step6000` result as the strict matched comparison, and label any best-intermediate result explicitly as
post-hoc unless checkpoint selection used a frozen, disjoint training-validation split.

## Run The C4 Datafixed Light Suite

After merging the selected RL adapter and starting the candidate on port 8010 plus the frozen base-model judge on
port 8011, use the C4-specific wrapper below. Do not reuse `12_run_datafixed_light_eval.sh`: that script belongs to C3,
forces the C3 suite/checkpoint contract, and rewrites the C3 retention gate. The C4 wrapper is evaluation-only; it does
not merge a checkpoint or start/stop either inference service.

```bash
cd /root/autodl-tmp/ToolVista

export CHECKPOINT=/root/autodl-tmp/ToolVista/outputs/final/c4-dapo-datafixed-withnegative-seed42-step6000-merged
export RUN_ID=c4-dapo-datafixed-withnegative-light-seed42-step6000
export API_MODEL=c4-dapo-datafixed-withnegative-step6000
export HISTORY_SUMMARY=/root/autodl-tmp/ToolVista/runs/c3_datafixed_light_seed42_step1063/summary.json

CHECKPOINT="$CHECKPOINT" \
RUN_ID="$RUN_ID" \
API_MODEL="$API_MODEL" \
API_BASE=http://127.0.0.1:8010/v1 \
EVAL_URL=http://127.0.0.1:8010/v1 \
HISTORY_SUMMARY="$HISTORY_SUMMARY" \
JUDGE_API_BASE=http://127.0.0.1:8011/v1 \
JUDGE_MODEL=Qwen3-VL-4B-Instruct \
ALGORITHM=dapo \
bash scripts/autodl/lib/evaluate_c4_datafixed.sh
```

The wrapper fixes `c4_datafixed_light`, `benchmarks_datafixed.yaml`, `suites_datafixed.yaml`, seed 42, and the judge
protocol recorded beside the passed C3 datafixed summary. It derives `datafixed` versus `datafixed-withnegative` from
the checkpoint path and requires the checkpoint, run ID, stage, API model alias, and W&B group to remain in that same
track. Existing run directories are rejected unless `RESUME=true` is explicit. Set `WANDB_ENABLED=false` only when a
local non-W&B diagnostic run is intentional.

## Offline Prediction Audit

After a C4 evaluation finishes, audit its immutable per-query predictions with the negative-aware reward contract.
The analyzer reads either `summary.json` or a prediction JSONL, prints JSON, and writes only the optional report path.
It never rewrites the summary, predictions, COCO annotations, or reward plugin. For a ToolVista C4 run it automatically
selects the single Visual-RFT benchmark and uses the sibling filtered `coco_annotations.json` to reconstruct positive
GT boxes:

```bash
cd /root/autodl-tmp/ToolVista

PYTHON_BIN=/root/autodl-tmp/venvs/toolvista/bin/python
SUMMARY=/root/autodl-tmp/ToolVista/runs/<c4-run-id>/summary.json
REPORT=/root/autodl-tmp/ToolVista/runs/<c4-run-id>/visual_rft_withnegative_prediction_analysis.json

"$PYTHON_BIN" scripts/analyze_visual_rft_withnegative_predictions.py \
  "$SUMMARY" \
  --benchmark-id visual_rft_coco_light \
  --output "$REPORT"
```

For another JSONL schema, pass the field mapping explicitly, for example
`--prediction-field response --gold-field solution --label-field example_type`. Custom labels require both
`--positive-label` and `--negative-label`. If positive GT cannot be resolved, response-format and false-positive
metrics are still emitted, but the complete `positive_reward_ap_proxy_mean` is `null`; consult `coverage` and
`gold_resolution` before comparing runs. The reported positive proxy is the exact negative-aware reward's
multi-IoU F1 plus centered-Brier score, not official COCO AP.
