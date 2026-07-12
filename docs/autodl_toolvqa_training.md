# AutoDL ToolVQA to ms-swift Training

This workflow keeps the projects separate:

```text
/root/autodl-tmp/ToolVista   data conversion and exported JSONL
/root/autodl-tmp/ms-swift    existing training framework
/root/autodl-tmp/model       base model files
```

Run all commands in an AutoDL terminal.

## 1. Prepare repositories and Python environment

Put this ToolVista repository at `/root/autodl-tmp/ToolVista`, then run:

```bash
cd /root/autodl-tmp/ToolVista
bash scripts/autodl/00_setup_env.sh
```

The script reuses the PyTorch/CUDA environment in the AutoDL image. It clones
ms-swift only when `/root/autodl-tmp/ms-swift` does not already exist, installs
that local repository in editable mode, and installs Qwen3-VL dependencies.

## 2. Download ToolVQA and the base model

Official Hugging Face:

```bash
cd /root/autodl-tmp/ToolVista
bash scripts/autodl/01_download_toolvqa.sh
bash scripts/autodl/01_download_qwen3_vl.sh
```

When direct Hugging Face access is unavailable, set a mirror for the same
commands:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/autodl/01_download_toolvqa.sh
bash scripts/autodl/01_download_qwen3_vl.sh
```

Expected paths:

```text
/root/autodl-tmp/ToolVista/data/raw/ToolVQA/train.jsonl
/root/autodl-tmp/ToolVista/data/raw/ToolVQA/test.jsonl
/root/autodl-tmp/ToolVista/data/raw/ToolVQA/images/
/root/autodl-tmp/model/Qwen3-VL-4B-Instruct/config.json
```

If the model is already present elsewhere, skip model download and set
`MODEL=/your/model/path` when starting training.

## 3. Run a 50-row conversion smoke test

```bash
cd /root/autodl-tmp/ToolVista
MAX_ITEMS=50 bash scripts/autodl/02_export_toolvqa_ms_swift.sh
```

Inspect the report before training:

```bash
cat data/exports/msswift_sft/toolvqa_train.report.json
head -n 1 data/exports/msswift_sft/toolvqa_train.jsonl
head -n 5 data/exports/msswift_sft/toolvqa_train.errors.jsonl
```

The current canonical schema intentionally skips ToolVQA rows whose final
answer is generated image media such as `image/output.png`, or whose later
tool calls consume generated intermediate images. These rows are recorded in
the errors JSONL instead of being silently converted into incorrect text data.

## 4. Start the smoke training run

```bash
cd /root/autodl-tmp/ToolVista
bash scripts/autodl/04_train_toolvqa_smoke.sh
```

This command converts 50 rows and starts Qwen3-VL-4B-Instruct LoRA SFT with
conservative single-GPU settings. The training script first runs the installed
ms-swift `MessagesPreprocessor` against the exported dataset.

For a model in another directory:

```bash
MODEL=/root/autodl-tmp/your-model \
bash scripts/autodl/04_train_toolvqa_smoke.sh
```

## 5. Convert the complete training split

`MAX_ITEMS=0` means no adapter limit:

```bash
cd /root/autodl-tmp/ToolVista
MAX_ITEMS=0 bash scripts/autodl/02_export_toolvqa_ms_swift.sh
```

Check the report again. The number of exported rows may be lower than 21,105
because output-image trajectories are rejected until the canonical schema gains
separate generated-media support.

## 6. Start the full LoRA SFT run

```bash
cd /root/autodl-tmp/ToolVista
EPOCHS=1 \
GRAD_ACC=16 \
IMAGE_MAX_TOKEN_NUM=768 \
MAX_LENGTH=4096 \
bash scripts/autodl/03_train_toolvqa_sft.sh
```

The default output directory is:

```text
/root/autodl-tmp/ToolVista/output/qwen3_vl_4b_toolvqa_lora
```

Watch GPU and training logs in separate terminals:

```bash
watch -n 1 nvidia-smi
tensorboard --logdir /root/autodl-tmp/ToolVista/output --port 6006 --bind_all
```

## GPU adjustments

For a 16 GiB GPU, use the 2B model or reduce image and sequence limits:

```bash
MODEL=/root/autodl-tmp/model/Qwen3-VL-2B-Instruct \
IMAGE_MAX_TOKEN_NUM=384 \
MAX_LENGTH=2048 \
GRAD_ACC=16 \
bash scripts/autodl/03_train_toolvqa_sft.sh
```

For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
GRAD_ACC=8 \
bash scripts/autodl/03_train_toolvqa_sft.sh
```

If a 4B single-GPU run still runs out of memory, first reduce
`IMAGE_MAX_TOKEN_NUM`, then `MAX_LENGTH`. Do not enable packing until the
non-packed smoke run completes successfully.

On GPUs that do not support bfloat16, add `TORCH_DTYPE=float16` to the
training command.
