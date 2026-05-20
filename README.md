# nanoQwen

A small Qwen-style LLM project built in PyTorch to understand how modern decoder-only language models work internally by implementing and experimenting with the core components from scratch.

Features:
- decoder-only transformer implementation
- HF weight loading and parity validation
- pretraining and supervised fine-tuning (SFT)
- checkpointing, resume training, and text generation workflows
- KV-cache based autoregressive inference

## Implemented
- `nanoqwen/qwen3.py`
  - Qwen-style architecture (`RoPE`, `RMSNorm`, `GQA`, gated MLP)
  - weight tying (`embed_tokens` and `lm_head`)
  - HF load path for `Qwen/Qwen3-0.6B`
- `nanoqwen/data.py`
  - pretraining token cache + memmap batches
- `nanoqwen/sft_data.py`
  - JSONL SFT ingestion (`prompt`/`response`)
  - label masking (`ignore_index=-100`) + next-token alignment
  - memmap cache for train/val
- `nanoqwen/train.py`
  - pretraining loop (AdamW, warmup + cosine/constant LR)
  - val loss/perplexity, token-budget reporting, checkpoint save/load, resume
  - generation with optional KV cache
  - optional `torch.compile`, AMP, W&B
- `nanoqwen/train_sft.py`
  - SFT loop with val loss, LR schedule, checkpointing/resume
  - load pretrained checkpoint (vocab-size/compile-prefix compatibility)
  - optional `torch.compile`, AMP, W&B
- `nanoqwen/compare_hf.py`
  - local vs HF logits/token comparison
- `scripts/download_dataset.py`
  - plain-text dataset download (`train`/`validation`/`both`)
- `scripts/download_sftdata.py`
  - SFT JSONL download/split (`prompt/response` or chat `messages`)
- `scripts/sft_length_stats.py`
  - tokenized length stats for choosing sequence length

## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

If you use W&B, install it as well:
```bash
uv pip install wandb
```

## Recommended pipeline: FineWeb-Edu + UltraChat

### 1) Download pretraining data (FineWeb-Edu sample)
```bash
python scripts/download_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --name sample-10BT \
  --split both \
  --text-field text \
  --max-samples 2000000 \
  --out data/fineweb_edu.txt
```
Creates:
- `data/fineweb_edu_train.txt`
- `data/fineweb_edu_val.txt`

### 2) Download SFT data (UltraChat 200k)
```bash
python scripts/download_sftdata.py \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split both \
  --format chat_messages \
  --messages-field messages \
  --min-turns 2 \
  --max-samples 180000 \
  --out data/ultrachat_200k.jsonl
```
Creates:
- `data/ultrachat_200k_train.jsonl`
- `data/ultrachat_200k_val.jsonl`

### 3) Pretrain (~100M config)
```bash
python -m nanoqwen.train \
  --train-file-name fineweb_edu_train.txt \
  --val-file-name fineweb_edu_val.txt \
  --device cuda \
  --max-steps 20000 \
  --batch-size 2 \
  --block-size 384 \
  --n-layer 10 \
  --n-head 8 \
  --n-kv-head 4 \
  --head-dim 64 \
  --hidden-size 512 \
  --learning-rate 3e-4 \
  --warmup-steps 500 \
  --lr-scheduler cosine \
  --save-interval 2000 \
  --checkpoint-dir checkpoints \
  --use-compile \
  --use-wandb \
  --wandb-project nanoqwen \
  --wandb-run-name pretrain-100m-fineweb-edu
```
Saves checkpoints like:
- `checkpoints/train_step_002000.pt`
- `...`
- `checkpoints/train_step_020000.pt`

### 4) SFT from pretrained checkpoint
```bash
python -m nanoqwen.train_sft \
  --data-dir data \
  --sft-train-file ultrachat_200k_train.jsonl \
  --sft-val-file ultrachat_200k_val.jsonl \
  --pretrained-ckpt checkpoints/train_step_020000.pt \
  --device cuda \
  --batch-size 2 \
  --block-size 384 \
  --max-steps 6000 \
  --learning-rate 1e-4 \
  --warmup-steps 300 \
  --lr-scheduler cosine \
  --log-interval 100 \
  --eval-iters 100 \
  --save-interval 1000 \
  --checkpoint-dir checkpoints \
  --use-compile \
  --rebuild-sft-cache
```
Saves checkpoints like:
- `checkpoints/sft_step_001000.pt`
- `...`
- `checkpoints/sft_step_006000.pt`

### 5) Generate from SFT checkpoint
```bash
python -m nanoqwen.train \
  --train-file-name fineweb_edu_train.txt \
  --val-file-name fineweb_edu_val.txt \
  --device cuda \
  --max-steps 0 \
  --resume-from checkpoints/sft_step_006000.pt \
  --n-layer 10 \
  --n-head 8 \
  --n-kv-head 4 \
  --head-dim 64 \
  --hidden-size 512 \
  --block-size 384 \
  --gen-prompt $'Human: Explain gravity in simple terms.\n\nAssistant:' \
  --gen-max-new-tokens 200 \
  --gen-temperature 1.0 \
  --gen-top-k 5 \
  --use-compile \
  --use-kv-cache
```

### 6) HF parity check
```bash
python -m nanoqwen.compare_hf \
  --prompt "Hello from Qwen" \
  --max-new-tokens 120 \
  --top-k 50 \
  --temperature 1.0 \
  --seed 81 \
  --device-mode cpu \
  --hf-dtype float32
```

## Notes
- Use `--resume-from` with either pretrain or SFT checkpoints to continue training.
- On low-VRAM GPUs, reduce `--block-size` first, then `--batch-size`.
- `torch.compile` can improve throughput significantly on this setup.
- Built this project inspired from Karapathy's nanoGPT and nanoChat for my own understanding of the concepts and as a hands-on implementation to study Qwen architecture and LLM training workflows from scratch. Used codex-5.3 for assistance and as instructor to learn concepts, and implementation and debugging


## To Do
- Separate evaluation pipeline
- Add CLI options for Karapathy's nanoGPT code in misc folder
- Write a blog explaining my understanding of the concepts realted to Qwen architecture
- Training and SFT for longer steps and add the plots