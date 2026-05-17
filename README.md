# nanoQwen

Minimal Qwen-style LLM project in PyTorch with:
- decoder-only model implementation
- HF weight loading + parity checks
- pretraining loop
- SFT data pipeline + SFT training loop
- checkpointing/resume/generation workflows

## Implemented
- `nanoqwen/qwen3.py`
  - Qwen-style architecture (`RoPE`, `RMSNorm`, `GQA`, gated MLP)
  - weight tying (`embed_tokens` and `lm_head`)
  - HF load path for `Qwen/Qwen3-0.6B`
- `nanoqwen/data.py`
  - pretraining token cache + memmap batches
- `nanoqwen/train.py`
  - pretraining loop with AdamW, warmup + cosine/constant LR, val loss/perplexity
  - checkpoint save/load, resume, generation
  - optional KV-cache decoding for faster autoregressive inference
  - optional `torch.compile`, optional AMP, optional W&B
- `nanoqwen/sft_data.py`
  - JSONL SFT ingestion (`prompt`/`response`)
  - label masking (`ignore_index=-100`) + next-token alignment
  - memmap cache for train/val
- `nanoqwen/train_sft.py`
  - SFT loop with val loss, LR schedule, optional compile/AMP
  - load pretrained checkpoint (including vocab-size/compile-prefix compatibility)
- `nanoqwen/compare_hf.py`
  - local vs HF logits/token comparison
- `scripts/download_dataset.py`
  - generic plain-text dataset download (`train`/`validation`/`both`)
- `scripts/download_sftdata.py`
  - SFT JSONL dataset download/split
- `scripts/sft_length_stats.py`
  - tokenized length stats for choosing sequence length

## Repo layout
- `nanoqwen/`: model + training + data pipeline code
- `scripts/`: dataset prep scripts
- `data/`: local datasets and caches
- `docs/`: sample outputs and notes
- `misc/`: GPT-2 code and experiments from Karpathy nanoGPT-style learning

## Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Main commands

### 1) Download pretraining data (TinyStories)
```bash
python scripts/download_dataset.py \
  --dataset roneneldan/TinyStories \
  --split both \
  --text-field text \
  --out data/tinystories
```

### 2) Pretrain
```bash
python -m nanoqwen.train \
  --train-file-name tinystories_train.txt \
  --val-file-name tinystories_val.txt \
  --device cuda \
  --max-steps 5000 \
  --block-size 360 \
  --use-compile \
  --use-wandb \
  --wandb-project nanoqwen \
  --wandb-run-name tinystories-pretrain
```

### 3) Download SFT data (Dahoas/sft-static)
```bash
python scripts/download_sftdata.py \
  --dataset Dahoas/sft-static \
  --split both \
  --prompt-field prompt \
  --response-field response \
  --out data/sft_static
```

### 4) SFT from pretrained checkpoint
```bash
python -m nanoqwen.train_sft \
  --data-dir data \
  --sft-train-file sft_static_train.jsonl \
  --sft-val-file sft_static_val.jsonl \
  --pretrained-ckpt checkpoints/pretrain_tinystories_step5000.pt \
  --device cuda \
  --use-compile \
  --batch-size 2 \
  --block-size 256 \
  --max-steps 5000
```

### 5) Generate from checkpoint
```bash
python -m nanoqwen.train \
  --train-file-name tinystories_train.txt \
  --val-file-name tinystories_val.txt \
  --device cuda \
  --max-steps 0 \
  --resume-from checkpoints/sft_dahoas_step5000.pt \
  --gen-prompt $'Human: Explain gravity in simple terms.\n\nAssistant:' \
  --gen-max-new-tokens 200 \
  --gen-temperature 0.0 \
  --gen-top-k 1
```

### 5b) Generate with KV cache
```bash
python -m nanoqwen.train \
  --train-file-name tinystories_train.txt \
  --val-file-name tinystories_val.txt \
  --device cuda \
  --max-steps 0 \
  --resume-from checkpoints/sft_dahoas_100m_step15000.pt \
  --n-layer 10 \
  --n-head 8 \
  --n-kv-head 4 \
  --head-dim 64 \
  --hidden-size 512 \
  --block-size 256 \
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

## Performance snapshot
Hardware used: NVIDIA GeForce GTX 1650 Ti (~4GB VRAM), CUDA 12.2.

| Mode | Avg step time (ms) | Avg tokens/sec |
|---|---:|---:|
| Baseline | ~43.0 | ~1536 |
| `--use-compile` | ~19.2 | ~3251 |
| `--use-amp` | ~60.3 | ~1064 |
| `--use-compile --use-amp` | ~46.8 | ~1366 |

On this setup, `torch.compile` gave the best throughput.

## KV cache inference snapshot
Measured on the same hardware with the same checkpoint/prompt/sampling settings (`max_new_tokens=200`, `temperature=1.0`, `top_k=5`).

| Mode | TTFT (ms) | Decode time (s) | Decode throughput (tokens/s) | Peak GPU memory (MiB) |
|---|---:|---:|---:|---:|
| `kv_cache_off` | 4873.41 | 28.5895 | 7.00 | 1503.60 |
| `kv_cache_on` | 9192.01 | 12.4653 | 16.04 | 1271.27 |

- Decode throughput speedup with KV cache: `16.04 / 7.00 = 2.29x`
- In this run, TTFT increased with cache mode (likely prefill and/or compile warmup effects), but decode throughput improved significantly.
