# nanoQwen

## What this repo does
This repo is a from-scratch PyTorch implementation of a Qwen-style decoder model and training workflow.
It includes:
- model implementation (`RMSNorm`, `RoPE`, `GQA`, gated MLP)
- loading Hugging Face Qwen weights for parity checks
- local training on `data/input.txt` (Tiny Shakespeare)
- generation and HF-vs-local comparison CLIs

## What is implemented
- Qwen-style model in `nanoqwen/qwen3.py`
- HF weight loading for `Qwen/Qwen3-0.6B`
- Parity script for logits and token generation comparison
- Training loop with:
  - AdamW
  - warmup LR
  - optional `torch.compile`
  - optional AMP (`autocast`)
  - periodic validation loss/perplexity
  - periodic checkpoint save
  - resume from checkpoint
  - optional Weights & Biases logging
- Post-training text generation

## Repository structure
- `nanoqwen/qwen3.py`: model + HF loading
- `nanoqwen/data.py`: tokenizer-based dataset + batch sampling
- `nanoqwen/train.py`: train/eval/checkpoint/resume/W&B CLI
- `nanoqwen/generate.py`: generation CLI
- `nanoqwen/compare_hf.py`: local vs HF parity CLI
- `data/input.txt`: Tiny Shakespeare dataset
- `misc/`: GPT-2 implementation and feature experiments inspired by Karpathy's nanoGPT walkthrough

## Setup
Create local env:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Available CLIs

### 1) Train
```bash
python -m nanoqwen.train \
  --file-name input.txt \
  --device cuda \
  --max-steps 500 \
  --log-interval 20 \
  --eval-iters 20 \
  --save-interval 100 \
  --checkpoint-dir checkpoints
```

Resume from checkpoint:
```bash
python -m nanoqwen.train \
  --file-name input.txt \
  --device cuda \
  --resume-from checkpoints/step_000500.pt \
  --max-steps 1000
```

With W&B:
```bash
python -m nanoqwen.train \
  --file-name input.txt \
  --device cuda \
  --use-wandb \
  --wandb-project nanoqwen \
  --wandb-run-name tinyshakespeare-baseline
```

### 2) Generate
```bash
python -m nanoqwen.generate \
  --prompt "Explain grouped-query attention" \
  --max-new-tokens 120 \
  --top-k 50 \
  --temperature 0.9
```

### 3) Compare local model vs HF
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
