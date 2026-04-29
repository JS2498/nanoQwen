# nanoQwen

nanoQwen is a step-by-step educational rebuild of a small Qwen-style language model trainer, inspired by nanoGPT and nanochat. The goal is to implement each core LLM component from first principles, commit it, and only then move to the next feature.

This project is intentionally starting as a skeleton. No training or model code has been added yet.

## Blog

Detailed concept notes will live in the companion blog:

- Blog link: TODO - add your blog URL here

Each implementation milestone should link back to the relevant blog section once the post is available.

## Project Goals

- Build a tokenizer pipeline.
- Implement RMSNorm.
- Implement RoPE positional embeddings.
- Implement grouped-query attention.
- Add KV cache support for fast generation.
- Build a minimal training loop.
- Build an inference script.
- Keep the code readable enough to teach from.
- Push the project before adding each new feature.

## Skeleton Layout

```text
nanoQwen/
  README.md
  docs/
  data/
  nanoqwen/
  scripts/
  tests/
  checkpoints/
```

Planned responsibilities:

- `docs/`: design notes, milestone notes, and blog cross-links.
- `data/`: local datasets or prepared tokenizer artifacts.
- `nanoqwen/`: future Python package for model, tokenizer, and generation code.
- `scripts/`: future CLI scripts for training, tokenization, and inference.
- `tests/`: future focused tests for each implemented feature.
- `checkpoints/`: local model checkpoints, usually ignored by git once git is set up.

## Step-by-Step Roadmap

### 0. Repository Setup

Status: current step.

Deliverables:

- Create the `nanoQwen` folder.
- Add this README.
- Add empty scaffold directories.
- Initialize or confirm git remote before the first implementation step.

Checkpoint:

- Commit/push message: `chore: scaffold nanoQwen`

### 1. Tokenizer Pipeline

Goal: prepare text data and build the encode/decode path.

Planned deliverables:

- Decide tokenizer strategy: byte-level, BPE, or SentencePiece-style.
- Add dataset preparation script.
- Add tokenizer training or loading path.
- Add encode/decode smoke tests.
- Document vocabulary size, special tokens, and tradeoffs.

Checkpoint:

- Commit/push before moving to RMSNorm.

### 2. RMSNorm

Goal: implement Qwen-style normalization.

Planned deliverables:

- Add RMSNorm module.
- Add shape and numerical behavior tests.
- Compare briefly against LayerNorm in docs.

Checkpoint:

- Commit/push before moving to RoPE.

### 3. RoPE

Goal: add rotary positional embeddings.

Planned deliverables:

- Implement frequency cache construction.
- Apply RoPE to attention query/key tensors.
- Add tests for shape, dtype, device movement, and cache slicing.
- Document why RoPE helps extrapolate position information.

Checkpoint:

- Commit/push before moving to GQA.

### 4. Grouped-Query Attention

Goal: implement attention where query heads outnumber key/value heads.

Planned deliverables:

- Add attention projection shapes for `n_heads` and `n_kv_heads`.
- Repeat or expand KV heads correctly for attention.
- Add causal mask support.
- Add tests for valid head configurations.
- Document the memory and latency motivation.

Checkpoint:

- Commit/push before moving to transformer block integration.

### 5. Transformer Block And Model Skeleton

Goal: assemble embeddings, attention, MLP, RMSNorm, and logits.

Planned deliverables:

- Add config object.
- Add decoder block.
- Add model forward pass.
- Add parameter count utility.
- Add tiny overfit test target.

Checkpoint:

- Commit/push before moving to KV cache.

### 6. KV Cache

Goal: support efficient autoregressive generation.

Planned deliverables:

- Add cache structure for keys and values.
- Update attention forward path for prefill and decode.
- Add generation consistency tests between cached and uncached paths.
- Document prefill vs decode.

Checkpoint:

- Commit/push before moving to training.

### 7. Training Loop

Goal: train the model on a small local corpus.

Planned deliverables:

- Add dataloader and batching.
- Add optimizer setup.
- Add loss logging.
- Add checkpoint save/load.
- Add simple validation loop.
- Add reproducible tiny training command.

Checkpoint:

- Commit/push before moving to inference.

### 8. Inference Script

Goal: load a checkpoint and generate text.

Planned deliverables:

- Add prompt-based generation CLI.
- Add temperature and top-k or top-p controls.
- Use KV cache during generation.
- Document example commands and expected behavior.

Checkpoint:

- Commit/push before cleanup.

### 9. Polish And Learning Notes

Goal: make the project easy to read and explain.

Planned deliverables:

- Add architecture diagram or high-level flow.
- Add links from README to blog sections.
- Add common debugging notes.
- Add final end-to-end run instructions.

Checkpoint:

- Commit/push final tutorial baseline.

## Working Rule

For every feature:

1. Start from a clean working tree.
2. Implement only one concept.
3. Add or update focused tests.
4. Update README/docs with what changed.
5. Run the smallest useful verification.
6. Commit and push.
7. Move to the next concept.

## Next Step

Initialize git or confirm the existing remote, then commit this scaffold. After that, the first implementation feature should be the tokenizer pipeline.
