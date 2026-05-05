import argparse
import time

import torch

from nanoqwen.data import HFTokenDataModule
from nanoqwen.qwen3 import Qwen, QwenConfig


def build_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    return base_lr


def generate_text(
    model: Qwen,
    dm: HFTokenDataModule,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    device: str,
    seed: int,
) -> str:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_ids = dm.encode(prompt)
    max_ctx = model.config.max_position_embeddings
    if len(prompt_ids) == 0:
        raise ValueError("gen_prompt tokenized to an empty sequence")

    # Keep only the last max_ctx tokens if prompt is longer than context window.
    x = torch.tensor([prompt_ids[-max_ctx:]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Rolling context window for autoregressive decoding.
            x_cond = x[:, -max_ctx:]
            logits, _ = model(x_cond)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                k = min(top_k, probs.size(-1))
                topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
                sampled = torch.multinomial(topk_probs, 1, generator=generator)
                next_token = torch.gather(topk_idx, -1, sampled)

            x = torch.cat((x, next_token), dim=1)

    return dm.decode(x[0].tolist())


def train(args: argparse.Namespace) -> None:
    if args.log_interval <= 0:
        raise ValueError("log_interval must be > 0")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dm = HFTokenDataModule(
        model_name=args.model_name,
        data_dir=args.data_dir,
        file_name=args.file_name,
        train_split=args.train_split,
        seed=args.seed,
    )

    config = QwenConfig(
        max_position_embeddings=args.block_size,
        vocab_size=dm.vocab_size,
        hidden_size=args.hidden_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_kv_head=args.n_kv_head,
        head_dim=args.head_dim,
    )
    model = Qwen(config).to(device)
    model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"wandb disabled: {e}")
            wandb_run = None

    step_digits = len(str(args.max_steps))

    for step in range(args.max_steps):
        t0 = time.perf_counter()

        lr = build_lr(step, args.max_steps, args.learning_rate, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xb, yb = dm.get_batch("train", args.batch_size, args.block_size, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        dt = (time.perf_counter() - t0)
        dt_ms = dt * 1000
        tokens = args.batch_size * args.block_size
        tok_per_sec = tokens / max(dt, 1e-9)
        global_step = step + 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "global_step": global_step,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_per_sec,
                },
                step=global_step,
            )

        step_str = f"{step:0{step_digits}d}/{args.max_steps:0{step_digits}d}"
        if (step % args.log_interval == 0) or (step == args.max_steps - 1):
            print(
                f"step: {step_str:<{(step_digits * 2) + 1}} | "
                f"loss: {loss.item():>9.6f} | "
                f"lr: {lr:>8.2e} | "
                f"time: {dt_ms:>6.2f}ms | "
                f"tokens_per_sec: {tok_per_sec:>9.2f}"
            )
        

    generated = generate_text(
        model=model,
        dm=dm,
        prompt=args.gen_prompt,
        max_new_tokens=args.gen_max_new_tokens,
        top_k=args.gen_top_k,
        temperature=args.gen_temperature,
        device=device,
        seed=args.seed,
    )
    print("\n=== Generated Text ===")
    print(generated)
    if wandb_run is not None:
        wandb_run.log({"generation/text": generated}, step=args.max_steps)
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nanoQwen from scratch on a local text file")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B", help="HF model name for tokenizer")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--file-name", type=str, default="input.txt")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--gen-prompt", type=str, default="Hello from nanoQwen")
    parser.add_argument("--gen-max-new-tokens", type=int, default=80)
    parser.add_argument("--gen-top-k", type=int, default=50)
    parser.add_argument("--gen-temperature", type=float, default=1.0)

    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nanoqwen")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    train(args)


# Things to do from Tuesday (Check Obsidian notes for more details):
# 1. Train it on some useful dataset
# 2. Evaluation at certain intervals during training
# 3. Blog with your own writeup and analysis of the training process and results
# 4. Saving the checkpoint and retraining from it, and sharing the checkpoint for others to load and use
# 5. Implement SFT and evaluate the model
# 6. Include the RLHF training loop and evaluate the model after RLHF training as well
