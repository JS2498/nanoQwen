import argparse
import time
from pathlib import Path
import math

import torch

from nanoqwen.data import HFTokenDataModule
from nanoqwen.qwen3 import Qwen, QwenConfig


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_token_budget(
    train_tokens: int,
    val_tokens: int,
    batch_size: int,
    block_size: int,
    max_steps: int,
) -> dict[str, float]:
    tokens_per_step = int(batch_size * block_size)
    planned_train_tokens = int(tokens_per_step * max_steps)
    dataset_total_tokens = int(train_tokens + val_tokens)
    train_coverage = (planned_train_tokens / max(train_tokens, 1)) * 100.0
    train_epochs_equivalent = planned_train_tokens / max(train_tokens, 1)
    return {
        "train_tokens": float(train_tokens),
        "val_tokens": float(val_tokens),
        "dataset_total_tokens": float(dataset_total_tokens),
        "tokens_per_step": float(tokens_per_step),
        "planned_train_tokens": float(planned_train_tokens),
        "train_coverage_percent": train_coverage,
        "train_epochs_equivalent": train_epochs_equivalent,
    }


def build_lr(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int,
    scheduler: str,
    min_lr_ratio: float,
) -> float:
    # Linear warmup: 0 -> base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if scheduler == "constant":
        return base_lr

    # Cosine decay after warmup: base_lr -> min_lr
    min_lr = base_lr * min_lr_ratio
    if max_steps <= warmup_steps:
        return min_lr
    decay_steps = max_steps - warmup_steps
    decay_progress = min(max(step - warmup_steps, 0) / max(decay_steps, 1), 1.0)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine_coeff


def save_checkpoint(
    ckpt_path: Path,
    model: Qwen,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )


def load_checkpoint(
    ckpt_path: Path,
    model: Qwen,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only yet.
        checkpoint = torch.load(ckpt_path, map_location=device)
    model_state = checkpoint["model_state_dict"]
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(model_state.keys())
    model_has_orig = any(k.startswith("_orig_mod.") for k in model_keys)
    ckpt_has_orig = any(k.startswith("_orig_mod.") for k in ckpt_keys)

    if ckpt_has_orig and not model_has_orig:
        model_state = {k.replace("_orig_mod.", "", 1): v for k, v in model_state.items()}
    elif model_has_orig and not ckpt_has_orig:
        model_state = {f"_orig_mod.{k}": v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # resume from next step
    return int(checkpoint["step"]) + 1


def generate_text(
    model: Qwen,
    dm: HFTokenDataModule,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    device: str,
    seed: int,
    use_kv_cache: bool = False,
) -> str:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_ids = dm.encode(prompt)
    max_ctx = model.config.max_position_embeddings
    if len(prompt_ids) == 0:
        raise ValueError("gen_prompt tokenized to an empty sequence")

    # Keep only the last max_ctx tokens if prompt is longer than context window.
    x = torch.tensor([prompt_ids[-max_ctx:]], dtype=torch.long, device=device)

    t_gen_start = time.perf_counter()
    per_token_times: list[float] = []
    ttft_ms: float | None = None


    # sync and reset CUDA peak memory so measurement is only this generation.
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    if not use_kv_cache:
        for step in range(max_new_tokens):
            if device == "cuda":
                torch.cuda.synchronize()
            t_step0 = time.perf_counter()
            
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

            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_step0
            per_token_times.append(dt)

            if step == 0:
                ttft_ms = (time.perf_counter() - t_gen_start) * 1000.0

        decode_tokens = len(per_token_times)
        decode_time = sum(per_token_times)
        decode_tok = decode_tokens / max(decode_time, 1e-9)

        peak_mem_mib = -1.0
        if device == "cuda":
            peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

        mode = "kv_cache_on" if use_kv_cache else "kv_cache_off"

        print(
            f"\nGeneration Metrics ({mode})\n"
            f"  TTFT               : {ttft_ms:8.2f} ms\n"
            f"  Decode Tokens      : {decode_tokens:8d} tokens\n"
            f"  Decode Time        : {decode_time:8.4f} s\n"
            f"  Decode Throughput  : {decode_tok:8.2f} tokens/s\n"
            f"  Peak GPU Memory    : {peak_mem_mib:8.2f} MiB\n"
        )
        return dm.decode(x[0].tolist())
    
    # prefill KV cache with prompt
    with torch.no_grad():
        logits, _, past_key_values = model(x, use_cache=True)
    
    for step in range(max_new_tokens):
        if device == "cuda":
            torch.cuda.synchronize()
        t_step0 = time.perf_counter()
        
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

        with torch.no_grad():
            logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step0
        per_token_times.append(dt)

        if step == 0:
            ttft_ms = (time.perf_counter() - t_gen_start) * 1000.0


    decode_tokens = len(per_token_times)
    decode_time = sum(per_token_times)
    decode_tok = decode_tokens / max(decode_time, 1e-9)


    peak_mem_mib = -1.0
    if device == "cuda":
        peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    mode = "kv_cache_on" if use_kv_cache else "kv_cache_off"
    # make this print more human readable with units and aligned decimals and asthetic 
    print(
        f"\nGeneration Metrics ({mode})\n"
        f"  TTFT               : {ttft_ms:8.2f} ms\n"
        f"  Decode Tokens      : {decode_tokens:8d} tokens\n"
        f"  Decode Time        : {decode_time:8.4f} s\n"
        f"  Decode Throughput  : {decode_tok:8.2f} tokens/s\n"
        f"  Peak GPU Memory    : {peak_mem_mib:8.2f} MiB\n"
    )

    return dm.decode(x[0].tolist())


@torch.no_grad()
def estimate_val_loss(
    model: Qwen,
    dm: HFTokenDataModule,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = dm.get_batch("val", batch_size, block_size, device=device)
        if device == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(xb, yb)
        else:
            _, loss = model(xb, yb)
        losses.append(loss.item())

    mean_loss = float(sum(losses) / max(len(losses), 1))
    ppl = float(math.exp(mean_loss)) if mean_loss < 20 else float("inf")
    model.train()
    return mean_loss, ppl


def train(args: argparse.Namespace) -> None:
    if args.log_interval <= 0:
        raise ValueError("log_interval must be > 0")
    if args.save_interval <= 0:
        raise ValueError("save_interval must be > 0")

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
        file_name=args.train_file_name,
        val_file_name=args.val_file_name,
        train_split=args.train_split,
        seed=args.seed,
        token_cache_dir=args.token_cache_dir,
        rebuild_token_cache=args.rebuild_token_cache,
    )

    budget = compute_token_budget(
        train_tokens=len(dm.train_data),
        val_tokens=len(dm.val_data),
        batch_size=args.minibatch_size * args.grad_accum_steps,
        block_size=args.block_size,
        max_steps=args.max_steps,
    )

    print(
        "\nToken Budget\n"
        f"  Dataset Total      : {budget['dataset_total_tokens']/1e6:8.2f} M tokens\n"
        f"  Train Split        : {budget['train_tokens']/1e6:8.2f} M tokens\n"
        f"  Val Split          : {budget['val_tokens']/1e6:8.2f} M tokens\n"
        f"  Effective Batch    : {args.minibatch_size * args.grad_accum_steps:8d}\n"
        f"  Tokens / Step      : {int(budget['tokens_per_step']):8d} tokens\n"
        f"  Planned Train      : {budget['planned_train_tokens']/1e6:8.2f} M tokens\n"
        f"  Train Coverage     : {budget['train_coverage_percent']:8.3f} %\n"
        f"  Epochs Equivalent  : {budget['train_epochs_equivalent']:8.6f} x\n"
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
    if args.use_compile:
        model = torch.compile(model)
    model.train()
    total_params, trainable_params = count_parameters(model)
    print(
        "model_params: "
        f"total={total_params:,} ({total_params/1e6:.2f}M) | "
        f"trainable={trainable_params:,} ({trainable_params/1e6:.2f}M)"
    )

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
                id=args.wandb_run_id,
                resume=args.wandb_resume,
                config=vars(args),
            )
        except Exception as e:
            print(f"wandb disabled: {e}")
            wandb_run = None

    step_digits = len(str(args.max_steps))
    start_step = 0

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_step = load_checkpoint(resume_path, model, optimizer, device)
        print(f"resumed_from: {resume_path} | start_step: {start_step}")



    for step in range(start_step, args.max_steps):
        t0 = time.perf_counter()

        lr = build_lr(
            step=step,
            max_steps=args.max_steps,
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            scheduler=args.lr_scheduler,
            min_lr_ratio=args.min_lr_ratio,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_list = []

        for _ in range(args.grad_accum_steps):
            xb, yb = dm.get_batch("train", args.minibatch_size, args.block_size, device=device)

            if device == "cuda" and args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(xb, yb)
                    loss = loss/args.grad_accum_steps  # scaling by 1/steps so that the average loss is used for gradient
                    loss.backward()
            else:
                _, loss = model(xb, yb)
                loss = loss/args.grad_accum_steps
                loss.backward()

            loss_list.append(float(loss.item()))

        optimizer.step()
        train_loss = sum(loss_list)

        # without gradient accumulation
        # xb, yb = dm.get_batch("train", args.batch_size, args.block_size, device=device)

        # optimizer.zero_grad(set_to_none=True)

        # if device == "cuda" and args.use_amp:
        #     with torch.autocast(device_type="cuda", dtype=torch.float16):
        #         _, loss = model(xb, yb)
        # else:
        #     _, loss = model(xb, yb)
        # loss.backward()
        # optimizer.step()

        dt = (time.perf_counter() - t0)
        dt_ms = dt * 1000
        tokens = args.minibatch_size * args.grad_accum_steps * args.block_size
        tok_per_sec = tokens / max(dt, 1e-9)
        global_step = step + 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": step,
                    "global_step": global_step,
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/tokens_seen": global_step * args.minibatch_size * args.grad_accum_steps * args.block_size,
                },
                step=global_step,
            )

        step_str = f"{step:0{step_digits}d}/{args.max_steps:0{step_digits}d}"
        if (step % args.log_interval == 0) or (step == args.max_steps - 1):
            val_loss, val_ppl = estimate_val_loss(
                model=model,
                dm=dm,
                batch_size=args.minibatch_size * args.grad_accum_steps,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                device=device,
                use_amp=args.use_amp,
            )
            print(
                f"step: {step_str:<{(step_digits * 2) + 1}} | "
                f"train_loss: {train_loss:>9.6f} | "
                f"val_loss: {val_loss:>9.6f} | "
                f"lr: {lr:>8.2e} | "
                f"time: {dt_ms:>6.2f}ms | "
                f"tokens_per_sec: {tok_per_sec:>9.2f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                    },
                    step=global_step,
                )
        if ((step + 1) % args.save_interval == 0) or (step == args.max_steps - 1):
            ckpt_file = Path(args.checkpoint_dir) / f"train_step_{step+1:06d}.pt"
            save_checkpoint(ckpt_file, model, optimizer, step, args)
            if wandb_run is not None:
                wandb_run.log({"checkpoint/step": step + 1}, step=global_step)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
    generated = generate_text(
        model=model,
        dm=dm,
        prompt=args.gen_prompt,
        max_new_tokens=args.gen_max_new_tokens,
        top_k=args.gen_top_k,
        temperature=args.gen_temperature,
        device=device,
        seed=args.seed,
        use_kv_cache=args.use_kv_cache,
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
    parser.add_argument("--file-name", type=str, default=None, help="Deprecated alias for --train-file-name")
    parser.add_argument("--train-file-name", type=str, default="input.txt")
    parser.add_argument("--val-file-name", type=str, default=None)
    parser.add_argument("--token-cache-dir", type=str, default="data/cache")
    parser.add_argument("--rebuild-token-cache", action="store_true")
    parser.add_argument("--train-split", type=float, default=0.9) 
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--minibatch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)

    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gen-prompt", type=str, default="Hello from nanoQwen")
    parser.add_argument("--gen-max-new-tokens", type=int, default=80)
    parser.add_argument("--gen-top-k", type=int, default=50)
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--use-kv-cache", action="store_true", help="Use KV cache for faster autoregressive decoding during generation")

    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--use-amp", action="store_true")  # for mixed precision training on CUDA (autocast)
    
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nanoqwen")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
    )
    args = parser.parse_args()
    if args.file_name is not None:
        args.train_file_name = args.file_name
    train(args)


# Things to do from Tuesday (Check Obsidian notes for more details):
# 3. Blog with your own writeup and analysis of the training process and results
# 6. Include the RLHF training loop and evaluate the model after RLHF training as well
