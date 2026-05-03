import argparse

import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoqwen.qwen3 import Qwen


def sample_next_token(
    logits: torch.Tensor,
    top_k: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0.0:
        # True greedy decoding for deterministic comparison.
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    k = min(top_k, probs.size(-1))
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    sampled = torch.multinomial(topk_probs, num_samples=1, generator=generator)
    return torch.gather(topk_indices, -1, sampled)


def generate_local(
    model: Qwen,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    x = input_ids.clone()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            next_token = sample_next_token(logits[:, -1, :], top_k, temperature, generator)
            x = torch.cat((x, next_token), dim=1)
    return x


def generate_hf(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    x = input_ids.clone()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x).logits
            next_token = sample_next_token(logits[:, -1, :], top_k, temperature, generator)
            x = torch.cat((x, next_token), dim=1)
    return x


def compare(
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
    device_mode: str,
    hf_dtype: str,
):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    local_model = Qwen.from_pretrained(model_name).eval()
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    # without the dtype match the ouput logits can be very different 
    # and the generation can diverge quickly, so we will just load the HF model in float32 for parity checks
    # below is the max and mean abs diff when loading the HF model without dtype match:
    # max_abs_diff: 0.479673
    # mean_abs_diff: 0.049872

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype_map[hf_dtype]
    ).eval()

    has_cuda = torch.cuda.is_available()
    if device_mode == "cpu" or not has_cuda:
        local_device = "cpu"
        hf_device = "cpu"
    else:
        local_device = "cuda"
        hf_device = "cuda"

    local_model.to(local_device)
    hf_model.to(hf_device)

    inputs_cpu = tokenizer(prompt, return_tensors="pt").input_ids
    local_inputs = inputs_cpu.to(local_device)
    hf_inputs = inputs_cpu.to(hf_device)

    with torch.no_grad():
        local_logits, _ = local_model(local_inputs)
        hf_logits = hf_model(hf_inputs).logits

    diff = (local_logits.cpu() - hf_logits.cpu()).abs()
    print(f"prompt: {prompt}")
    print(f"device_mode: {device_mode} (local={local_device}, hf={hf_device})")
    print(f"hf_dtype: {hf_dtype}")
    print(f"max_abs_diff: {diff.max().item():.6f}")
    print(f"mean_abs_diff: {diff.mean().item():.6f}")



    local_ids = generate_local(local_model, local_inputs, max_new_tokens, top_k, temperature, seed)
    hf_ids = generate_hf(hf_model, hf_inputs, max_new_tokens, top_k, temperature, seed)

    local_ids_cpu = local_ids.cpu()
    hf_ids_cpu = hf_ids.cpu()
    local_new = local_ids_cpu[:, inputs_cpu.size(1):]
    hf_new = hf_ids_cpu[:, inputs_cpu.size(1):]
    token_matches = (local_new == hf_new).sum().item()
    total = local_new.numel()
    match_pct = 100.0 * token_matches / max(total, 1)

    print(f"seed: {seed}")
    print(f"max_new_tokens: {max_new_tokens}, top_k: {top_k}, temperature: {temperature}")
    print(f"token_match: {token_matches}/{total} ({match_pct:.2f}%)")
    print("\n--- Local output ---")
    print(tokenizer.decode(local_ids_cpu[0], skip_special_tokens=True))
    print("\n--- HF output ---")
    print(tokenizer.decode(hf_ids_cpu[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare local Qwen vs HF logits")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument(
        "--device-mode",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="cpu=both on cpu, cuda=both on cuda",
    )
    parser.add_argument(
        "--hf-dtype",
        type=str,
        default="float32",
        choices=["float32", "auto", "bfloat16", "float16"],
        help="dtype to load HF baseline model for parity checks",
    )
    args = parser.parse_args()
    compare(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
        device_mode=args.device_mode,
        hf_dtype=args.hf_dtype,
    )
