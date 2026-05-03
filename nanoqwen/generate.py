import argparse

import torch
from transformers import AutoTokenizer

from nanoqwen.qwen3 import Qwen


def generate_text(prompt: str, max_new_tokens: int, top_k: int, temperature: float, seed: int):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen.from_pretrained(model_name=model_name)
    model.eval()

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            sampled = torch.multinomial(topk_probs, 1)
            x_new = torch.gather(topk_indices, -1, sampled)
            x = torch.cat((x, x_new), dim=1)

    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with local Qwen3-0.6B reimplementation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
