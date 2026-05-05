from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer


class HFTokenDataModule:
    """HF-tokenizer based dataset + batching utility for next-token training."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        data_dir: str = "data",
        file_name: str = "input.txt",
        train_split: float = 0.9,
        seed: int = 1337,
    ) -> None:
        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1")

        self.data_path = Path(data_dir) / file_name
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.train_split = train_split

        text = self.data_path.read_text(encoding="utf-8")
        # We tokenize the full corpus once and later train on short blocks from it.
        # Disable tokenizer verbosity to avoid misleading max-length warnings here.
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
        )
        self.tokens = enc.input_ids.squeeze(0).to(torch.long)

        if self.tokens.numel() < 2:
            raise ValueError("Dataset must tokenize to at least 2 tokens")

        n = int(self.tokens.numel() * self.train_split)
        self.train_data = self.tokens[:n]
        self.val_data = self.tokens[n:]

        if self.train_data.numel() < 2 or self.val_data.numel() < 2:
            raise ValueError("Train/val split too small after tokenization")

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError("split must be one of {'train', 'val'}")

        data = self.train_data if split == "train" else self.val_data
        if data.numel() <= block_size:
            raise ValueError(
                f"block_size={block_size} is too large for {split} split of length {data.numel()}"
            )

        starts = torch.randint(
            low=0,
            high=data.numel() - block_size,
            size=(batch_size,),
            generator=self._rng,
        )

        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

        return x.to(device), y.to(device)


__all__ = ["HFTokenDataModule"]
