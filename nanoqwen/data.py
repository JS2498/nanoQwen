import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _iter_line_batches(path: Path, batch_size: int = 1024) -> Iterator[list[str]]:
    batch: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            batch.append(line.rstrip("\n"))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


class HFTokenDataModule:
    """HF-tokenizer data module with disk token cache + memmap batching.

    If val_file_name is provided, train/val are loaded from separate files.
    Otherwise, data is loaded from one file and split by train_split.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        data_dir: str = "data",
        file_name: str = "input.txt",
        val_file_name: str | None = None,
        train_split: float = 0.9,
        seed: int = 1337,
        token_cache_dir: str = "data/cache",
        rebuild_token_cache: bool = False,
    ) -> None:
        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1")

        self.data_path = Path(data_dir) / file_name
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.val_data_path = Path(data_dir) / val_file_name if val_file_name else None
        if self.val_data_path is not None and not self.val_data_path.exists():
            raise FileNotFoundError(f"Validation data file not found: {self.val_data_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use full tokenizer length (includes added/special tokens) for safe embedding size.
        self.vocab_size = len(self.tokenizer)
        self.train_split = train_split

        self.cache_dir = Path(token_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        if self.val_data_path is not None:
            train_bin, train_count = self._build_or_load_cache(self.data_path, "train", rebuild_token_cache)
            val_bin, val_count = self._build_or_load_cache(self.val_data_path, "val", rebuild_token_cache)
            self.train_data = np.memmap(train_bin, dtype=np.uint32, mode="r", shape=(train_count,))
            self.val_data = np.memmap(val_bin, dtype=np.uint32, mode="r", shape=(val_count,))
        else:
            full_bin, full_count = self._build_or_load_cache(self.data_path, "full", rebuild_token_cache)
            full_data = np.memmap(full_bin, dtype=np.uint32, mode="r", shape=(full_count,))
            split_idx = int(full_count * self.train_split)
            self.train_data = full_data[:split_idx]
            self.val_data = full_data[split_idx:]

        if len(self.train_data) < 2 or len(self.val_data) < 2:
            raise ValueError("Train/val split too small after tokenization")

        print(
            f"token_cache: train_tokens={len(self.train_data):,} | "
            f"val_tokens={len(self.val_data):,}"
        )

    def _cache_prefix(self, src_path: Path, split_name: str) -> Path:
        tok_name = _sanitize_name(self.tokenizer.name_or_path)
        stem = f"{src_path.stem}.{split_name}.{tok_name}"
        return self.cache_dir / stem

    def _build_or_load_cache(self, src_path: Path, split_name: str, rebuild: bool) -> tuple[Path, int]:
        prefix = self._cache_prefix(src_path, split_name)
        bin_path = prefix.with_suffix(".tokens.bin")
        meta_path = prefix.with_suffix(".tokens.meta.json")

        if (not rebuild) and bin_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return bin_path, int(meta["num_tokens"])

        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        num_tokens = 0

        with bin_path.open("wb") as fout:
            for lines in _iter_line_batches(src_path, batch_size=1024):
                enc = self.tokenizer(lines, add_special_tokens=False, verbose=False)
                for ids in enc["input_ids"]:
                    if ids:
                        arr = np.asarray(ids, dtype=np.uint32)
                        arr.tofile(fout)
                        num_tokens += int(arr.size)
                    if newline_ids:
                        nl = np.asarray(newline_ids, dtype=np.uint32)
                        nl.tofile(fout)
                        num_tokens += int(nl.size)

        meta = {
            "source_file": str(src_path),
            "split_name": split_name,
            "tokenizer": self.tokenizer.name_or_path,
            "dtype": "uint32",
            "num_tokens": num_tokens,
            "bin_path": str(bin_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return bin_path, num_tokens

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
        n = len(data)
        if n <= block_size:
            raise ValueError(f"block_size={block_size} is too large for {split} split of length {n}")

        starts = torch.randint(
            low=0,
            high=n - block_size,
            size=(batch_size,),
            generator=self._rng,
        ).tolist()

        x_np = np.empty((batch_size, block_size), dtype=np.int64)
        y_np = np.empty((batch_size, block_size), dtype=np.int64)

        for i, s in enumerate(starts):
            x_np[i] = data[s : s + block_size]
            y_np[i] = data[s + 1 : s + block_size + 1]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x.to(device), y.to(device)


__all__ = ["HFTokenDataModule"]
