import json
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _truncate_prompt_response(
    prompt_ids: list[int],
    response_ids: list[int],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Truncate to max_seq_len while preferring to keep response tokens.

    Strategy:
    - If overflow, trim prompt first.
    - If prompt is exhausted and still overflow, trim response from the left.
    """
    total = len(prompt_ids) + len(response_ids)
    if total <= max_seq_len:
        return prompt_ids, response_ids

    overflow = total - max_seq_len

    trim_prompt = min(len(prompt_ids), overflow)
    prompt_ids = prompt_ids[trim_prompt:]
    overflow -= trim_prompt

    if overflow > 0:
        response_ids = response_ids[overflow:]

    return prompt_ids, response_ids


class SFTMemmapDataModule:
    """Memmap-backed SFT data module.

    Input files must be JSONL rows with:
      {"prompt": "...", "response": "..."}

    It tokenizes once into cache files:
    - <prefix>.input_ids.bin   (int32 concatenated tokens)
    - <prefix>.labels.bin      (int32 concatenated labels with -100 masking)
    - <prefix>.offsets.npy     (int64 array of shape [N,2]: start, length)
    - <prefix>.meta.json       (stats/config)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        data_dir: str = "data",
        train_file_name: str = "sft_static_train.jsonl",
        val_file_name: str | None = None,
        train_split: float = 0.95,
        seed: int = 42,
        max_seq_len: int = 256,
        cache_dir: str = "data/cache_sft",
        rebuild_cache: bool = False,
        prompt_field: str = "prompt",
        response_field: str = "response",
    ) -> None:
        if max_seq_len <= 1:
            raise ValueError("max_seq_len must be > 1")
        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.prompt_field = prompt_field
        self.response_field = response_field

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        data_dir_path = Path(data_dir)
        train_path = data_dir_path / train_file_name
        if not train_path.exists():
            raise FileNotFoundError(f"SFT train file not found: {train_path}")

        val_path = data_dir_path / val_file_name if val_file_name else None
        if val_path is not None and not val_path.exists():
            raise FileNotFoundError(f"SFT val file not found: {val_path}")

        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        if val_path is not None:
            self.train_cache = self._build_or_load_cache(
                src_path=train_path,
                split_name="train",
                cache_root=cache_root,
                rebuild_cache=rebuild_cache,
            )
            self.val_cache = self._build_or_load_cache(
                src_path=val_path,
                split_name="val",
                cache_root=cache_root,
                rebuild_cache=rebuild_cache,
            )
        else:
            # Single-file mode: deterministic split before cache build.
            self.train_cache, self.val_cache = self._build_or_load_split_caches(
                src_path=train_path,
                cache_root=cache_root,
                rebuild_cache=rebuild_cache,
                train_split=train_split,
                seed=seed,
            )

        self.train_input = np.memmap(
            self.train_cache["input_path"], dtype=np.int32, mode="r", shape=(self.train_cache["num_tokens"],)
        )
        self.train_labels = np.memmap(
            self.train_cache["label_path"], dtype=np.int32, mode="r", shape=(self.train_cache["num_tokens"],)
        )
        self.train_offsets = np.load(self.train_cache["offsets_path"], mmap_mode="r")

        self.val_input = np.memmap(
            self.val_cache["input_path"], dtype=np.int32, mode="r", shape=(self.val_cache["num_tokens"],)
        )
        self.val_labels = np.memmap(
            self.val_cache["label_path"], dtype=np.int32, mode="r", shape=(self.val_cache["num_tokens"],)
        )
        self.val_offsets = np.load(self.val_cache["offsets_path"], mmap_mode="r")

        if len(self.train_offsets) == 0 or len(self.val_offsets) == 0:
            raise ValueError("SFT train/val has zero usable samples after preprocessing")

        print(
            f"sft_cache: train_samples={len(self.train_offsets):,} | val_samples={len(self.val_offsets):,} | "
            f"max_seq_len={self.max_seq_len}"
        )

    def _cache_prefix(self, src_path: Path, split_name: str, cache_root: Path) -> Path:
        tok_name = _sanitize_name(self.tokenizer.name_or_path)
        stem = f"{src_path.stem}.{split_name}.{tok_name}.sft"
        return cache_root / stem

    def _cache_paths(self, prefix: Path) -> dict:
        return {
            "input_path": prefix.with_suffix(".input_ids.bin"),
            "label_path": prefix.with_suffix(".labels.bin"),
            "offsets_path": prefix.with_suffix(".offsets.npy"),
            "meta_path": prefix.with_suffix(".meta.json"),
        }

    def _build_or_load_cache(
        self,
        src_path: Path,
        split_name: str,
        cache_root: Path,
        rebuild_cache: bool,
        preloaded_rows: list[dict] | None = None,
    ) -> dict:
        prefix = self._cache_prefix(src_path, split_name, cache_root)
        paths = self._cache_paths(prefix)

        if (
            not rebuild_cache
            and paths["input_path"].exists()
            and paths["label_path"].exists()
            and paths["offsets_path"].exists()
            and paths["meta_path"].exists()
        ):
            return json.loads(paths["meta_path"].read_text(encoding="utf-8"))

        rows_iter = preloaded_rows if preloaded_rows is not None else _iter_jsonl(src_path)

        offsets: list[tuple[int, int]] = []
        token_cursor = 0
        num_samples = 0

        with paths["input_path"].open("wb") as finput, paths["label_path"].open("wb") as flabel:
            for row in rows_iter:
                prompt = row.get(self.prompt_field)
                response = row.get(self.response_field)
                if prompt is None or response is None:
                    continue

                prompt_ids = self.tokenizer.encode(str(prompt).strip(), add_special_tokens=False)
                response_ids = self.tokenizer.encode(str(response).strip(), add_special_tokens=False)
                if len(prompt_ids) == 0 or len(response_ids) == 0:
                    continue

                # Teach explicit stopping behavior for instruction responses.
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    response_ids = response_ids + [eos_id]

                prompt_ids, response_ids = _truncate_prompt_response(prompt_ids, response_ids, self.max_seq_len)
                if len(response_ids) == 0:
                    continue

                input_ids = prompt_ids + response_ids
                p = len(prompt_ids)
                n = len(input_ids)

                # Next-token aligned labels:
                # labels[t] supervises prediction of input_ids[t+1].
                # We only supervise positions whose target token belongs to response/EOS region.
                labels = [IGNORE_INDEX] * n
                for t in range(n - 1):
                    if (t + 1) >= p:
                        labels[t] = input_ids[t + 1]

                inp = np.asarray(input_ids, dtype=np.int32)
                lab = np.asarray(labels, dtype=np.int32)

                inp.tofile(finput)
                lab.tofile(flabel)

                offsets.append((token_cursor, int(inp.size)))
                token_cursor += int(inp.size)
                num_samples += 1

        offsets_arr = np.asarray(offsets, dtype=np.int64)
        np.save(paths["offsets_path"], offsets_arr)

        meta = {
            "source_file": str(src_path),
            "split_name": split_name,
            "tokenizer": self.tokenizer.name_or_path,
            "max_seq_len": self.max_seq_len,
            "num_samples": num_samples,
            "num_tokens": int(token_cursor),
            "input_path": str(paths["input_path"]),
            "label_path": str(paths["label_path"]),
            "offsets_path": str(paths["offsets_path"]),
            "prompt_field": self.prompt_field,
            "response_field": self.response_field,
            "ignore_index": IGNORE_INDEX,
        }
        paths["meta_path"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def _build_or_load_split_caches(
        self,
        src_path: Path,
        cache_root: Path,
        rebuild_cache: bool,
        train_split: float,
        seed: int,
    ) -> tuple[dict, dict]:
        # Deterministically split rows in-memory once for single-file mode.
        rows = list(_iter_jsonl(src_path))
        if len(rows) < 2:
            raise ValueError("Need at least 2 rows for train/val split")

        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        rng.shuffle(idx)

        n_train = int(len(rows) * train_split)
        n_train = min(max(n_train, 1), len(rows) - 1)

        train_rows = [rows[i] for i in idx[:n_train]]
        val_rows = [rows[i] for i in idx[n_train:]]

        train_meta = self._build_or_load_cache(
            src_path=src_path,
            split_name="train",
            cache_root=cache_root,
            rebuild_cache=rebuild_cache,
            preloaded_rows=train_rows,
        )
        val_meta = self._build_or_load_cache(
            src_path=src_path,
            split_name="val",
            cache_root=cache_root,
            rebuild_cache=rebuild_cache,
            preloaded_rows=val_rows,
        )
        return train_meta, val_meta

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
        if block_size != self.max_seq_len:
            raise ValueError(
                f"For SFT, block_size must match max_seq_len used in cache build. "
                f"Got block_size={block_size}, max_seq_len={self.max_seq_len}"
            )

        offsets = self.train_offsets if split == "train" else self.val_offsets
        input_mem = self.train_input if split == "train" else self.val_input
        label_mem = self.train_labels if split == "train" else self.val_labels

        n_samples = len(offsets)
        sample_ids = torch.randint(0, n_samples, (batch_size,), generator=self._rng).tolist()

        x_np = np.zeros((batch_size, block_size), dtype=np.int64)
        y_np = np.full((batch_size, block_size), IGNORE_INDEX, dtype=np.int64)

        for i, sid in enumerate(sample_ids):
            start, length = offsets[sid]
            start = int(start)
            length = int(length)

            inp = np.asarray(input_mem[start : start + length], dtype=np.int64)
            lab = np.asarray(label_mem[start : start + length], dtype=np.int64)

            take = min(length, block_size)
            x_np[i, :take] = inp[:take]
            y_np[i, :take] = lab[:take]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x.to(device), y.to(device)


__all__ = ["SFTMemmapDataModule", "IGNORE_INDEX"]
