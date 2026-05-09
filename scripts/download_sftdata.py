import argparse
import json
import random
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, load_dataset_builder


def _iter_rows(ds, prompt_field: str, response_field: str) -> Iterator[dict]:
    for row in ds:
        prompt = row.get(prompt_field)
        response = row.get(response_field)
        if prompt is None or response is None:
            continue
        prompt = str(prompt).strip()
        response = str(response).strip()
        if not prompt or not response:
            continue
        yield {"prompt": prompt, "response": response}


def _available_splits(dataset: str, name: str | None) -> set[str]:
    builder = load_dataset_builder(dataset, name)
    return set(builder.info.splits.keys())


def _resolve_splits(requested: str, available: set[str]) -> list[str]:
    req = requested.lower()
    if req == "both":
        out = []
        if "train" in available:
            out.append("train")
        if "validation" in available:
            out.append("validation")
        elif "val" in available:
            out.append("val")
        elif "test" in available:
            out.append("test")
        if not out:
            raise ValueError(f"No suitable splits for 'both'. Available: {sorted(available)}")
        return out

    if req == "val":
        if "validation" in available:
            return ["validation"]
        if "val" in available:
            return ["val"]
        raise ValueError(f"Requested val/validation split not found. Available: {sorted(available)}")

    if requested in available:
        return [requested]

    raise ValueError(f"Split '{requested}' not found. Available: {sorted(available)}")


def _output_for_split(base_out: Path, split_name: str, multi: bool) -> Path:
    if not multi:
        return base_out
    suffix = "_val" if split_name in {"validation", "val"} else f"_{split_name}"
    return base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")


def _write_jsonl(examples: Iterator[dict], out_path: Path, max_samples: int | None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SFT dataset and materialize prompt/response JSONL")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. Dahoas/sft-static")
    parser.add_argument("--name", type=str, default=None, help="Optional config name/subset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="train | validation | val | test | both | any dataset-defined split",
    )
    parser.add_argument("--prompt-field", type=str, default="prompt")
    parser.add_argument("--response-field", type=str, default="response")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-val-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    available = _available_splits(args.dataset, args.name)
    base_out = Path(args.out)

    if args.split.lower() == "both" and available == {"train"}:
        if not (0.0 < args.local_val_ratio < 1.0):
            raise ValueError("local-val-ratio must be between 0 and 1")

        ds = load_dataset(args.dataset, args.name, split="train")
        all_examples = list(_iter_rows(ds, args.prompt_field, args.response_field))
        if len(all_examples) < 2:
            raise ValueError("Not enough examples to create local train/val split")

        rng = random.Random(args.split_seed)
        rng.shuffle(all_examples)
        n_val = max(1, int(len(all_examples) * args.local_val_ratio))
        val_examples = all_examples[:n_val]
        train_examples = all_examples[n_val:]

        split_to_examples = {"train": train_examples, "val": val_examples}
        for split_name, examples in split_to_examples.items():
            out_path = _output_for_split(base_out, split_name, multi=True)
            count = _write_jsonl(examples, out_path, args.max_samples)
            meta = {
                "dataset": args.dataset,
                "name": args.name,
                "requested_split": args.split,
                "actual_split": split_name,
                "prompt_field": args.prompt_field,
                "response_field": args.response_field,
                "max_samples": args.max_samples,
                "written_samples": count,
                "output": str(out_path),
                "available_splits": sorted(available),
                "local_split": True,
                "local_val_ratio": args.local_val_ratio,
                "split_seed": args.split_seed,
                "source_total_examples": len(all_examples),
            }
            meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Wrote {count} samples from local '{split_name}' split to {out_path}")
            print(f"Metadata: {meta_path}")
        return

    splits = _resolve_splits(args.split, available)
    for split_name in splits:
        ds = load_dataset(args.dataset, args.name, split=split_name)
        out_path = _output_for_split(base_out, split_name, multi=len(splits) > 1)
        count = _write_jsonl(_iter_rows(ds, args.prompt_field, args.response_field), out_path, args.max_samples)

        meta = {
            "dataset": args.dataset,
            "name": args.name,
            "requested_split": args.split,
            "actual_split": split_name,
            "prompt_field": args.prompt_field,
            "response_field": args.response_field,
            "max_samples": args.max_samples,
            "written_samples": count,
            "output": str(out_path),
            "available_splits": sorted(available),
        }
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Wrote {count} samples from '{split_name}' to {out_path}")
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
