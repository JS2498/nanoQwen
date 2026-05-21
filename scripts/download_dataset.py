import argparse
from html import parser
import json
from pathlib import Path

from datasets import load_dataset, load_dataset_builder


def write_text_lines(dataset_split, text_field: str, out_path: Path, max_samples: int | None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset_split:
            text = row.get(text_field)
            if not text:
                continue
            f.write(text.replace("\n", " ").strip() + "\n")
            n += 1
            if max_samples is not None and n >= max_samples:
                break
    return n


def infer_splits(dataset: str, name: str | None) -> set[str]:
    builder = load_dataset_builder(dataset, name)
    return set(builder.info.splits.keys())


def resolve_requested_splits(requested: str, available: set[str]) -> list[str]:
    req = requested.lower()
    if req in {"validation", "val"}:
        if "validation" in available:
            return ["validation"]
        if "val" in available:
            return ["val"]
        raise ValueError(f"Requested validation split, but available splits are: {sorted(available)}")

    if req == "train":
        if "train" not in available:
            raise ValueError(f"Requested train split, but available splits are: {sorted(available)}")
        return ["train"]

    if req == "both":
        if "train" not in available:
            raise ValueError(f"Requested both, but train split is missing. Available: {sorted(available)}")
        if "validation" in available:
            return ["train", "validation"]
        if "val" in available:
            return ["train", "val"]
        raise ValueError(
            "Requested both, but no validation-like split found. "
            f"Available splits are: {sorted(available)}"
        )

    if requested in available:
        return [requested]
    raise ValueError(f"Unsupported split '{requested}'. Available splits are: {sorted(available)}")


def output_for_split(base_out: Path, split_name: str, multi: bool) -> Path:
    if not multi:
        return base_out
    suffix = "_val" if split_name in {"validation", "val"} else f"_{split_name}"
    return base_out.with_name(f"{base_out.stem}{suffix}{base_out.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize HF datasets for nanoQwen")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. roneneldan/TinyStories")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="train | validation | val | both | or any dataset-defined split name",
    )
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--out", type=str, required=True, help="Output text file path")
    parser.add_argument("--max-samples", type=int, default=None)  # 20000 worked fine for my GPU during training
    parser.add_argument("--name", type=str, default=None, help="Optional config name/subset")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset without downloading full shards to local cache")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    available_splits = infer_splits(args.dataset, args.name)
    splits_to_export = resolve_requested_splits(args.split, available_splits)
    base_out = Path(args.out)


    if not args.streaming:
        print(
            "Warning: streaming is disabled. Large datasets (e.g., FineWeb sample-10BT) "
            "may download many parquet shards before writing output."
        )

    summary: list[dict] = []
    for split_name in splits_to_export:
        # ds = load_dataset(args.dataset, args.name, split=split_name)
        ds = load_dataset(args.dataset, args.name, split=split_name, streaming=args.streaming,)
        
        # for shuffling
        if args.streaming:
            ds = ds.shuffle(seed=args.seed, buffer_size=10000)
        out_path = output_for_split(base_out, split_name, multi=len(splits_to_export) > 1)
        count = write_text_lines(ds, args.text_field, out_path, args.max_samples)

        meta = {
            "dataset": args.dataset,
            "name": args.name,
            "requested_split": args.split,
            "actual_split": split_name,
            "text_field": args.text_field,
            "max_samples": args.max_samples,
            "written_samples": count,
            "output": str(out_path),
            "available_splits": sorted(available_splits),
        }
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Wrote {count} samples from '{split_name}' to {out_path}")
        print(f"Metadata: {meta_path}")
        summary.append({"split": split_name, "samples": count, "output": str(out_path)})

    if len(summary) > 1:
        total = sum(x["samples"] for x in summary)
        print(f"Done. Exported {len(summary)} splits, total samples written: {total}")


if __name__ == "__main__":
    main()
