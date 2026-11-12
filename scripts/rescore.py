#!/usr/bin/env python3
"""Re-score a predictions.jsonl file with one dataset-aware extractor.

This exists because the saved ``metrics.json`` files from the original runs
were produced by different versions of ``src/datasets.py``.  Re-scoring all
methods with the same extractor is required for a fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiment import read_jsonl, rescore_rows  # noqa: E402


def load_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score FLARE predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["2wikihop", "2wikimultihopqa", "strategyqa", "asqa", "wikiasp"],
    )
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument(
        "--no-reextract",
        action="store_true",
        help="Score the stored prediction field instead of re-extracting.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    ids = load_ids(args.ids_file)
    rescored, metrics = rescore_rows(
        rows,
        dataset_name=args.dataset,
        ids=ids,
        limit=args.limit,
        reextract=not args.no_reextract,
    )

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions)), "metrics_rescored.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
        with open(args.predictions_out, "w", encoding="utf-8") as handle:
            for row in rescored:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rescored)} rows -> {args.predictions_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
