#!/usr/bin/env python3
"""Download the datasets used by the Web-FLARE pilot.

Only 2WikiMultiHopQA and StrategyQA are needed for the matrix.  The script is
idempotent: existing ``dev.json`` files are left untouched unless ``--force``
is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def save_json(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"  saved {len(records)} examples -> {path}")


def download_2wiki(data_dir: str) -> None:
    import datasets as hf_datasets

    path = os.path.join(data_dir, "2wikimultihopqa", "dev.json")
    print("[download] 2WikiMultiHopQA ...")
    ds = hf_datasets.load_dataset("xanhho/2WikiMultihopQA", split="validation")
    records = [
        {
            "_id": row.get("id", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "type": row.get("type", ""),
            "context": row.get("context", []),
            "supporting_facts": row.get("supporting_facts", []),
        }
        for row in ds
    ]
    save_json(records, path)


def download_strategyqa(data_dir: str) -> None:
    import datasets as hf_datasets

    path = os.path.join(data_dir, "strategyqa", "dev.json")
    print("[download] StrategyQA ...")
    try:
        ds = hf_datasets.load_dataset("ChilleD/StrategyQA", split="test")
    except Exception as first_error:  # noqa: BLE001 - dataset mirrors move
        print(f"  ChilleD/StrategyQA failed ({first_error}); trying parquet mirror")
        ds = hf_datasets.load_dataset(
            "parquet",
            data_files="hf://datasets/wics/strategy-qa/data/test-*.parquet",
            split="train",
        )

    records = [
        {
            "qid": row.get("qid", row.get("id", "")),
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")).lower(),
        }
        for row in ds
    ]
    save_json(records, path)


DOWNLOADERS = {
    "2wikimultihopqa": download_2wiki,
    "2wiki": download_2wiki,
    "strategyqa": download_strategyqa,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pilot datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2wikimultihopqa", "strategyqa"],
        choices=sorted(DOWNLOADERS),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.datasets:
        expected = {
            "2wikimultihopqa": os.path.join(args.data_dir, "2wikimultihopqa", "dev.json"),
            "2wiki": os.path.join(args.data_dir, "2wikimultihopqa", "dev.json"),
            "strategyqa": os.path.join(args.data_dir, "strategyqa", "dev.json"),
        }[name]
        if os.path.exists(expected) and not args.force:
            print(f"[skip] {name}: {expected} already exists")
            continue
        DOWNLOADERS[name](args.data_dir)

    print("Datasets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
