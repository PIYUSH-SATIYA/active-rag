#!/usr/bin/env python3
"""Build a self-contained pilot split from existing prediction files.

Why this exists
---------------
The Colab runner must use exactly the same example IDs as the saved BM25
baselines, and it must not depend on HuggingFace dataset mirrors staying
online.  This script copies the first ``--n`` rows of each baseline
``predictions.jsonl`` into:

    data/pilot/<dataset>/dev.json          (id/question/answer)
    data/pilot/baselines/<source>/predictions.jsonl

The generated files are small and committed, so `git clone` is enough to run
the Web-FLARE comparison.

Usage (from the repository root, where the original results live)::

    python scripts/make_pilot_data.py --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_DIR = os.path.join(REPO_ROOT, "data", "pilot")

# source directory -> (dataset output directory, id field, answer field)
SOURCES = {
    "2wiki_flare": ("2wikimultihopqa", "_id", "answer"),
    "strategyqa_flare": ("strategyqa", "qid", "answer"),
    "strategyqa_no_retrieval": ("strategyqa", "qid", "answer"),
}


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=0)
    print(f"  wrote {len(payload)} rows -> {path}")


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the committed pilot split.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--out-dir", default=PILOT_DIR)
    args = parser.parse_args()

    datasets: dict[str, list[dict]] = {}
    dataset_keys: dict[str, list[tuple[str, str]]] = {
        "2wikimultihopqa": [],
        "strategyqa": [],
    }

    for source, (dataset_dir, id_field, answer_field) in SOURCES.items():
        predictions = os.path.join(args.results_dir, source, "predictions.jsonl")
        if not os.path.exists(predictions):
            print(f"[skip] {source}: {predictions} not found")
            continue
        rows = read_jsonl(predictions)[: args.n]
        write_jsonl(
            os.path.join(args.out_dir, "baselines", source, "predictions.jsonl"),
            rows,
        )
        if dataset_dir not in datasets:
            datasets[dataset_dir] = []
            dataset_keys[dataset_dir] = []
        for row in rows:
            dataset_keys[dataset_dir].append((str(row["id"]), id_field))
            datasets[dataset_dir].append(
                {
                    id_field: str(row["id"]),
                    "question": row["question"],
                    answer_field: row["gold_answer"],
                }
            )

    # The no-retrieval file must follow the same order as strategyqa_flare.
    flare_rows = [
        str(row["id"])
        for row in read_jsonl(
            os.path.join(args.out_dir, "baselines", "strategyqa_flare", "predictions.jsonl")
        )
    ]
    no_ret_path = os.path.join(
        args.out_dir, "baselines", "strategyqa_no_retrieval", "predictions.jsonl"
    )
    if os.path.exists(no_ret_path):
        no_ret = {str(row["id"]): row for row in read_jsonl(no_ret_path)}
        ordered = [no_ret[item] for item in flare_rows if item in no_ret]
        write_jsonl(no_ret_path, ordered)
        # Rebuild the StrategyQA split from the flare order, not no-retrieval.
        datasets["strategyqa"] = [
            {
                "qid": str(row["id"]),
                "question": row["question"],
                "answer": row["gold_answer"],
            }
            for row in read_jsonl(
                os.path.join(
                    args.out_dir, "baselines", "strategyqa_flare", "predictions.jsonl"
                )
            )
        ]

    for dataset_dir, rows in datasets.items():
        if rows:
            write_json(os.path.join(args.out_dir, dataset_dir, "dev.json"), rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
