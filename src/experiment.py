"""Shared experiment logic for the Web-FLARE pilot.

The original ``scripts/run_flare.py`` reloads the model for every invocation.
The Colab runner loads Qwen once and calls :func:`run_job` for each assigned
job, which is the main wall-clock saving on a single GPU.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.datasets import ASQA, StrategyQA, WikiAsp, WikiMultiHopQA, get_dataset
from src.flare.agent import FLAREAgent, extract_answer
from src.retrieval.factory import build_retriever

_ANSWER_IS_RE = re.compile(
    r"(?:so\s+)?(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([^.\n]+)",
    re.IGNORECASE,
)
_BOOL_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)

_SCORERS = {
    "2wikihop": WikiMultiHopQA,
    "2wikimultihopqa": WikiMultiHopQA,
    "wikimultihopqa": WikiMultiHopQA,
    "strategyqa": StrategyQA,
    "strategy": StrategyQA,
    "asqa": ASQA,
    "wikiasp": WikiAsp,
    "wikilsp": WikiAsp,
}


def normalise_dataset_name(name: str) -> str:
    return (name or "").lower().replace("-", "").replace("_", "")


def get_scorer(dataset_name: str):
    key = normalise_dataset_name(dataset_name)
    if key not in _SCORERS:
        raise ValueError(f"Unknown dataset for scoring: {dataset_name!r}")
    return _SCORERS[key]


def extract_prediction(dataset_name: str, text: str) -> str:
    """Dataset-aware answer extraction.

    The same extractor is applied to every method during re-scoring, so the
    comparison is not biased by which run produced the row.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    first_paragraph = raw.split("\n\n")[0].strip()
    match = _ANSWER_IS_RE.search(first_paragraph)
    candidate = match.group(1).strip() if match else extract_answer(first_paragraph)

    key = normalise_dataset_name(dataset_name)
    if key in {"strategyqa", "strategy"}:
        for source in (candidate, first_paragraph):
            matches = _BOOL_RE.findall(source)
            if matches:
                return matches[-1].lower()
    return candidate.strip()


def score_prediction(dataset_name: str, prediction: str, gold: str) -> tuple[float, float]:
    scorer = get_scorer(dataset_name)
    em = scorer.exact_match_score(prediction, gold or "")
    f1 = scorer.f1_score(prediction, gold or "")
    return float(em.get("correct", 0)), float(f1.get("f1", 0.0))


def select_examples(
    examples: list[dict],
    ids: Optional[Iterable[str]] = None,
    n: Optional[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[dict]:
    """Select examples by explicit ID order, or the first ``n`` in dataset order."""
    if ids is not None:
        wanted = [str(item) for item in ids]
        by_id = {str(example.get("id")): example for example in examples}
        selected = [by_id[item] for item in wanted if item in by_id]
    else:
        selected = list(examples[: int(n)] if n else examples)

    if num_shards and num_shards > 1:
        selected = [
            example
            for index, example in enumerate(selected)
            if index % int(num_shards) == int(shard_id)
        ]
    return selected


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def prediction_ids(path: str, n: Optional[int] = None) -> list[str]:
    ids: list[str] = []
    for row in read_jsonl(path):
        ids.append(str(row.get("id")))
        if n is not None and len(ids) >= int(n):
            break
    return ids


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001 - metadata only
        return "unknown"


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _retriever_summary(retriever: Any) -> dict:
    if retriever is None:
        return {}
    summarise = getattr(retriever, "summary", None)
    if callable(summarise):
        try:
            return summarise()
        except Exception:  # noqa: BLE001 - diagnostics only
            return {}
    return {"retriever": getattr(retriever, "name", type(retriever).__name__)}


def run_job(
    job: dict,
    generator: Any,
    base_config: dict,
    results_root: str,
    budget_deadline: Optional[float] = None,
    split: str = "dev",
    reextract: bool = True,
    verbose: bool = True,
) -> dict:
    """Run one matrix job with resume support and a wall-clock budget."""
    job_id = str(job["id"])
    out_dir = os.path.join(results_root, "jobs", job_id)
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = job["dataset"]
    data_path = job["data_path"]
    dataset = get_dataset(dataset_name, data_dir=data_path, split=split)
    examples = list(dataset)
    if not examples:
        raise RuntimeError(
            f"No examples loaded for {dataset_name!r} from {data_path!r}. "
            "Run scripts/download_data.py first."
        )

    selected = select_examples(
        examples,
        ids=job.get("ids"),
        n=job.get("n"),
        shard_id=int(job.get("shard_id", 0)),
        num_shards=int(job.get("num_shards", 1)),
    )
    expected_ids = [str(example["id"]) for example in selected]
    if not selected:
        raise RuntimeError(f"Job {job_id} selected zero examples.")

    config = deepcopy(base_config)
    config.update(job.get("config", {}) or {})
    retriever_name = (job.get("retriever") or "none").lower()
    mode = (job.get("mode") or "flare").lower()

    retriever = build_retriever(
        retriever_name,
        config,
        options=job.get("retriever_options", {}) or {},
    )

    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)

    pred_path = os.path.join(out_dir, "predictions.jsonl")
    trace_path = os.path.join(out_dir, "traces.jsonl")
    existing_rows = read_jsonl(pred_path)
    completed_ids = {str(row.get("id")) for row in existing_rows}
    rows_by_id: dict[str, dict] = {str(row.get("id")): row for row in existing_rows}

    started_at = time.time()
    stopped_early = False
    errors = 0

    with open(pred_path, "a", encoding="utf-8") as pred_file, open(
        trace_path, "a", encoding="utf-8"
    ) as trace_file:
        for index, example in enumerate(selected, start=1):
            example_id = str(example["id"])
            if example_id in completed_ids:
                continue

            if budget_deadline is not None and time.time() >= budget_deadline:
                stopped_early = True
                if verbose:
                    print(f"[{job_id}] budget reached after {index - 1} examples; stopping.")
                break

            question = example["question"]
            gold = example["answer"]
            if verbose:
                print(f"[{job_id}] {index}/{len(selected)} id={example_id}")

            started = time.perf_counter()
            try:
                result = agent.generate(question, eval_mode=mode)
            except Exception as exc:  # noqa: BLE001 - keep a resumable job alive
                errors += 1
                print(f"[{job_id}] example {example_id} failed: {exc}")
                traceback.print_exc()
                continue
            latency_s = time.perf_counter() - started

            prediction = (
                extract_prediction(dataset_name, result.text)
                if reextract
                else extract_answer(result.text)
            )
            em, f1 = score_prediction(dataset_name, prediction, gold)

            row = {
                "id": example_id,
                "question": question,
                "gold_answer": gold,
                "full_output": result.text,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "n_retrievals": len(result.retrieval_history),
                "n_steps": len(result.confidence_history),
                "latency_s": round(latency_s, 4),
            }
            pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_file.flush()
            rows_by_id[example_id] = row
            completed_ids.add(example_id)

            trace_file.write(
                json.dumps(
                    {
                        "id": example_id,
                        "question": question,
                        "generation_history": result.generation_history,
                        "retrieval_history": result.retrieval_history,
                        "confidence_history": result.confidence_history,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            trace_file.flush()

    # Metrics are recomputed from all rows for the expected IDs, so resuming a
    # job produces exactly the same numbers as a fresh run.
    scored_rows = [rows_by_id[i] for i in expected_ids if i in rows_by_id]
    n = len(scored_rows)
    metrics = {
        "job_id": job_id,
        "dataset": dataset_name,
        "retriever": retriever_name,
        "mode": mode,
        "n_expected": len(expected_ids),
        "n_examples": n,
        "coverage": round(n / len(expected_ids), 4) if expected_ids else 0.0,
        "exact_match": round(sum(row["em"] for row in scored_rows) / n, 4) if n else 0.0,
        "f1": round(sum(row["f1"] for row in scored_rows) / n, 4) if n else 0.0,
        "avg_retrievals": round(
            sum(row.get("n_retrievals", 0) for row in scored_rows) / n, 4
        )
        if n
        else 0.0,
        "avg_steps": round(sum(row.get("n_steps", 0) for row in scored_rows) / n, 4)
        if n
        else 0.0,
        "avg_latency_s": round(
            sum(row.get("latency_s", 0.0) for row in scored_rows) / n, 4
        )
        if n
        else 0.0,
        "errors": errors,
        "stopped_early": stopped_early,
        "wall_clock_s": round(time.time() - started_at, 1),
        "retriever_stats": _retriever_summary(retriever),
    }
    _write_json(os.path.join(out_dir, "metrics.json"), metrics)
    _write_json(
        os.path.join(out_dir, "run_meta.json"),
        {
            "job": job,
            "git_commit": _git_commit(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "data_path": data_path,
            "split": split,
            "config": {
                key: config.get(key)
                for key in (
                    "topk",
                    "max_query_length",
                    "look_ahead_steps",
                    "look_ahead_filter_prob",
                    "beta_masking_threshold",
                    "max_generation_length",
                )
            },
        },
    )

    if verbose:
        print(
            f"[{job_id}] done: n={n}/{len(expected_ids)} "
            f"EM={metrics['exact_match']:.3f} F1={metrics['f1']:.3f} "
            f"retrievals={metrics['avg_retrievals']:.2f}"
        )
    return metrics


def rescore_rows(
    rows: list[dict],
    dataset_name: str,
    ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    reextract: bool = True,
) -> tuple[list[dict], dict]:
    """Re-score prediction rows with one shared extractor.

    Returns ``(rescored_rows, metrics)``.  Duplicate IDs keep the last row,
    which matches the append-and-resume log format.
    """
    wanted = [str(item) for item in ids] if ids is not None else None
    wanted_set = set(wanted) if wanted is not None else None

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        row_id = str(row.get("id"))
        if wanted_set is not None and row_id not in wanted_set:
            continue
        if row_id not in by_id:
            order.append(row_id)
        by_id[row_id] = row

    if wanted is not None:
        order = [item for item in wanted if item in by_id]
    if limit is not None:
        order = order[: int(limit)]

    rescored: list[dict] = []
    for row_id in order:
        row = by_id[row_id]
        gold = row.get("gold_answer", "")
        prediction = row.get("prediction", "")
        if reextract:
            prediction = extract_prediction(
                dataset_name, row.get("full_output") or prediction
            )
        em, f1 = score_prediction(dataset_name, prediction, gold)
        item = dict(row)
        item["prediction"] = prediction
        item["em"] = em
        item["f1"] = f1
        rescored.append(item)

    n = len(rescored)
    metrics = {
        "dataset": dataset_name,
        "n_examples": n,
        "coverage": round(n / len(wanted), 4) if wanted else 1.0,
        "exact_match": round(sum(row["em"] for row in rescored) / n, 4) if n else 0.0,
        "f1": round(sum(row["f1"] for row in rescored) / n, 4) if n else 0.0,
        "avg_retrievals": round(
            sum(row.get("n_retrievals", 0) for row in rescored) / n, 4
        )
        if n
        else 0.0,
        "avg_latency_s": round(
            sum(row.get("latency_s", 0.0) for row in rescored) / n, 4
        )
        if n
        else 0.0,
    }
    return rescored, metrics
