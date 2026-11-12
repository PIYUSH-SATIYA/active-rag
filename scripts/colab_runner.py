#!/usr/bin/env python3
"""One-command runner for the Web-FLARE pilot.

Examples
--------
Full pipeline on one Colab worker::

    python scripts/colab_runner.py all --worker-id 0 --num-workers 3 \
        --budget-minutes 75

After every worker finishes, build the tables::

    python scripts/colab_runner.py merge

The script loads Qwen once per worker, resumes per example, stops starting new
examples when the wall-clock budget is exhausted, and writes a coverage-aware
summary even for partial runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_URL = "https://github.com/PIYUSH-SATIYA/active-rag.git"
DEFAULT_BRANCH = "webflare"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/active-rag-webflare"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def is_colab() -> bool:
    return os.path.isdir("/content") and "COLAB_GPU" in os.environ or os.path.isdir(
        "/content/drive"
    )


def load_matrix(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> int:
    print("$", " ".join(str(part) for part in cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def default_results_root() -> str:
    if os.path.isdir("/content/drive/MyDrive"):
        return DEFAULT_DRIVE_ROOT
    return str(REPO_ROOT / "results" / "webflare")


def _has_required_baselines(path: str) -> bool:
    required = ("2wiki_flare", "strategyqa_flare", "strategyqa_no_retrieval")
    return all(
        os.path.exists(os.path.join(path, source, "predictions.jsonl"))
        for source in required
    )


def resolve_baseline_dir(matrix: dict, override: Optional[str]) -> Optional[str]:
    if override:
        return override if os.path.isdir(override) else None
    candidates = [
        path
        for path in matrix.get("baseline_dir_candidates", [])
        if path and os.path.isdir(path)
    ]
    for candidate in candidates:
        if _has_required_baselines(candidate):
            return candidate
    # A partial Drive copy is worse than the committed pilot copy, so only
    # fall back to an incomplete directory when nothing else exists.
    return candidates[0] if candidates else None


def resolve_workdir(args) -> Path:
    if args.workdir:
        return Path(args.workdir).expanduser().resolve()
    if os.path.isdir("/content"):
        return Path("/content/active-rag")
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def cmd_setup(args) -> None:
    workdir = resolve_workdir(args)
    repo_url = args.repo_url
    branch = args.branch

    if (workdir / ".git").is_dir():
        print(f"[setup] repo already present at {workdir}")
        if not args.no_pull:
            run(["git", "fetch", "origin", branch], cwd=workdir, check=False)
            run(["git", "checkout", branch], cwd=workdir, check=False)
            run(["git", "pull", "--ff-only", "origin", branch], cwd=workdir, check=False)
    elif workdir.exists() and any(workdir.iterdir()):
        print(f"[setup] {workdir} exists but is not a git repo; using it as-is")
    else:
        workdir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--branch", branch, "--depth", "1", repo_url, str(workdir)])

    os.chdir(workdir)
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    if not args.skip_install:
        # Install only what the pilot needs. Colab already provides torch and
        # transformers; pulling the full requirements.txt would drag in
        # Pyserini/FAISS/Spacy and waste 10+ minutes.
        needed = ["torch", "transformers", "bitsandbytes", "datasets", "ddgs", "requests"]
        missing = [name for name in needed if not have_module(name)]
        if missing:
            run([sys.executable, "-m", "pip", "install", "-q", *missing])

    if args.download_data:
        run(
            [
                sys.executable,
                "scripts/download_data.py",
                "--datasets",
                "2wikimultihopqa",
                "strategyqa",
            ]
        )
    else:
        pilot = workdir / "data" / "pilot"
        if not (pilot / "2wikimultihopqa" / "dev.json").exists():
            raise SystemExit(
                "[setup] data/pilot is missing. The branch should ship a "
                "self-contained 100-example split; re-clone the branch or "
                "pass --download-data to fetch the full datasets."
            )

    print("[setup] ready")


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------
def expected_ids(job: dict, baseline_dir: Optional[str], limit: Optional[int]) -> list[str]:
    """Resolve the exact examples a job should use.

    Preference order:
    1. the baseline predictions file named by ``ids_source``;
    2. the first ``n`` examples in dataset order.
    """
    from src.experiment import prediction_ids
    from src.datasets import get_dataset

    n = int(limit if limit is not None else job.get("n", 100))
    source = job.get("ids_source")
    if source and baseline_dir:
        path = os.path.join(baseline_dir, source, "predictions.jsonl")
        if os.path.exists(path):
            ids = prediction_ids(path, n)
            if ids:
                return ids

    dataset = get_dataset(job["dataset"], data_dir=job["data_path"], split="dev")
    ids = [str(example["id"]) for example in list(dataset)[:n]]
    if not ids:
        raise RuntimeError(
            f"Cannot resolve IDs for {job['id']}: no baseline predictions and "
            f"no dataset at {job['data_path']}."
        )
    return ids


def job_predictions_path(job: dict, results_root: str, baseline_dir: Optional[str]) -> Optional[str]:
    if job.get("kind", "run") == "baseline":
        if not baseline_dir:
            return None
        return os.path.join(baseline_dir, job["source"], "predictions.jsonl")
    return os.path.join(results_root, "jobs", job["id"], "predictions.jsonl")


def job_is_complete(job: dict, results_root: str, baseline_dir: Optional[str], limit: Optional[int]) -> bool:
    from src.experiment import read_jsonl

    path = job_predictions_path(job, results_root, baseline_dir)
    if not path or not os.path.exists(path):
        return False
    ids = set(expected_ids(job, baseline_dir, limit))
    done = {str(row.get("id")) for row in read_jsonl(path)}
    return ids.issubset(done)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def select_jobs(matrix: dict, args) -> list[dict]:
    jobs = [job for job in matrix["jobs"] if job.get("kind", "run") == "run"]
    if args.jobs:
        wanted = set(args.jobs)
        jobs = [job for job in jobs if job["id"] in wanted]
    if not args.include_optional and not args.jobs:
        jobs = [job for job in jobs if not job.get("optional", False)]
    return jobs


def ensure_java() -> None:
    """Install Pyserini + Java 21 only when a hybrid job is actually assigned."""
    if not have_module("pyserini"):
        run([sys.executable, "-m", "pip", "install", "-q", "pyserini", "faiss-cpu"])
    if not is_colab() or not shutil.which("apt-get"):
        return
    if not os.path.exists("/usr/lib/jvm/java-21-openjdk-amd64"):
        run(["apt-get", "update", "-qq"], check=False)
        run(
            ["apt-get", "install", "-y", "-qq", "openjdk-21-jdk-headless"],
            check=False,
        )
    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    os.environ["PATH"] = (
        os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")
    )


def job_limit(job: dict, args) -> int:
    base = int(job.get("n", 100))
    if args.n:
        return min(base, int(args.n))
    if args.quick:
        return min(base, 50)
    return base


def cmd_run(args, deadline: Optional[float] = None) -> None:
    workdir = resolve_workdir(args)
    os.chdir(workdir)
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    from src.experiment import run_job

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = workdir / matrix_path
    matrix = load_matrix(matrix_path)

    results_root = args.results_root or default_results_root()
    os.makedirs(results_root, exist_ok=True)

    baseline_dir = resolve_baseline_dir(matrix, args.baseline_dir)
    if not baseline_dir:
        raise SystemExit(
            "[run] No baseline predictions found. Expected the Drive copy at "
            "/content/drive/MyDrive/active-rag-results or the committed pilot "
            "copy at data/pilot/baselines. Pass --baseline-dir to override."
        )
    print(f"[run] baseline dir: {baseline_dir}")

    jobs = select_jobs(matrix, args)
    # Longest job first balances the round-robin split across workers.
    jobs.sort(key=lambda job: -job_limit(job, args))
    assigned = [
        job for index, job in enumerate(jobs) if index % args.num_workers == args.worker_id
    ]
    if not assigned:
        print(f"[run] worker {args.worker_id} has no jobs")
        return
    print(f"[run] worker {args.worker_id}/{args.num_workers}: {[j['id'] for j in assigned]}")

    if any((job.get("retriever") or "").lower() == "hybrid" for job in assigned):
        ensure_java()

    if deadline is None and args.budget_minutes:
        deadline = time.time() + args.budget_minutes * 60

    # Resolve IDs and skip completed jobs before loading the model.
    prepared: list[dict] = []
    for job in assigned:
        limit = job_limit(job, args)
        job = dict(job)
        job["n"] = limit
        if job_is_complete(job, results_root, baseline_dir, limit):
            print(f"[run] {job['id']} already complete; skipping")
            continue
        job["ids"] = expected_ids(job, baseline_dir, limit)
        prepared.append(job)

    if not prepared:
        print("[run] all assigned jobs are complete")
        return

    if args.dry_run:
        for job in prepared:
            print(
                f"[dry-run] {job['id']}: n={len(job['ids'])} "
                f"first={job['ids'][:2]} retriever={job.get('retriever')}"
            )
        return

    from src.models.qwen import QwenModel

    base_config_path = Path(matrix.get("base_config", "configs/2wikihop_qwen_flare_config.json"))
    if not base_config_path.is_absolute():
        base_config_path = workdir / base_config_path
    with open(base_config_path, "r", encoding="utf-8") as handle:
        base_config = json.load(handle)
    base_config["model_name"] = matrix.get("model_name", base_config.get("model_name"))
    base_config["quantize_4bit"] = bool(
        matrix.get("quantize_4bit", base_config.get("quantize_4bit", True))
    )
    web_config = dict(matrix.get("web", {}))
    web_config["cache_dir"] = os.path.join(results_root, "web_cache")
    base_config["web"] = web_config

    print(f"[run] loading {base_config['model_name']} (4-bit={base_config['quantize_4bit']})")
    try:
        generator = QwenModel.from_pretrained(
            model_name=base_config["model_name"],
            quantize_4bit=base_config["quantize_4bit"],
        )
    except Exception as exc:  # noqa: BLE001 - bitsandbytes is optional
        if not base_config["quantize_4bit"]:
            raise
        print(f"[run] 4-bit load failed ({exc}); retrying in fp16")
        base_config["quantize_4bit"] = False
        generator = QwenModel.from_pretrained(
            model_name=base_config["model_name"],
            quantize_4bit=False,
        )

    for job in prepared:
        if deadline is not None and time.time() >= deadline:
            print("[run] global budget exhausted; remaining jobs will be partial")
            break
        run_job(
            job=job,
            generator=generator,
            base_config=base_config,
            results_root=results_root,
            budget_deadline=deadline,
            split="dev",
            reextract=True,
        )


# ---------------------------------------------------------------------------
# Merge + summary
# ---------------------------------------------------------------------------
def _method_label(job: dict) -> str:
    return job.get("label") or job["id"].split("_", 1)[-1].replace("_", "-")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def cmd_merge(args) -> None:
    workdir = resolve_workdir(args)
    os.chdir(workdir)
    if str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    from src.experiment import read_jsonl, rescore_rows

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = workdir / matrix_path
    matrix = load_matrix(matrix_path)

    results_root = args.results_root or default_results_root()
    baseline_dir = resolve_baseline_dir(matrix, args.baseline_dir)
    if not baseline_dir:
        print("[merge] WARNING: no baseline dir; baseline rows will be skipped")

    summary: list[dict] = []
    for job in matrix["jobs"]:
        limit = job_limit(job, args)
        source_path = job_predictions_path(job, results_root, baseline_dir)
        if not source_path or not os.path.exists(source_path):
            print(f"[merge] {job['id']}: no predictions at {source_path}; skipping")
            continue

        ids = expected_ids(job, baseline_dir, limit)
        rows = read_jsonl(source_path)
        rescored, metrics = rescore_rows(
            rows,
            dataset_name=job["dataset"],
            ids=ids,
            limit=limit,
            reextract=True,
        )
        if not rescored:
            print(f"[merge] {job['id']}: zero matching rows; skipping")
            continue

        out_dir = Path(results_root) / "jobs" / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
            for row in rescored:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        existing_metrics_path = out_dir / "metrics.json"
        existing_metrics = {}
        if existing_metrics_path.exists():
            try:
                existing_metrics = json.loads(existing_metrics_path.read_text())
            except ValueError:
                existing_metrics = {}
        metrics = {
            **existing_metrics,
            **metrics,
            "job_id": job["id"],
            "retriever": job.get("retriever"),
            "mode": job.get("mode"),
        }
        _write_json(out_dir / "metrics.json", metrics)

        summary.append(
            {
                "dataset": job["dataset"],
                "method": _method_label(job),
                "job_id": job["id"],
                "retriever": job.get("retriever"),
                "mode": job.get("mode"),
                "n": metrics["n_examples"],
                "n_expected": len(ids),
                "coverage": metrics["coverage"],
                "exact_match": metrics["exact_match"],
                "f1": metrics["f1"],
                "avg_retrievals": metrics.get("avg_retrievals", 0.0),
                "avg_latency_s": metrics.get("avg_latency_s", 0.0),
            }
        )

    summary.sort(key=lambda row: (row["dataset"], row["method"]))
    results_path = Path(results_root)

    _write_json(results_path / "summary.json", summary)
    with open(results_path / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    lines = [
        "# Web-FLARE pilot results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Results root: `{results_root}`",
        f"Baseline dir: `{baseline_dir}`",
        "",
        "| Dataset | Method | N | Coverage | EM | F1 | Avg retrievals | Avg latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {method} | {n} | {coverage:.0%} | {exact_match:.3f} | "
            "{f1:.3f} | {avg_retrievals:.2f} | {avg_latency_s:.2f} |".format(**row)
        )
    (results_path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[merge] wrote {results_path / 'summary.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--matrix", default="configs/experiment_matrix.json")
    common.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    common.add_argument("--branch", default=DEFAULT_BRANCH)
    common.add_argument("--workdir", default=None)
    common.add_argument("--results-root", default=None)
    common.add_argument("--baseline-dir", default=None)
    common.add_argument("--worker-id", type=int, default=0)
    common.add_argument("--num-workers", type=int, default=1)
    common.add_argument("--budget-minutes", type=float, default=75.0)
    common.add_argument(
        "--quick", action="store_true", help="cap every job at 50 examples"
    )
    common.add_argument("--n", type=int, default=None, help="cap every job at N examples")
    common.add_argument("--jobs", nargs="+", default=None, help="only run these job IDs")
    common.add_argument(
        "--include-optional",
        action="store_true",
        help="also run the hybrid and non-Wikipedia ablations (hybrid needs Pyserini)",
    )
    common.add_argument("--skip-install", action="store_true")
    common.add_argument(
        "--download-data",
        action="store_true",
        help="fetch full datasets instead of the committed pilot split",
    )
    common.add_argument("--no-pull", action="store_true")
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve jobs and IDs, then exit before loading the model",
    )

    parser = argparse.ArgumentParser(description="Web-FLARE Colab runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "run", "merge", "all"):
        sub.add_parser(name, parents=[common])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        parser.error("--worker-id must be in [0, num-workers)")

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "all":
        cmd_setup(args)
        # Start the clock after setup: pip installs and model downloads are
        # fixed costs, not part of the experiment budget.
        deadline = (
            time.time() + args.budget_minutes * 60 if args.budget_minutes else None
        )
        cmd_run(args, deadline=deadline)
        cmd_merge(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
