# Web-FLARE: Browser-Assisted Active Retrieval

This branch adds a pluggable retrieval layer to the existing FLARE
reimplementation and a one-command Colab runner for a small comparison
between BM25, live web search, and a hybrid of the two.

The FLARE algorithm itself is unchanged. Only the retrieval source is swapped,
so the comparison isolates one variable.

## What was added

| Path | Purpose |
|---|---|
| `src/retrieval/base.py` | `RetrievalResult` with source/URL metadata and the retriever protocol |
| `src/retrieval/web.py` | DuckDuckGo Lite + Yahoo + Bing HTML, `ddgs` fallback, Tavily (optional), disk cache, domain exclusion, cooldowns |
| `src/retrieval/hybrid.py` | BM25 + web reciprocal-rank fusion with duplicate removal |
| `src/retrieval/factory.py` | `build_retriever(name, config, options)` |
| `src/experiment.py` | Shared job loop, resume, budget guard, shared answer extraction, re-scoring |
| `scripts/colab_runner.py` | `setup` / `run` / `merge` / `all`, plus `--dry-run` |
| `scripts/make_pilot_data.py` | Builds the committed 100-example pilot split from saved predictions |
| `scripts/rescore.py` | Re-score any `predictions.jsonl` with one extractor |
| `scripts/download_data.py` | Optional full 2WikiMultiHopQA / StrategyQA downloader |
| `configs/experiment_matrix.json` | The pilot matrix |

`src/flare/agent.py` now also records `retrieved_meta` (source, URL, title,
score) alongside the existing `retrieved_docs` strings. Existing tests that
read `retrieved_docs` are unaffected.

## Matrix

| Job | Dataset | N | Retriever | Mode | Default |
|---|---|---:|---|---|---|
| `2wiki_bm25_flare` | 2Wiki | 100 | BM25 | flare | re-scored baseline |
| `2wiki_web_flare` | 2Wiki | 100 | Web | flare | yes |
| `2wiki_web_single` | 2Wiki | 50 | Web | single_retrieval | yes |
| `strategyqa_bm25_flare` | StrategyQA | 100 | BM25 | flare | re-scored baseline |
| `strategyqa_web_flare` | StrategyQA | 100 | Web | flare | yes |
| `strategyqa_bm25_no_ret` | StrategyQA | 100 | None | no_retrieval | re-scored baseline |
| `2wiki_hybrid_flare` | 2Wiki | 100 | Hybrid | flare | `--include-optional` |
| `2wiki_web_nwiki` | 2Wiki | 50 | Web minus Wikipedia | flare | yes |

The BM25 rows are **not regenerated**. They are re-scored from the existing
`results/2wiki_flare`, `results/strategyqa_flare`, and
`results/strategyqa_no_retrieval` predictions. A trimmed copy of the first 100
rows and the matching questions lives in `data/pilot/`, so a fresh clone can
run and merge without Google Drive.

On Colab the runner prefers `/content/drive/MyDrive/active-rag-results` when
that directory contains all three baseline files, and otherwise uses the
committed `data/pilot/baselines`.

## Running

One worker:

```bash
python scripts/colab_runner.py all \
    --worker-id 0 --num-workers 3 --budget-minutes 75
```

Use the same `--num-workers` and different `--worker-id` values (0, 1, 2) in
the other sessions. A single GPU works with `--num-workers 1`.

Jobs are assigned longest-first to the least-loaded worker (LPT), so the
default matrix splits 100/100/100 examples across three workers.

Useful flags:

- `--quick` caps every job at 50 examples.
- `--n 40` caps every job at 40 examples.
- `--include-optional` adds the hybrid job. It needs Pyserini, Java 21,
  and a multi-GB prebuilt index, so it is off by default.
- `--budget-minutes` is checked before each new example; partial results are kept.
- `--dry-run` prints the assignment and example IDs without loading the model.
- Before loading Qwen, the runner probes two live searches and prints provider
  statistics. A 0/2 result means the Colab IP is blocked; reconnect for a new
  IP or set `TAVILY_API_KEY`. `--skip-probe` disables this check.
- `--skip-install` skips pip; `--download-data` fetches full datasets.
- `--baseline-dir` overrides baseline discovery.

Merge after the workers finish:

```bash
python scripts/colab_runner.py merge
```

If the three workers used **separate Colab/Drive accounts**, collect their
result folders and point `--merge-roots` at them:

```bash
# on each worker: produce a zip to share
python scripts/colab_runner.py pack --pack-out worker0.zip

# after downloading the other zips and unzipping them into ./shared/
python scripts/colab_runner.py merge --merge-roots shared/worker1 shared/worker2
```

The primary `--results-root` wins when the same example ID exists in more than
one copy. `pack` includes `jobs/`, `web_cache/`, and any `summary.*` files.

Outputs:

```
<results-root>/summary.md
<results-root>/summary.csv
<results-root>/summary.json
<results-root>/jobs/<job-id>/predictions.jsonl
<results-root>/jobs/<job-id>/metrics.json
<results-root>/web_cache/*.json
```

Default results root is `/content/drive/MyDrive/active-rag-webflare` on Colab
and `results/webflare` elsewhere.

## Search providers and latency

No API key is required. Each query tries, in a per-instance rotation:

1. DuckDuckGo Lite HTML (`lite.duckduckgo.com`) — usually 0.4–2.5 s.
2. Yahoo Search HTML — usually 0.5–3 s.
3. Bing HTML with `mkt=en-US` — usually 0.5–4 s.
4. `ddgs` package — slowest; last resort.
5. Tavily, only when `TAVILY_API_KEY` is set.

A provider that answers with a bot check (HTTP 202/403/429) is put on a
180-second cooldown instead of being retried on every query. Results are also
checked for minimal token overlap with the query, which rejects the unrelated
SERPs that bot-detection pages sometimes return.

Local measurements on 10 2Wiki-style queries (warm network, cold cache):
mean 1.8 s/query, 10/10 returned results, Yahoo 8 and Bing 2. On a fresh
Colab IP DuckDuckGo Lite usually participates as well.

## Caching and reproducibility

Every successful web query is cached as a timestamped JSON file under
`web_cache/`. Workers sharing a Drive results root also share the cache.
Re-running a job reuses cached queries, so partial runs are cheap and the
retrieved snapshot is stable.

The cache key includes the backend, query, top-k, excluded domains, and
page-fetch flag. Failed searches are not cached.

## Answer extraction and re-scoring

The original saved `metrics.json` files were produced by different versions of
the evaluation code. `src/experiment.py` applies one extractor to every method:

- 2Wiki: the final `So the answer is ...` sentence.
- StrategyQA: the final yes/no/true/false after the answer sentence, mapped to
  the gold `true`/`false` label.

`scripts/rescore.py` exposes the same logic for a single file:

```bash
python scripts/rescore.py \
    --predictions results/strategyqa_flare/predictions.jsonl \
    --dataset strategyqa --limit 100
```

## Tests

```bash
pytest -q tests/test_web_retriever.py tests/test_hybrid.py \
         tests/test_experiment.py tests/test_run_job.py
```

The web and hybrid tests use mocked providers; no network or GPU is required.
`tests/test_flare.py` has two pre-existing failures on `master` unrelated to
this branch (the mock look-ahead tokens do not cover the mock sentence, so the
implicit-masking query is shorter than the test expects).
