"""Retriever factory used by the experiment runner."""

from __future__ import annotations

from typing import Any, Optional

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.web import WebRetriever, _WIKIPEDIA_DOMAINS

_WEB_KEYS = {
    "backend",
    "topk",
    "cache_dir",
    "exclude_domains",
    "fetch_pages",
    "max_page_chars",
    "max_chars",
    "include_url",
    "tavily_key",
    "min_interval",
    "timeout",
    "max_retries",
}


def build_retriever(
    name: str,
    config: dict,
    options: Optional[dict] = None,
) -> Optional[Any]:
    """Build ``bm25``, ``web``, ``hybrid`` or ``None``.

    ``options`` are per-job overrides from the experiment matrix.  Nested
    ``bm25``/``web`` dictionaries are accepted for the hybrid backend.
    """
    name = (name or "none").lower()
    options = dict(options or {})

    if name in {"none", "no_retrieval", ""}:
        return None

    if name == "bm25":
        from src.retriever import Retriever

        es_cfg = config.get("elasticsearch", {}) or {}
        index = options.get("index") or es_cfg.get("index", "wikipedia-dpr")
        return Retriever(index_name=index)

    if name == "web":
        return _build_web(config, options)

    if name == "hybrid":
        bm25_options = options.get("bm25", {}) or {}
        web_options = options.get("web", {}) or {}
        bm25 = build_retriever("bm25", config, bm25_options)
        web = build_retriever("web", config, web_options)
        return HybridRetriever(
            bm25=bm25,
            web=web,
            rrf_k=int(options.get("rrf_k", 60)),
        )

    raise ValueError(f"Unknown retriever: {name!r}")


def _build_web(config: dict, options: dict) -> WebRetriever:
    web_cfg = dict(config.get("web", {}) or {})
    web_cfg.update({k: v for k, v in options.items() if k in _WEB_KEYS})
    web_cfg.setdefault("topk", config.get("topk", 2))
    web_cfg.setdefault("cache_dir", ".cache/web")
    if options.get("exclude_wikipedia"):
        existing = tuple(web_cfg.get("exclude_domains", ()) or ())
        web_cfg["exclude_domains"] = tuple(dict.fromkeys(existing + _WIKIPEDIA_DOMAINS))
    allowed = {k: v for k, v in web_cfg.items() if k in _WEB_KEYS}
    return WebRetriever(**allowed)
