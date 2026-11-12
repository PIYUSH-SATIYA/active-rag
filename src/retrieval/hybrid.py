"""Hybrid BM25 + web retrieval using reciprocal rank fusion (RRF)."""

from __future__ import annotations

import hashlib
from typing import Optional

from src.retrieval.base import RetrievalResult


class HybridRetriever:
    """Fuse a local BM25 index and live web results.

    RRF is used instead of score normalisation because BM25 and web-search
    scores live on different, non-comparable scales.
    """

    name = "hybrid"

    def __init__(self, bm25, web, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.web = web
        self.rrf_k = max(1, int(rrf_k))

    def retrieve(
        self,
        queries: list[str],
        topk: int = 2,
        max_query_length: Optional[int] = None,
    ) -> list[list[RetrievalResult]]:
        bm25_batches = (
            self.bm25.retrieve(queries, topk=topk, max_query_length=max_query_length)
            if self.bm25 is not None
            else [[] for _ in queries]
        )
        web_batches = (
            self.web.retrieve(queries, topk=topk, max_query_length=max_query_length)
            if self.web is not None
            else [[] for _ in queries]
        )
        return [
            self._fuse(bm25_rows, web_rows, topk)
            for bm25_rows, web_rows in zip(bm25_batches, web_batches)
        ]

    def _fuse(
        self,
        bm25_rows: list[RetrievalResult],
        web_rows: list[RetrievalResult],
        topk: int,
    ) -> list[RetrievalResult]:
        fused: dict[str, dict] = {}

        def add(rows: list[RetrievalResult]) -> None:
            for rank, row in enumerate(rows, start=1):
                key = self._key(row)
                entry = fused.setdefault(key, {"row": row, "score": 0.0})
                entry["score"] += 1.0 / (self.rrf_k + rank)
                # Prefer richer metadata (web has URL/title) on collision.
                if row.url and not entry["row"].url:
                    entry["row"] = row

        add(bm25_rows)
        add(web_rows)

        ranked = sorted(
            fused.values(), key=lambda item: item["score"], reverse=True
        )
        results: list[RetrievalResult] = []
        for item in ranked[:topk]:
            row: RetrievalResult = item["row"]
            results.append(
                RetrievalResult(
                    doc_id=row.doc_id,
                    text=row.text,
                    score=item["score"],
                    source=row.source,
                    url=row.url,
                    title=row.title,
                )
            )
        return results

    @staticmethod
    def _key(row: RetrievalResult) -> str:
        # Text is what the generator consumes, so identical passages from
        # BM25 and the web are duplicates even when their IDs differ.
        normalized = " ".join((row.text or "").lower().split())
        if normalized:
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
            return "text:" + digest
        if row.url:
            return "url:" + row.url.lower().rstrip("/")
        return "id:" + str(row.doc_id)

    def summary(self) -> dict:
        payload = {"retriever": self.name, "rrf_k": self.rrf_k}
        for label, backend in (("bm25", self.bm25), ("web", self.web)):
            summarise = getattr(backend, "summary", None)
            if callable(summarise):
                payload[label] = summarise()
        return payload
