"""Common retrieval data structures.

`RetrievalResult` is intentionally a superset of the old dataclass in
``src/retriever.py``: the FLARE agent only reads ``.text``, while the web
backend adds source metadata that is useful in traces and analysis.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    source: str = "bm25"
    url: Optional[str] = None
    title: Optional[str] = None

    def as_meta(self) -> dict:
        """Serialisable metadata for ``retrieval_history``."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "score": self.score,
        }


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Interface consumed by ``FLAREAgent``."""

    name: str

    def retrieve(
        self,
        queries: list[str],
        topk: int = 2,
        max_query_length: Optional[int] = None,
    ) -> list[list[RetrievalResult]]:
        ...
