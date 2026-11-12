"""Pluggable retrieval backends for FLARE.

The original project used a single Pyserini BM25 retriever.  The Web-FLARE
extension keeps the retriever interface identical so the FLARE agent does not
need to know whether passages came from a local index or the live web.
"""

from src.retrieval.base import RetrievalResult, RetrieverProtocol

__all__ = ["RetrievalResult", "RetrieverProtocol"]
