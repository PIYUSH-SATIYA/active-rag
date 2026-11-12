from src.retrieval.base import RetrievalResult
from src.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    name = "fake"

    def __init__(self, batches):
        self.batches = batches

    def retrieve(self, queries, topk=2, max_query_length=None):
        return self.batches


def _result(doc_id, text, url=None, score=1.0, source="bm25"):
    return RetrievalResult(
        doc_id=doc_id, text=text, score=score, source=source, url=url
    )


def test_rrf_prefers_documents_seen_by_both_retrievers():
    bm25 = FakeRetriever(
        [[_result("a", "A"), _result("b", "B")]]
    )
    web = FakeRetriever(
        [[_result("b", "B", url="https://b.example"), _result("c", "C", url="https://c.example", source="web")]]
    )
    hybrid = HybridRetriever(bm25=bm25, web=web, rrf_k=60)

    rows = hybrid.retrieve(["q"], topk=3)[0]
    assert [row.text for row in rows] == ["B", "A", "C"]
    assert rows[0].url == "https://b.example"


def test_rrf_deduplicates_same_url():
    bm25 = FakeRetriever([[]])
    web = FakeRetriever(
        [[
            _result("w1", "same", url="https://x.example/a", source="web"),
            _result("w2", "same", url="https://x.example/a/", source="web"),
        ]]
    )
    hybrid = HybridRetriever(bm25=bm25, web=web)
    rows = hybrid.retrieve(["q"], topk=5)[0]
    assert len(rows) == 1


def test_hybrid_handles_query_count():
    bm25 = FakeRetriever([[_result("a", "A")], [_result("b", "B")]])
    web = FakeRetriever([[_result("c", "C", source="web")], []])
    hybrid = HybridRetriever(bm25=bm25, web=web)
    batches = hybrid.retrieve(["q1", "q2"], topk=2)
    assert len(batches) == 2
    assert batches[0][0].text == "A"
    assert batches[1][0].text == "B"
