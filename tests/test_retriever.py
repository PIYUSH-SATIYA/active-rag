import pytest
from unittest.mock import patch

from src.retriever import RetrievalResult, Retriever
from elasticsearch import exceptions
from src.datasets import WikiMultiHopQA
from scripts.build_index import build_index

# A. RetrievalResult Tests
def test_retrieval_result_creation():
    result = RetrievalResult(doc_id="doc1", text="some text", score=0.99)
    assert result.doc_id == "doc1"
    assert result.text == "some text"
    assert result.score == 0.99
    assert isinstance(result.score, float)

# B. Retriever Tests
def test_retriever_empty_query_list(mocked_retriever):
    results = mocked_retriever.retrieve([])
    assert results == []

def test_retriever_single_query(mocked_retriever, mock_es_client):
    results = mocked_retriever.retrieve(["test query"])
    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0].doc_id == "1"
    assert results[0][0].text == "This is a dummy text."
    assert results[0][0].score == 1.23

def test_retriever_multiple_queries(mocked_retriever):
    results = mocked_retriever.retrieve(["query 1", "query 2"])
    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2

def test_retriever_topk(mocked_retriever, mock_es_client):
    mocked_retriever.retrieve(["query"], topk=5)
    call_args = mock_es_client.search.call_args[1]
    assert call_args["body"]["size"] == 5

def test_retriever_query_truncation(mocked_retriever, mock_es_client):
    query = "this is a long query with many words"
    mocked_retriever.retrieve([query], max_query_length=3)
    call_args = mock_es_client.search.call_args[1]
    assert call_args["body"]["query"]["multi_match"]["query"] == "this is a"

def test_retriever_connection_failure(mocked_retriever, mock_es_client):
    mock_es_client.search.side_effect = exceptions.ConnectionError(500, "Connection failed", "info")
    with pytest.raises(RuntimeError, match="Elasticsearch is unreachable"):
        mocked_retriever.retrieve(["query"])

def test_retriever_missing_index(mocked_retriever, mock_es_client):
    mock_es_client.search.side_effect = exceptions.NotFoundError(404, "Index missing")
    with pytest.raises(ValueError, match="Index dummy_index missing"):
        mocked_retriever.retrieve(["query"])

def test_retriever_unexpected_error(mocked_retriever, mock_es_client):
    mock_es_client.search.side_effect = Exception("Unknown error")
    with pytest.raises(RuntimeError, match="Elasticsearch retrieval failed"):
        mocked_retriever.retrieve(["query"])

# C. WikiMultiHopQA Dataset Tests
def test_wikimultihopqa_dummy_loads(dummy_data_dir):
    dataset = WikiMultiHopQA(data_dir=dummy_data_dir, split="dev")
    assert len(dataset) == 2

    first_item = dataset[0]
    assert first_item["id"] == "dummy_1"
    assert first_item["question"] == "What is the capital of France?"
    assert first_item["answer"] == "Paris"
    
    # Check required fields
    assert "gold_contexts" in first_item
    assert "supporting_facts" in first_item

def test_wikimultihopqa_alias(dummy_data_dir):
    dataset = WikiMultiHopQA(data_dir=dummy_data_dir, split="dev")
    aliases = dataset.get_aliases("dummy_1")
    assert aliases == ["dummy1_alias"]
    assert dataset.get_aliases("missing") == []

def test_answer_normalization():
    # If this is not implemented, the test will fail and prompt us to add it
    assert WikiMultiHopQA.normalize_answer("The Paris!") == "paris"

def test_exact_match():
    # If exact_match_score is missing, this will fail
    score = WikiMultiHopQA.exact_match_score("Paris", "The Paris!")
    assert score["correct"] == 1

def test_f1_score():
    # If f1_score is missing, this will fail
    score = WikiMultiHopQA.f1_score("Paris France", "Paris")
    assert score["f1"] > 0.0

# D. Build-index logic Tests
def test_build_index(mock_es_client, dummy_tsv_path):
    with patch("scripts.build_index.Elasticsearch", return_value=mock_es_client), \
         patch("scripts.build_index.helpers.bulk", return_value=(20, 0)) as mock_bulk:
        
        build_index("http://localhost:9200", "test_index", dummy_tsv_path)
        
        mock_es_client.indices.create.assert_called_once()
        mock_bulk.assert_called_once()
        mock_es_client.indices.refresh.assert_called_with(index="test_index")

def test_build_index_recreate(mock_es_client, dummy_tsv_path):
    mock_es_client.indices.exists.return_value = True
    with patch("scripts.build_index.Elasticsearch", return_value=mock_es_client), \
         patch("scripts.build_index.helpers.bulk", return_value=(20, 0)):
        
        build_index("http://localhost:9200", "test_index", dummy_tsv_path)
        
        mock_es_client.indices.delete.assert_called_with(index="test_index")
        mock_es_client.indices.create.assert_called_once()

# E. Integration Tests (require live Elasticsearch at localhost:9200)
# Probe availability first so CI/dev machines without Docker get a clean skip.
def _es_is_reachable(host: str = "localhost", port: int = 9200) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False

@pytest.mark.integration
def test_retriever_live_integration():
    """Run only when a real Elasticsearch instance is available.

    - ES unreachable → pytest.skip (clean skip, not a failure)
    - ES reachable but retrieval fails → pytest.fail (genuine bug)
    - ES reachable and retrieval succeeds → full assertions on result shape and types
    """
    if not _es_is_reachable():
        pytest.skip("Elasticsearch not available at localhost:9200 — skipping live integration test")

    retriever = Retriever(host="http://localhost:9200", index_name="wikipedia_dpr_dummy")

    # Must return the correct outer shape
    results = retriever.retrieve(["Albert Einstein", "Marie Curie"], topk=2)
    assert isinstance(results, list), "retrieve() must return a list"
    assert len(results) == 2, "One result-list per query"

    for query_results in results:
        assert isinstance(query_results, list), "Each per-query result must be a list"
        for item in query_results:
            assert isinstance(item, RetrievalResult), "Items must be RetrievalResult instances"
            assert isinstance(item.doc_id, str) and item.doc_id, "doc_id must be a non-empty string"
            assert isinstance(item.text, str) and item.text, "text must be a non-empty string"
            assert isinstance(item.score, float) and item.score >= 0.0, "score must be a non-negative float"
