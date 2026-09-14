import pytest
import os
from unittest.mock import Mock, patch
from elasticsearch import Elasticsearch
from src.retriever import Retriever

@pytest.fixture
def dummy_data_dir():
    return "data/dummy"

@pytest.fixture
def dummy_tsv_path(dummy_data_dir):
    return os.path.join(dummy_data_dir, "psgs_w100_dummy.tsv")

@pytest.fixture
def mock_es_response():
    return {
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_score": 1.23,
                    "_source": {
                        "id": "1",
                        "text": "This is a dummy text.",
                        "title": "Dummy Title"
                    }
                },
                {
                    "_id": "2",
                    "_score": 0.89,
                    "_source": {
                        "id": "2",
                        "text": "This is another dummy text.",
                        "title": "Another Dummy"
                    }
                }
            ]
        }
    }

@pytest.fixture
def mock_es_client(mock_es_response):
    client = Mock(spec=Elasticsearch)
    client.search.return_value = mock_es_response
    
    # For build_index tests
    client.indices = Mock()
    client.indices.exists.return_value = False
    return client

@pytest.fixture
def mocked_retriever(mock_es_client):
    with patch("src.retriever.Elasticsearch", return_value=mock_es_client):
        retriever = Retriever(host="http://localhost:9200", index_name="dummy_index")
        # Override the actual es client with the mock because the patch might just 
        # intercept the constructor.
        retriever.es = mock_es_client
        return retriever
