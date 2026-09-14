from dataclasses import dataclass
from typing import List, Optional
from elasticsearch import Elasticsearch, exceptions

@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float

class Retriever:
    def __init__(self, host: str = "http://localhost:9200", index_name: str = "wikipedia_dpr_dummy"):
        self.es = Elasticsearch([host])
        self.index_name = index_name

    def retrieve(self, queries: List[str], topk: int = 2, max_query_length: Optional[int] = None) -> List[List[RetrievalResult]]:
        results = []
        for query in queries:
            if max_query_length:
                # Simple word-based truncation for dummy implementation
                query = " ".join(query.split()[:max_query_length])
            
            body = {
                "size": topk,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "text"]
                    }
                }
            }
            
            try:
                response = self.es.search(index=self.index_name, body=body)
                hits = response.get("hits", {}).get("hits", [])
                
                query_results = []
                for hit in hits:
                    source = hit.get("_source", {})
                    query_results.append(RetrievalResult(
                        doc_id=source.get("id", ""),
                        text=source.get("text", ""),
                        score=hit.get("_score", 0.0)
                    ))
                results.append(query_results)
            except exceptions.ConnectionError as e:
                raise RuntimeError(f"Elasticsearch is unreachable: {e}")
            except exceptions.NotFoundError as e:
                raise ValueError(f"Index {self.index_name} missing: {e}")
            except Exception as e:
                raise RuntimeError(f"Elasticsearch retrieval failed: {e}")
                
        return results
