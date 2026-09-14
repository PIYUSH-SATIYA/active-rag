from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float

class Retriever:
    def __init__(self, host: str = None, index_name: str = "wikipedia-dpr"):
        print(f"Loading prebuilt index {index_name} via pyserini...")
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError:
            raise ImportError("Please install pyserini: pip install pyserini faiss-cpu")
        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)

    def retrieve(self, queries: List[str], topk: int = 2, max_query_length: Optional[int] = None) -> List[List[RetrievalResult]]:
        results = []
        for query in queries:
            if max_query_length:
                query = " ".join(query.split()[:max_query_length])
            
            # Pyserini BM25 search
            hits = self.searcher.search(query, k=topk)
            
            query_results = []
            for hit in hits:
                # Extract the text content from the Pyserini hit
                try:
                    doc_json = json.loads(hit.raw)
                    text = doc_json.get("contents", "")
                except Exception:
                    text = hit.raw
                    
                query_results.append(RetrievalResult(
                    doc_id=hit.docid,
                    text=text,
                    score=hit.score
                ))
            results.append(query_results)
        return results
