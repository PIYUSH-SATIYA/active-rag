import sys
import os

# Add root directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import Retriever

def run_tests():
    print("Initializing Retriever (pointing to local Elasticsearch dummy index)...")
    try:
        retriever = Retriever(host="http://localhost:9200", index_name="wikipedia_dpr_dummy")
    except Exception as e:
        print(f"Failed to initialize Retriever: {e}")
        return

    queries = [
        "Who developed the theory of relativity?",
        "Where is the Eiffel Tower located?",
        "Who was the first woman to win a Nobel Prize?"
    ]
    
    print(f"\nRunning {len(queries)} queries against the dummy index...")
    try:
        results = retriever.retrieve(queries, topk=2)
        
        for q, res_list in zip(queries, results):
            print(f"\nQUERY: {q}")
            if not res_list:
                print("  No results found (or Elasticsearch is not running).")
            else:
                for i, res in enumerate(res_list, 1):
                    print(f"  RESULT {i} (ID: {res.doc_id}, Score: {res.score:.4f})")
                    print(f"  {res.text}")
    except Exception as e:
        print(f"Failed during retrieval: {e}")
        print("Make sure Elasticsearch is running and the dummy index is built.")

if __name__ == "__main__":
    run_tests()
