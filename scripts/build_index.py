import os
import argparse
from elasticsearch import Elasticsearch, helpers

def build_index(es_host: str, index_name: str, tsv_path: str, chunk_size: int = 500):
    es = Elasticsearch([es_host])
    
    # Define mapping for BM25 (Elasticsearch defaults to BM25 for text fields)
    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"}
            }
        }
    }
    
    print(f"Creating index '{index_name}'...")
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mapping)
    
    def generate_actions():
        with open(tsv_path, 'r', encoding='utf-8') as f:
            header = f.readline() # Skip header
            for line in f:
                parts = line.strip('\n').split('\t')
                if len(parts) >= 3:
                    doc_id, text, title = parts[0], parts[1], parts[2]
                    yield {
                        "_index": index_name,
                        "_source": {
                            "id": doc_id,
                            "text": text,
                            "title": title
                        }
                    }

    print(f"Indexing documents from {tsv_path} into '{index_name}'...")
    success, failed = helpers.bulk(es, generate_actions(), chunk_size=chunk_size, stats_only=True)
    print(f"Successfully indexed {success} documents. Failed: {failed}")
    
    # Refresh to make documents immediately searchable
    es.indices.refresh(index=index_name)
    print("Index refresh completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Elasticsearch index from TSV")
    parser.add_argument("--host", type=str, default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--index", type=str, required=True, help="Elasticsearch index name")
    parser.add_argument("--tsv", type=str, required=True, help="Path to corpus TSV file")
    
    args = parser.parse_args()
    try:
        build_index(args.host, args.index, args.tsv)
    except Exception as e:
        print(f"Error building index: {e}")
        # When missing elasticsearch module, we handle it gracefully here if called without it
