import argparse
from scripts.build_index import build_index

def main():
    parser = argparse.ArgumentParser(description="Data preparation tasks for FLARE")
    parser.add_argument("--task", type=str, required=True, choices=["build_elasticsearch"], help="Task to perform")
    parser.add_argument("--inp", type=str, required=True, help="Input TSV path")
    parser.add_argument("index_name", type=str, help="Elasticsearch index name")
    
    args = parser.parse_args()
    
    if args.task == "build_elasticsearch":
        try:
            build_index(es_host="http://localhost:9200", index_name=args.index_name, tsv_path=args.inp)
        except Exception as e:
            print(f"Failed to build Elasticsearch index: {e}")

if __name__ == "__main__":
    main()
