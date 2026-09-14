"""
Full FLARE runner — loads dataset, runs FLARE on each example, saves results.

Usage (from repo root, with venv active):
    PYTHONPATH=. python scripts/run_flare.py --eval_mode flare

Output files:
    results/predictions.jsonl  — per-example prediction + metadata
    results/metrics.json       — EM and F1 aggregate scores
    results/traces.jsonl       — full FLARE trace per example
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.qwen import QwenModel
from src.flare.agent import FLAREAgent, extract_answer
from src.retriever import Retriever
from src.datasets import get_dataset

CONFIG_PATH = "configs/2wikihop_qwen_flare_config.json"
RESULTS_DIR = "results"


def _sep(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run FLARE or baselines.")
    parser.add_argument("--eval_mode", type=str, default="flare", 
                        choices=["no_retrieval", "single_retrieval", "flare"],
                        help="Evaluation mode to run")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name (e.g. strategyqa, asqa, wikiasp, 2wikihop)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override dataset path (e.g. data/strategyqa)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    print("Loading config...")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    ds_cfg = config.get("dataset", {})
    max_examples = ds_cfg.get("max_examples", 500)
    dataset_name = args.dataset if args.dataset else ds_cfg.get("name", "2wikihop")
    data_path = args.data_path if args.data_path else ds_cfg.get("path", "data/2wikimultihopqa")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    _sep("Loading Qwen model")
    print(f"Model : {config['model_name']}")
    print(f"4-bit : {config.get('quantize_4bit', True)}")
    print("This may take 1–2 minutes on first load...")

    generator = QwenModel.from_pretrained(
        model_name=config["model_name"],
        quantize_4bit=config.get("quantize_4bit", True),
    )
    print("Model loaded.")

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------
    if args.eval_mode == "no_retrieval":
        retriever = None
        print("Skipping retriever initialization for no_retrieval mode.")
    else:
        _sep("Connecting to Retriever (Pyserini)")
        es_cfg = config.get("elasticsearch", {}) # Keeping key name for backward compatibility
        es_index = es_cfg.get("index", "wikipedia-dpr")
        print(f"Index : {es_index}")
        retriever = Retriever(index_name=es_index)
        print("Retriever ready.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    _sep("Loading dataset")
    print(f"Name   : {dataset_name}")
    print(f"Path   : {data_path}")
    dataset = get_dataset(dataset_name, data_dir=data_path, split="dev")
    examples = list(dataset)[:max_examples]
    print(f"Loaded {len(examples)} examples (max_examples={max_examples}).")

    # ------------------------------------------------------------------
    # FLARE Agent
    # ------------------------------------------------------------------
    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)

    # ------------------------------------------------------------------
    # Run (with Resume Capability)
    # ------------------------------------------------------------------
    pred_path = os.path.join(RESULTS_DIR, "predictions.jsonl")
    trace_path = os.path.join(RESULTS_DIR, "traces.jsonl")
    
    completed_ids = set()
    all_em = []
    all_f1 = []

    if os.path.exists(pred_path):
        print("\nFound existing predictions.jsonl! Loading completed examples for resume...")
        with open(pred_path, "r") as f:
            for line in f:
                data = json.loads(line)
                completed_ids.add(data["id"])
                all_em.append(data["em"])
                all_f1.append(data["f1"])
        print(f"Resuming from example {len(completed_ids) + 1}...")

    pred_file = open(pred_path, "a")
    trace_file = open(trace_path, "a")

    for i, example in enumerate(examples):
        ex_id = str(example["id"])
        
        if ex_id in completed_ids:
            continue

        question = example["question"]
        gold_answer = example["answer"]

        _sep(f"Example {i + 1}/{len(examples)}")
        print(f"ID       : {ex_id}")
        print(f"Question : {question}")
        print(f"Gold     : {gold_answer}")

        result = agent.generate(question, eval_mode=args.eval_mode)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        # Extract short answer from the chain-of-thought output
        predicted_answer = extract_answer(result.text)

        em_scores = dataset.exact_match_score(predicted_answer, gold_answer)
        f1_scores = dataset.f1_score(predicted_answer, gold_answer)
        all_em.append(em_scores["correct"])
        all_f1.append(f1_scores["f1"])
        completed_ids.add(ex_id)

        print(f"\nFull output   : {result.text}")
        print(f"Predicted ans : {predicted_answer}")
        print(f"Gold answer   : {gold_answer}")
        print(f"EM            : {em_scores['correct']}")
        print(f"F1            : {f1_scores['f1']:.4f}")

        # Confidence / retrieval summary
        n_retrievals = len(result.retrieval_history)
        n_steps = len(result.confidence_history)
        print(f"Steps      : {n_steps}  |  Retrievals triggered: {n_retrievals}")

        # Print trace
        for entry in result.generation_history:
            kind = entry.get("type", "?")
            lookahead = entry.get("lookahead_text", "")
            min_p = entry.get("min_probability", None)
            flag = "✓" if entry.get("type") == "direct" else "↩"
            print(f"  {flag} Step {entry['step']} [{kind}] lookahead={lookahead!r}  min_prob={min_p:.3f}" if min_p is not None else f"  {flag} Step {entry['step']} [{kind}]")

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        pred_file.write(json.dumps({
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "full_output": result.text,
            "prediction": predicted_answer,
            "em": em_scores["correct"],
            "f1": f1_scores["f1"],
            "n_retrievals": n_retrievals,
            "n_steps": n_steps,
        }) + "\n")
        pred_file.flush()

        trace_file.write(json.dumps({
            "id": ex_id,
            "question": question,
            "generation_history": result.generation_history,
            "retrieval_history": result.retrieval_history,
            "confidence_history": result.confidence_history,
        }) + "\n")
        trace_file.flush()

    pred_file.close()
    trace_file.close()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    n = len(all_em)
    metrics = {
        "n_examples": n,
        "exact_match": round(sum(all_em) / n, 4) if n else 0.0,
        "f1": round(sum(all_f1) / n, 4) if n else 0.0,
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _sep("FINAL RESULTS")
    print(f"Examples : {metrics['n_examples']}")
    print(f"EM       : {metrics['exact_match']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"\nOutputs saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
