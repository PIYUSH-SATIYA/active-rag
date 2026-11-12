import json

from src.experiment import run_job
from src.models.qwen import GeneratorResult


class MockGenerator:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        text = f"Reasoning about the question. So the answer is {self.answer}."
        return GeneratorResult(
            text=text,
            token_ids=[1, 2, 3],
            tokens=["Reasoning", " about", " the"],
            probabilities=[0.99, 0.98, 0.97],
            finish_reason="eos_token",
        )


def _write_dataset(path):
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"_id": "ex1", "question": "q1", "answer": "Paris"},
        {"_id": "ex2", "question": "q2", "answer": "Berlin"},
    ]
    (path / "dev.json").write_text(json.dumps(rows))


def test_run_job_writes_metrics_and_resumes(tmp_path):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)
    results_root = tmp_path / "results"
    generator = MockGenerator("Paris")

    job = {
        "id": "smoke",
        "dataset": "2wikihop",
        "data_path": str(data_dir),
        "retriever": "none",
        "mode": "no_retrieval",
        "n": 2,
    }
    config = {
        "max_generation_length": 32,
        "look_ahead_filter_prob": 0.8,
        "topk": 2,
        "max_query_length": 64,
        "look_ahead_steps": 16,
    }

    metrics = run_job(
        job=job,
        generator=generator,
        base_config=config,
        results_root=str(results_root),
        verbose=False,
    )
    assert metrics["n_examples"] == 2
    assert metrics["coverage"] == 1.0
    assert metrics["exact_match"] == 0.5
    assert generator.calls == 2

    pred_path = results_root / "jobs" / "smoke" / "predictions.jsonl"
    assert len(pred_path.read_text().strip().splitlines()) == 2

    # Re-running must not append duplicate rows.
    metrics_again = run_job(
        job=job,
        generator=generator,
        base_config=config,
        results_root=str(results_root),
        verbose=False,
    )
    assert generator.calls == 2
    assert metrics_again["n_examples"] == 2
    assert len(pred_path.read_text().strip().splitlines()) == 2
