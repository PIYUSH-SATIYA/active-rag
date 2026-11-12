from src.experiment import (
    extract_prediction,
    rescore_rows,
    score_prediction,
    select_examples,
)


def _examples():
    return [
        {"id": "a", "question": "qa", "answer": "1"},
        {"id": "b", "question": "qb", "answer": "2"},
        {"id": "c", "question": "qc", "answer": "3"},
        {"id": "d", "question": "qd", "answer": "4"},
    ]


def test_select_examples_preserves_explicit_id_order():
    selected = select_examples(_examples(), ids=["c", "a"])
    assert [row["id"] for row in selected] == ["c", "a"]


def test_select_examples_limit_and_shard():
    selected = select_examples(_examples(), n=4, shard_id=1, num_shards=2)
    assert [row["id"] for row in selected] == ["b", "d"]


def test_extract_prediction_strategyqa_maps_yes_no():
    assert extract_prediction("strategyqa", "So the answer is yes.") == "yes"
    assert extract_prediction(
        "strategyqa", "The evidence says no. So the answer is no."
    ) == "no"


def test_extract_prediction_2wiki_uses_answer_sentence():
    text = "Some reasoning. So the answer is Małgorzata Braunek."
    assert extract_prediction("2wikihop", text) == "Małgorzata Braunek"


def test_score_prediction_strategyqa_normalises_yes_to_true():
    em, f1 = score_prediction("strategyqa", "yes", "true")
    assert em == 1.0
    assert f1 == 1.0


def test_rescore_rows_uses_shared_extractor_and_dedupes():
    rows = [
        {
            "id": "1",
            "gold_answer": "true",
            "prediction": "wrong",
            "full_output": "Reasoning. So the answer is yes.",
        },
        {
            "id": "1",
            "gold_answer": "true",
            "prediction": "yes",
            "full_output": "So the answer is yes.",
        },
        {
            "id": "2",
            "gold_answer": "false",
            "prediction": "no",
            "full_output": "So the answer is no.",
        },
    ]
    rescored, metrics = rescore_rows(rows, "strategyqa", ids=["1", "2"])
    assert len(rescored) == 2
    assert [row["id"] for row in rescored] == ["1", "2"]
    assert metrics["n_examples"] == 2
    assert metrics["exact_match"] == 1.0
    assert metrics["coverage"] == 1.0


def test_rescore_rows_reports_partial_coverage():
    rows = [
        {
            "id": "1",
            "gold_answer": "yes",
            "prediction": "yes",
            "full_output": "So the answer is yes.",
        }
    ]
    _, metrics = rescore_rows(rows, "strategyqa", ids=["1", "2"])
    assert metrics["n_examples"] == 1
    assert metrics["coverage"] == 0.5
