import pytest
from dataclasses import dataclass
from src.flare.agent import FLAREAgent
from src.models.qwen import GeneratorResult


@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float


class MockRetriever:
    def retrieve(self, queries: list[str], topk: int = 2, max_query_length=None):
        return [
            [
                RetrievalResult(doc_id="1", text="Mock retrieved context 1.", score=0.9),
                RetrievalResult(doc_id="2", text="Mock retrieved context 2.", score=0.8),
            ]
            for _ in queries
        ]


class MockGenerator:
    """
    Deterministic mock generator. Returns pre-configured responses in order.
    Accepts any kwargs (e.g. max_new_tokens, do_sample) without error.
    """

    def __init__(self, responses: list[GeneratorResult]):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> GeneratorResult:
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return GeneratorResult(
            text="", token_ids=[], tokens=[], probabilities=[], finish_reason="eos_token"
        )


def test_high_confidence_no_retrieval():
    """When all token probs are above threshold, no retrieval should happen."""
    generator = MockGenerator([
        GeneratorResult(
            text="This is a confident answer.",
            token_ids=[1, 2, 3, 4],
            tokens=["This", " is", " a", " confident."],
            probabilities=[0.9, 0.95, 0.99, 0.91],
            finish_reason="eos_token",
        ),
    ])

    retriever = MockRetriever()
    config = {
        "look_ahead_filter_prob": 0.8,
        "max_generation_length": 100,
        "topk": 2,
        "max_query_length": 64,
        "look_ahead_steps": 64,
    }

    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)
    result = agent.generate("What is FLARE?")

    assert "confident" in result.text
    assert len(result.retrieval_history) == 0
    assert len(result.confidence_history) > 0
    assert result.confidence_history[0]["is_confident"] is True


def test_low_confidence_retrieval_triggered():
    """When a token prob drops below threshold, retrieval + regen should occur."""
    generator = MockGenerator([
        # Step 1 look-ahead — low confidence (one token below 0.8)
        GeneratorResult(
            text="I think it works.",
            token_ids=[1, 2],
            tokens=["I", " think"],
            probabilities=[0.9, 0.4],   # 0.4 < 0.8 → uncertain
            finish_reason=None,
        ),
        # Step 1 regen — with context injected
        GeneratorResult(
            text="It is a RAG method.",
            token_ids=[3, 4, 5],
            tokens=[" It", " is", " a"],
            probabilities=[0.9, 0.95, 0.99],
            finish_reason="eos_token",
        ),
    ])

    retriever = MockRetriever()
    config = {
        "look_ahead_filter_prob": 0.8,
        "max_generation_length": 100,
        "topk": 2,
        "max_query_length": 64,
        "look_ahead_steps": 64,
    }

    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)
    result = agent.generate("What is FLARE?")

    # Retrieval must have been triggered exactly once
    assert len(result.retrieval_history) == 1

    # The retrieval query must be the lookahead text ONLY, not current_answer + lookahead
    assert result.retrieval_history[0]["query"] == "I think it works."

    # Confidence history shows first step was uncertain
    assert result.confidence_history[0]["is_confident"] is False

    # Retrieved docs must appear in retrieval history
    assert "Mock retrieved context 1." in result.retrieval_history[0]["retrieved_docs"]

    # Final answer must contain the regen text, not the failed lookahead
    assert "RAG method" in result.text


def test_retrieval_query_is_only_lookahead_not_full_answer():
    """Core FLARE invariant: retrieval query == look-ahead text, not current_answer + lookahead."""
    generator = MockGenerator([
        # First confident chunk — this becomes current_answer
        GeneratorResult(
            text="FLARE is a method.",
            token_ids=[1, 2, 3],
            tokens=["FLARE", " is", " a"],
            probabilities=[0.95, 0.95, 0.95],
            finish_reason=None,
        ),
        # Second look-ahead — uncertain
        GeneratorResult(
            text="It uses uncertain retrieval.",
            token_ids=[4, 5],
            tokens=["It", " uses"],
            probabilities=[0.95, 0.3],  # low confidence
            finish_reason=None,
        ),
        # Regen with context
        GeneratorResult(
            text="It retrieves passages actively.",
            token_ids=[6, 7, 8],
            tokens=["It", " retrieves", " passages"],
            probabilities=[0.95, 0.95, 0.95],
            finish_reason="eos_token",
        ),
    ])

    retriever = MockRetriever()
    config = {
        "look_ahead_filter_prob": 0.8,
        "max_generation_length": 100,
        "topk": 2,
        "max_query_length": 64,
        "look_ahead_steps": 64,
    }

    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)
    result = agent.generate("What is FLARE?")

    assert len(result.retrieval_history) == 1
    query = result.retrieval_history[0]["query"]

    # The query must NOT contain the first answer chunk
    assert "FLARE is a method" not in query
    # The query must be derived from the uncertain lookahead only
    assert "It uses uncertain retrieval" in query
