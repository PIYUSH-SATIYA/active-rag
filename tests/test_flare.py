import json
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
        results = []
        for q in queries:
            results.append([
                RetrievalResult(doc_id="1", text="Mock retrieved context 1.", score=0.9),
                RetrievalResult(doc_id="2", text="Mock retrieved context 2.", score=0.8)
            ])
        return results


class MockGenerator:
    def __init__(self, responses: list[GeneratorResult]):
        self.responses = responses
        self.call_count = 0
        
    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            # Simple mock tokenization by splitting spaces
            return {"input_ids": [text.split()]}
            
    @property
    def tokenizer(self):
        return self.MockTokenizer()

    def generate(self, prompt: str, **kwargs) -> GeneratorResult:
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return GeneratorResult(text="", token_ids=[], tokens=[], probabilities=[])


def test_high_confidence_no_retrieval():
    # Setup mock generator to return high confidence
    generator = MockGenerator([
        GeneratorResult(
            text="This is a confident answer.",
            token_ids=[1, 2, 3, 4],
            tokens=["This", " is", " a", " confident"],
            probabilities=[0.9, 0.95, 0.99, 0.9]
        ),
        GeneratorResult(text="", token_ids=[], tokens=[], probabilities=[])
    ])
    
    retriever = MockRetriever()
    config = {"confidence_threshold": 0.8, "max_generation_length": 100}
    
    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)
    result = agent.generate("What is FLARE?")
    
    assert "confident" in result.text
    assert len(result.retrieval_history) == 0
    assert len(result.confidence_history) > 0
    assert result.confidence_history[0]["is_confident"] is True


def test_low_confidence_retrieval_triggered():
    # Setup mock generator to return low confidence on first call
    # And then a confident response on regeneration
    generator = MockGenerator([
        GeneratorResult(
            text="I think",
            token_ids=[1, 2],
            tokens=["I", " think"],
            probabilities=[0.9, 0.4] # Low confidence
        ),
        GeneratorResult(
            text=" it is a RAG method.",
            token_ids=[3, 4, 5],
            tokens=[" it", " is", " a"],
            probabilities=[0.9, 0.95, 0.99]
        ),
        GeneratorResult(text="", token_ids=[], tokens=[], probabilities=[])
    ])
    
    retriever = MockRetriever()
    config = {"confidence_threshold": 0.8, "max_generation_length": 100}
    
    agent = FLAREAgent(generator=generator, retriever=retriever, config=config)
    result = agent.generate("What is FLARE?")
    
    assert len(result.retrieval_history) == 1
    assert result.confidence_history[0]["is_confident"] is False
    assert "Mock retrieved context 1." in result.retrieval_history[0]["retrieved_docs"][0]
    assert "it is a RAG method" in result.text
