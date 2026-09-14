import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.flare.confidence import calculate_confidence
from src.flare.lookahead import look_ahead_generate
from src.templates import QA_PROMPT_TEMPLATE


@dataclass
class FLAREResult:
    text: str
    retrieval_history: list[dict] = field(default_factory=list)
    generation_history: list[dict] = field(default_factory=list)
    confidence_history: list[dict] = field(default_factory=list)


class FLAREAgent:
    def __init__(self, generator: Any, retriever: Any, config: dict):
        self.generator = generator
        self.retriever = retriever
        self.config = config

    def generate(self, question: str) -> FLAREResult:
        """
        Executes the FLARE generation algorithm for a single question.
        """
        max_length = self.config.get("max_generation_length", 256)
        threshold = self.config.get("confidence_threshold", 0.8)
        top_k = self.config.get("retrieval_top_k", 2)
        
        current_answer = ""
        retrieval_history = []
        generation_history = []
        confidence_history = []
        
        # Simple loop for generating in chunks
        # In a real FLARE implementation, it breaks sentences by punctuation, 
        # but for simplicity we'll generate chunks and evaluate confidence.
        
        step = 0
        while len(self.generator.tokenizer(current_answer, return_tensors="pt")["input_ids"][0]) < max_length:
            step += 1
            
            # 1. Look-ahead generation
            lookahead_result = look_ahead_generate(
                generator=self.generator,
                question=question,
                current_answer=current_answer,
                max_new_tokens=32  # Generate a short chunk
            )
            
            generation_history.append({
                "step": step,
                "text": lookahead_result.text,
                "probabilities": lookahead_result.probabilities
            })
            
            # 2. Check confidence
            is_confident = calculate_confidence(lookahead_result.probabilities, threshold)
            confidence_history.append({
                "step": step,
                "is_confident": is_confident,
                "min_probability": min(lookahead_result.probabilities) if lookahead_result.probabilities else 1.0
            })
            
            if is_confident:
                # 3a. Accept generation
                current_answer += lookahead_result.text
            else:
                # 3b. Retrieve and regenerate
                # We use the lookahead text as the query for retrieval
                query = current_answer + lookahead_result.text
                retrieved_docs = self.retriever.retrieve([query], topk=top_k)
                
                # retrieved_docs is a list of lists of RetrievalResult
                docs_for_query = retrieved_docs[0] if retrieved_docs else []
                context_texts = [doc.text for doc in docs_for_query]
                context_str = "\n\n".join(context_texts)
                
                retrieval_history.append({
                    "step": step,
                    "query": query,
                    "retrieved_docs": context_texts
                })
                
                # Regenerate with context
                prompt = QA_PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question
                )
                
                # Ask generator to continue the current answer given the new prompt
                # For simplicity, we just pass the prompt + current_answer and get the next chunk
                regen_prompt = prompt + "\n" + current_answer
                regen_result = self.generator.generate(regen_prompt, max_new_tokens=32)
                
                current_answer += regen_result.text
                
                generation_history.append({
                    "step": step,
                    "type": "regeneration",
                    "text": regen_result.text
                })
                
            # Basic stopping condition if the model generates a stop token or stops early
            # We check if the text generated is empty
            if not lookahead_result.text.strip():
                break
                
        return FLAREResult(
            text=current_answer.strip(),
            retrieval_history=retrieval_history,
            generation_history=generation_history,
            confidence_history=confidence_history
        )
