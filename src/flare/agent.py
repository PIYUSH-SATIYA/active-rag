import re
from dataclasses import dataclass, field
from typing import Any

from src.flare.confidence import calculate_confidence
from src.flare.lookahead import look_ahead_generate
from src.templates import build_prompt, build_retrieval_prompt


# Matches "So the answer is X." or "the answer is X" at end of text
_ANSWER_RE = re.compile(r'[Ss]o the answer is ([^.\n]+)\.?', re.IGNORECASE)


def extract_answer(text: str) -> str:
    """
    Extracts the final short answer from a chain-of-thought generation.
    Looks for 'So the answer is X.' pattern (original FLARE style).
    Falls back to the full text if pattern not found.
    """
    # Truncate at \n\n first (the FLARE loop stop boundary)
    text = text.split('\n\n')[0].strip()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


@dataclass
class FLAREResult:
    text: str
    retrieval_history: list[dict] = field(default_factory=list)
    generation_history: list[dict] = field(default_factory=list)
    confidence_history: list[dict] = field(default_factory=list)


# Sentence boundary pattern — FLARE truncates look-ahead at sentence boundaries
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


def _truncate_at_sentence_boundary(text: str) -> str:
    """
    Returns only the first complete sentence from the look-ahead text.
    Mirrors the original FLARE `look_ahead_truncate_at_boundary='sentence'` logic.
    """
    parts = _SENT_BOUNDARY.split(text.strip(), maxsplit=1)
    return parts[0].strip()


class FLAREAgent:
    def __init__(self, generator: Any, retriever: Any, config: dict):
        self.generator = generator
        self.retriever = retriever
        self.config = config

    def generate(self, question: str, eval_mode: str = "flare") -> FLAREResult:
        """
        Executes generation in one of 3 modes:
        - 'no_retrieval': plain generation without any retriever.
        - 'single_retrieval': retrieve once at the beginning based on the question.
        - 'flare': active forward-looking retrieval with implicit query masking.
        """
        if eval_mode == "no_retrieval":
            return self._generate_no_retrieval(question)
        elif eval_mode == "single_retrieval":
            return self._generate_single_retrieval(question)
        elif eval_mode == "flare":
            return self._generate_flare(question)
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

    def _generate_no_retrieval(self, question: str) -> FLAREResult:
        max_len = self.config.get("max_generation_length", 256)
        result = look_ahead_generate(
            generator=self.generator,
            question=question,
            current_answer="",
            retrieved_contexts=[],
            max_new_tokens=max_len,
        )
        return FLAREResult(text=result.text.split('\n\n')[0].strip())

    def _generate_single_retrieval(self, question: str) -> FLAREResult:
        max_len = self.config.get("max_generation_length", 256)
        top_k = self.config.get("topk", 2)
        
        # Retrieve once using the question
        retrieved_docs = self.retriever.retrieve([question], topk=top_k)
        contexts = [doc.text for doc in retrieved_docs[0]] if retrieved_docs else []
        
        result = look_ahead_generate(
            generator=self.generator,
            question=question,
            current_answer="",
            retrieved_contexts=contexts,
            max_new_tokens=max_len,
        )
        
        return FLAREResult(
            text=result.text.split('\n\n')[0].strip(),
            retrieval_history=[{"step": 0, "query": question, "retrieved_docs": contexts}]
        )

    def _generate_flare(self, question: str) -> FLAREResult:
        max_len = self.config.get("max_generation_length", 256)
        threshold = self.config.get("look_ahead_filter_prob", 0.8)
        beta_masking_threshold = self.config.get("beta_masking_threshold", 0.4)
        top_k = self.config.get("topk", 2)
        max_query_len = self.config.get("max_query_length", 64)
        look_ahead_steps = self.config.get("look_ahead_steps", 64)

        current_answer = ""
        retrieved_contexts: list[str] = []
        retrieval_history = []
        generation_history = []
        confidence_history = []

        step = 0
        total_tokens_generated = 0

        while total_tokens_generated < max_len:
            step += 1

            lookahead_result = look_ahead_generate(
                generator=self.generator,
                question=question,
                current_answer=current_answer,
                retrieved_contexts=retrieved_contexts,
                max_new_tokens=look_ahead_steps,
            )

            lookahead_sentence = _truncate_at_sentence_boundary(lookahead_result.text)

            truncated_len = len(lookahead_sentence)
            truncated_tokens = []
            truncated_probs = []
            char_pos = 0
            for tok, prob in zip(lookahead_result.tokens, lookahead_result.probabilities):
                if char_pos >= truncated_len:
                    break
                truncated_tokens.append(tok)
                truncated_probs.append(prob)
                char_pos += len(tok)

            generation_history.append({
                "step": step,
                "lookahead_text": lookahead_sentence,
                "probabilities": truncated_probs,
            })

            is_confident = calculate_confidence(truncated_probs, threshold)
            min_prob = min(truncated_probs) if truncated_probs else 1.0
            confidence_history.append({
                "step": step,
                "is_confident": is_confident,
                "min_probability": min_prob,
            })

            if is_confident:
                chunk = lookahead_sentence
                current_answer += (" " if current_answer else "") + chunk
                total_tokens_generated += len(truncated_tokens)

                generation_history[-1]["accepted"] = chunk
                generation_history[-1]["type"] = "direct"
            else:
                # ----------------------------------------------------------------
                # Implicit Masking (Confidence-based Query Formulation)
                # ----------------------------------------------------------------
                masked_query_tokens = []
                for tok, prob in zip(truncated_tokens, truncated_probs):
                    if prob < beta_masking_threshold:
                        # Skip or replace with nothing to avoid hallucinated constraints
                        pass
                    else:
                        masked_query_tokens.append(tok)
                
                query = "".join(masked_query_tokens).strip()
                # Fallback to the full sentence if the query became empty
                if not query:
                    query = lookahead_sentence

                query_words = query.split()
                query = " ".join(query_words[:max_query_len])

                retrieved_docs = self.retriever.retrieve(
                    [query], topk=top_k, max_query_length=max_query_len
                )
                docs_for_query = retrieved_docs[0] if retrieved_docs else []
                new_contexts = [doc.text for doc in docs_for_query]

                retrieved_contexts = new_contexts

                retrieval_history.append({
                    "step": step,
                    "query": query,
                    "retrieved_docs": new_contexts,
                })

                regen_result = look_ahead_generate(
                    generator=self.generator,
                    question=question,
                    current_answer=current_answer,
                    retrieved_contexts=retrieved_contexts,
                    max_new_tokens=look_ahead_steps,
                )

                regen_sentence = _truncate_at_sentence_boundary(regen_result.text)
                current_answer += (" " if current_answer else "") + regen_sentence
                total_tokens_generated += len(regen_result.tokens)

                generation_history[-1]["regen_text"] = regen_sentence
                generation_history[-1]["type"] = "retrieval_regen"

            if not lookahead_sentence.strip():
                break

            if lookahead_result.finish_reason in ("eos_token", "length"):
                if lookahead_result.finish_reason == "eos_token":
                    break

            if '\n\n' in current_answer:
                current_answer = current_answer.split('\n\n')[0].strip()
                break

        return FLAREResult(
            text=current_answer.strip(),
            retrieval_history=retrieval_history,
            generation_history=generation_history,
            confidence_history=confidence_history,
        )
