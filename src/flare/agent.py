import re
from dataclasses import dataclass, field
from typing import Any

from src.flare.confidence import calculate_confidence

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


from src.flare.lookahead import look_ahead_generate
from src.templates import build_prompt, build_retrieval_prompt


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

    def generate(self, question: str) -> FLAREResult:
        """
        Executes the FLARE generation algorithm for a single question.

        Algorithm (mirrors original QueryAgent in openai_api.py):
          1. Build the initial prompt (few-shot + question).
          2. Look-ahead: generate `look_ahead_steps` tokens.
          3. Truncate look-ahead at the first sentence boundary.
          4. Check confidence: if any token prob < look_ahead_filter_prob → uncertain.
          5. If uncertain: use look-ahead text as retrieval query → retrieve → regenerate
             the current chunk with the retrieved context injected.
          6. If confident: accept the look-ahead text directly.
          7. Repeat from step 2 until max_generation_length tokens or EOS.
        """
        max_len = self.config.get("max_generation_length", 256)
        threshold = self.config.get("look_ahead_filter_prob", 0.8)
        top_k = self.config.get("topk", 2)
        max_query_len = self.config.get("max_query_length", 64)
        look_ahead_steps = self.config.get("look_ahead_steps", 64)

        current_answer = ""
        retrieved_contexts: list[str] = []  # accumulated context passages
        retrieval_history = []
        generation_history = []
        confidence_history = []

        step = 0
        total_tokens_generated = 0

        while total_tokens_generated < max_len:
            step += 1

            # ----------------------------------------------------------------
            # 1. Look-ahead generation
            # ----------------------------------------------------------------
            lookahead_result = look_ahead_generate(
                generator=self.generator,
                question=question,
                current_answer=current_answer,
                retrieved_contexts=retrieved_contexts,
                max_new_tokens=look_ahead_steps,
            )

            # Truncate at first sentence boundary (mirrors original behaviour)
            lookahead_sentence = _truncate_at_sentence_boundary(lookahead_result.text)

            # Align token lists to the truncated sentence length
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

            # ----------------------------------------------------------------
            # 2. Confidence check
            # ----------------------------------------------------------------
            is_confident = calculate_confidence(truncated_probs, threshold)
            min_prob = min(truncated_probs) if truncated_probs else 1.0
            confidence_history.append({
                "step": step,
                "is_confident": is_confident,
                "min_probability": min_prob,
            })

            if is_confident:
                # ----------------------------------------------------------------
                # 3a. Accept look-ahead directly
                # ----------------------------------------------------------------
                chunk = lookahead_sentence
                current_answer += (" " if current_answer else "") + chunk
                total_tokens_generated += len(truncated_tokens)

                generation_history[-1]["accepted"] = chunk
                generation_history[-1]["type"] = "direct"
            else:
                # ----------------------------------------------------------------
                # 3b. Uncertain → retrieve using look-ahead text as query
                #     (NOT current_answer — this is the core FLARE idea)
                # ----------------------------------------------------------------

                # Build retrieval query: only the uncertain look-ahead sentence,
                # optionally word-truncated to max_query_length
                query_words = lookahead_sentence.split()
                query = " ".join(query_words[:max_query_len])

                retrieved_docs = self.retriever.retrieve(
                    [query], topk=top_k, max_query_length=max_query_len
                )
                docs_for_query = retrieved_docs[0] if retrieved_docs else []
                new_contexts = [doc.text for doc in docs_for_query]

                # Accumulate retrieved contexts (replace strategy, like original)
                retrieved_contexts = new_contexts

                retrieval_history.append({
                    "step": step,
                    "query": query,
                    "retrieved_docs": new_contexts,
                })

                # ----------------------------------------------------------------
                # Regenerate the current chunk WITH the retrieved context.
                # We do NOT keep the failed lookahead — we regenerate fresh.
                # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # 4. Stop conditions
            # ----------------------------------------------------------------
            # (a) Model returned EOS (empty output after stripping)
            if not lookahead_sentence.strip():
                break

            # (b) finish_reason from model signals end of sequence
            if lookahead_result.finish_reason in ("eos_token", "length"):
                if lookahead_result.finish_reason == "eos_token":
                    break
                # "length" just means we hit max_new_tokens — continue the loop

            # (c) Original FLARE final_stop_sym = '\n\n'
            # The model has finished the answer if it generated a double newline.
            if '\n\n' in current_answer:
                current_answer = current_answer.split('\n\n')[0].strip()
                break

        return FLAREResult(
            text=current_answer.strip(),
            retrieval_history=retrieval_history,
            generation_history=generation_history,
            confidence_history=confidence_history,
        )
