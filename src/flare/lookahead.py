from typing import Any

from src.templates import build_prompt, build_retrieval_prompt


def look_ahead_generate(
    generator: Any,
    question: str,
    current_answer: str,
    retrieved_contexts: list[str],
    max_new_tokens: int = 64,
):
    """
    Generates a look-ahead continuation of the current answer.

    If retrieved_contexts is non-empty, the prompt includes the retrieved passages
    (this is the "regenerate with context" path). Otherwise it is a plain look-ahead.

    Args:
        generator:          QwenModel instance with a .generate(prompt, **kwargs) method.
        question:           The original question being answered.
        current_answer:     The answer text generated so far.
        retrieved_contexts: List of retrieved passage strings. Empty during plain look-ahead.
        max_new_tokens:     Maximum tokens to generate in this step.

    Returns:
        GeneratorResult with text, tokens, token_ids, probabilities, finish_reason.
    """
    if retrieved_contexts:
        prompt = build_retrieval_prompt(
            question=question,
            current_answer=current_answer,
            contexts=retrieved_contexts,
        )
    else:
        prompt = build_prompt(
            question=question,
            current_answer=current_answer,
        )

    return generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # temperature=0 → deterministic (greedy)
        temperature=1.0,          # required placeholder when do_sample=False
    )
