from typing import Any
from src.templates import LOOK_AHEAD_PROMPT_TEMPLATE

def look_ahead_generate(
    generator: Any,
    question: str,
    current_answer: str,
    max_new_tokens: int = 64
):
    """
    Generates a look-ahead continuation of the current answer.
    
    Args:
        generator: The QwenModel instance (or mock) that has a `generate` method.
        question (str): The original question.
        current_answer (str): The generated answer so far.
        max_new_tokens (int): Maximum tokens for the look-ahead generation.
        
    Returns:
        GeneratorResult: Contains text, tokens, token_ids, and probabilities.
    """
    prompt = LOOK_AHEAD_PROMPT_TEMPLATE.format(
        question=question,
        current_answer=current_answer
    )
    
    # We call the generator's generate method directly.
    # The interface in QwenModel is generate(prompt, max_new_tokens=...)
    result = generator.generate(prompt, max_new_tokens=max_new_tokens)
    return result
