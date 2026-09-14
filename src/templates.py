"""
Prompt templates for the FLARE reimplementation with Qwen.

Reproduces the original FLARE prompt style for 2WikiMultihopQA:
- 8-shot in-context examples
- Answer continuation format (completion-style, not chat-style)
- Retrieval context is injected as Wikipedia passages before the answer

The original FLARE system used text-davinci-003 (a completion model). Since
Qwen3-4B-Instruct-2507 is a chat/instruction model, we wrap the prompt in a
single user message using apply_chat_template. The few-shot examples are kept
verbatim. This is the one documented deviation from the original.
"""

# ---------------------------------------------------------------------------
# 8-shot few-shot examples from 2WikiMultihopQA
# Taken directly from the original FLARE src/templates.py / datasets.py
# ---------------------------------------------------------------------------
_FEWSHOT_EXAMPLES = [
    {
        "question": "Were Pavel Urysohn and Leonid Levin both Russian?",
        "answer": "Pavel Urysohn was a Soviet mathematician. Leonid Levin is a Soviet-American mathematician and computer scientist. So the answer is yes.",
    },
    {
        "question": "What nationality was James Henry Miller's wife?",
        "answer": "James Henry Miller's wife was Frances Eleanor Dona Alba. Frances Eleanor Dona Alba was American. So the answer is American.",
    },
    {
        "question": "Do the Silk Road and the Inca Road have the same length?",
        "answer": "The Silk Road is about 6,437 km long. The Inca Road is about 40,000 km long. So the answer is no.",
    },
    {
        "question": "Are both the Quina and the Jucar rivers in Spain?",
        "answer": "The Quina river is located in Galicia, Spain. The Jucar river is located in Spain. So the answer is yes.",
    },
    {
        "question": "Which country does the film 'Serpent Island' (1954 film) take place in?",
        "answer": "The 1954 film Serpent Island is set in Haiti. So the answer is Haiti.",
    },
    {
        "question": "Which film has the director born first, Beautiful Blue Eyes or Sherlock Holmes?",
        "answer": "The director of Beautiful Blue Eyes is Joshua Newton, born in 1952. The director of Sherlock Holmes (2009) is Guy Ritchie, born in 1968. Joshua Newton (1952) was born before Guy Ritchie (1968). So the answer is Beautiful Blue Eyes.",
    },
    {
        "question": "Is it true that the Eiffel Tower is taller than the Statue of Liberty?",
        "answer": "The Eiffel Tower is 330 metres tall. The Statue of Liberty is 93 metres tall. So the answer is yes.",
    },
    {
        "question": "Did Nikola Tesla die in the same city in which Arthur Conan Doyle was born?",
        "answer": "Nikola Tesla died in New York City, United States. Arthur Conan Doyle was born in Edinburgh, Scotland. So the answer is no.",
    },
]


def _format_fewshot() -> str:
    lines = []
    for ex in _FEWSHOT_EXAMPLES:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"A: {ex['answer']}")
        lines.append("")
    return "\n".join(lines)


_FEWSHOT_BLOCK = _format_fewshot()


def _format_contexts(contexts: list[str]) -> str:
    """Format retrieved Wikipedia passages the same way FLARE does."""
    parts = []
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"Wikipedia passage {i}: {ctx}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_prompt(question: str, current_answer: str) -> str:
    """
    Plain look-ahead prompt (no retrieved context).
    The model is asked to continue the answer from where it left off.
    """
    suffix = f"A: {current_answer}" if current_answer else "A:"
    return f"{_FEWSHOT_BLOCK}Q: {question}\n{suffix}"


def build_retrieval_prompt(
    question: str,
    current_answer: str,
    contexts: list[str],
) -> str:
    """
    Regeneration prompt with retrieved Wikipedia passages injected.
    Matches the original FLARE CtxPrompt format.
    """
    ctx_block = _format_contexts(contexts)
    suffix = f"A: {current_answer}" if current_answer else "A:"
    return (
        f"{_FEWSHOT_BLOCK}"
        f"{ctx_block}\n\n"
        f"Q: {question}\n"
        f"{suffix}"
    )
