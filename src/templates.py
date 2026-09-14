"""
Prompt templates for FLARE model.
These templates format the instruction for the Qwen instruction-tuned model.
"""

QA_PROMPT_TEMPLATE = """You are an expert Question Answering system. Answer the following question accurately and concisely based on the context. If the context is empty, try to answer based on your internal knowledge.

Context:
{context}

Question:
{question}

Answer:"""

LOOK_AHEAD_PROMPT_TEMPLATE = """You are an expert Question Answering system. We are trying to answer a question. Continue the following answer naturally.

Question:
{question}

Current Answer:
{current_answer}"""
