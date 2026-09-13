# FLARE Team Interface & Development Agreement

**Project:** Modern FLARE — FLARE with Qwen3-4B-Instruct-2507  
**Original reference:** `jzbjyb/FLARE`  
**Purpose:** Define stable interfaces, ownership boundaries, integration rules, and AI-agent coding rules for the three-person team.

---

## 1. Project Goal

We are implementing the FLARE (Forward-Looking Active Retrieval Augmented Generation) algorithm while replacing the original `text-davinci-003` generator with:

```text
Qwen/Qwen3-4B-Instruct-2507
```

The following original FLARE components should remain conceptually unchanged unless explicitly agreed:

- FLARE look-ahead generation
- confidence-based retrieval triggering
- retrieval/regeneration mechanism
- BM25 retrieval
- DPR Wikipedia corpus
- 2WikiMultihopQA evaluation
- original FLARE configuration parameters, unless an experiment explicitly changes them

The model backend is the intentional modernization.

---

# 2. Team Ownership

## Person 1 — Retrieval

Owns:

```text
prep.py
src/retriever.py
src/datasets.py
tests/test_retriever.py
```

Responsibilities:

- DPR Wikipedia corpus handling
- Elasticsearch indexing
- Elasticsearch communication
- BM25 retrieval
- dataset loading
- retrieval-related tests

Person 1 must not change the generator or FLARE algorithm interfaces.

---

## Person 2 — Qwen Model Backend

Owns:

```text
src/models/qwen.py
tests/test_qwen.py
```

May modify:

```text
src/utils.py
```

only when required for model/backend functionality.

Responsibilities:

- Qwen model loading
- tokenizer
- generation
- token IDs
- generated tokens
- token probabilities
- offsets/position information
- GPU/quantization configuration
- model-specific tests

Person 2 must not implement FLARE retrieval decisions.

---

## Person 3 — FLARE Algorithm / Integration

Owns:

```text
src/flare/
├── agent.py
├── confidence.py
└── lookahead.py

src/templates.py
tests/test_flare.py
qwen.sh
```

Person 3 is responsible for:

- FLARE algorithm
- look-ahead generation
- confidence calculation/interpretation
- retrieval triggering
- regeneration
- context integration
- overall orchestration
- integration of Person 1 and Person 2 components
- experiments and benchmarking

The original `src/openai_api.py` is treated as the **reference implementation** for understanding `QueryAgent`.

Do not directly rewrite the original implementation without first identifying which parts are FLARE logic and which parts are OpenAI-specific.

---

# 3. High-Level Architecture

The system must maintain this dependency direction:

```text
                    FLARE Agent
                        |
             +----------+----------+
             |                     |
             v                     v
         Generator             Retriever
             |                     |
             v                     v
           Qwen              Elasticsearch
```

The FLARE layer may call the Generator and Retriever.

The Generator must NOT call the Retriever.

The Retriever must NOT call the Generator.

This prevents circular dependencies and keeps each component independently testable.

---

# 4. Generator Interface

File:

```text
src/models/qwen.py
```

The FLARE system must interact with Qwen through a model abstraction.

The FLARE code must NOT directly call:

```python
AutoModelForCausalLM.from_pretrained(...)
```

or:

```python
model.generate(...)
```

throughout the FLARE logic.

Those details belong inside the Qwen backend.

---

## 4.1 Required generator operation

Conceptually:

```python
result = generator.generate(...)
```

The exact class/function name may be changed by Person 2, but the interface semantics must remain stable.

The generator must provide enough information to construct:

```text
GeneratorResult
```

containing at least:

```text
text
token_ids
tokens
probabilities
```

and, where required:

```text
offsets
finish_reason
```

---

## 4.2 GeneratorResult contract

Conceptually:

```python
GeneratorResult(
    text: str,
    token_ids: list[int],
    tokens: list[str],
    probabilities: list[float],
    offsets: optional,
    finish_reason: optional
)
```

Requirements:

```text
len(token_ids)
    ==
len(tokens)
    ==
len(probabilities)
```

for generated tokens.

Each:

```text
probabilities[i]
```

must correspond to:

```text
token_ids[i]
```

and:

```text
tokens[i]
```

---

## 4.3 Probability semantics

`probabilities[i]` means:

```text
P(token_i | preceding generated/input tokens, model context)
```

It must represent the probability assigned by Qwen to the token that was actually generated.

The FLARE layer must not depend on the entire vocabulary probability distribution unless explicitly required.

The model layer is responsible for converting logits into the selected-token probabilities.

---

# 5. Retriever Interface

File:

```text
src/retriever.py
```

The FLARE layer must interact with retrieval through a stable retriever interface.

Conceptually:

```python
results = retriever.retrieve(
    queries,
    topk=2,
    max_query_length=None
)
```

The exact implementation may use Elasticsearch/BEIR internally.

The FLARE layer must not directly construct Elasticsearch queries.

---

## 5.1 RetrievalResult contract

Each retrieval result must contain enough information to identify:

```text
document ID
document text
retrieval score
```

Conceptually:

```python
RetrievalResult(
    doc_id: str,
    text: str,
    score: float
)
```

A retrieval call should return results grouped by query.

Conceptually:

```python
[
    [RetrievalResult(...), RetrievalResult(...)],
    ...
]
```

---

# 6. Dataset Interface

Dataset loading belongs to Person 1.

The FLARE algorithm must not contain dataset-specific loading logic.

The dataset layer should expose examples containing the information required by the evaluation pipeline.

At minimum:

```text
question
answer
```

and, when available:

```text
gold contexts
chain-of-thought / supporting reasoning
example metadata
```

Dataset-specific parsing must remain inside:

```text
src/datasets.py
```

---

# 7. FLARE Interface

The FLARE controller belongs to Person 3.

Conceptually:

```python
agent = FLAREAgent(
    generator=generator,
    retriever=retriever,
    config=config
)

result = agent.generate(question)
```

The FLARE agent is responsible for deciding:

```text
generate
    ↓
look ahead
    ↓
evaluate confidence
    ↓
retrieve?
    ├── NO → continue generation
    │
    └── YES
          ↓
       retrieve
          ↓
       regenerate
          ↓
       continue
```

The FLARE layer must not contain model-loading code.

---

# 8. FLARE Result

The final result should contain more than the final answer.

Conceptually:

```python
FLAREResult(
    text=...,
    retrieval_history=...,
    generation_history=...,
    confidence_history=...
)
```

This is required for debugging and benchmarking.

The implementation should allow us to answer:

- What did the model generate?
- When did FLARE trigger retrieval?
- Why did retrieval trigger?
- What query was sent to Elasticsearch?
- What documents were retrieved?
- What confidence values were observed?
- What text was regenerated?

---

# 9. Configuration Rules

Original configuration:

```text
configs/2wikihop_flare_config.json
```

must be treated as the baseline reference.

Create a modernized configuration separately, e.g.:

```text
configs/2wikihop_qwen_flare_config.json
```

Do not silently modify the original configuration.

The Qwen experiment should initially preserve the original FLARE parameters as much as practical.

Any parameter changed for an experiment must be documented.

Example:

```text
Original:
look_ahead_filter_prob = 0.8

Experiment:
look_ahead_filter_prob = 0.9

Reason:
threshold sensitivity experiment
```

---

# 10. Model Configuration

Qwen-specific settings belong separately from FLARE algorithm settings.

Example:

```text
configs/models/qwen3-4b.json
```

Qwen-specific configuration may include:

```text
model_name
quantization
dtype
max_new_tokens
generation settings
device settings
```

FLARE-specific configuration must not be mixed with model-loading implementation details.

---

# 11. Original OpenAI Code

The original files:

```text
src/openai_api.py
openai.sh
```

are reference material for the original FLARE implementation.

Do not delete them during the initial implementation.

The original implementation should be used to understand:

```text
QueryAgent
ApiReturn
CtxPrompt
look-ahead logic
confidence filtering
retrieval triggering
```

The Qwen implementation must replace the OpenAI backend, not accidentally change the FLARE algorithm.

Once the modern implementation is stable, the team may decide whether the old OpenAI-specific files should remain.

---

# 12. Qwen vs OpenAI Replacement Boundary

The intentional replacement is:

```text
ORIGINAL

QueryAgent
    ↓
OpenAI API
    ↓
text-davinci-003
```

becomes:

```text
MODERN

FLAREAgent
    ↓
Generator interface
    ↓
QwenModel
    ↓
Qwen3-4B-Instruct-2507
```

The following OpenAI-specific functionality must NOT leak into the FLARE layer:

```text
OpenAI API keys
OpenAI API calls
OpenAI SDK objects
OpenAI-specific tokenization
OpenAI-specific response parsing
```

---

# 13. Tokenization Rule

The Qwen tokenizer must be the single source of truth for Qwen tokenization.

Do not use:

```text
tiktoken
GPT token IDs
GPT token offsets
```

for Qwen-generated text.

The FLARE confidence mechanism must consume the probabilities produced by the Qwen backend.

---

# 14. Mocking Requirement

All three components must be testable without requiring the full system.

## Generator mock

Person 2 must provide or enable a mock generator capable of returning deterministic:

```text
text
tokens
token IDs
probabilities
```

---

## Retriever mock

Person 1 must allow retrieval tests without requiring the complete Wikipedia index where practical.

---

## FLARE mock test

Person 3 must test at least:

```text
HIGH CONFIDENCE
    ↓
no retrieval

LOW CONFIDENCE
    ↓
retrieval triggered

LOW CONFIDENCE
    ↓
retrieval
    ↓
regeneration
```

These tests must not require Qwen to be loaded.

---

# 15. GPU Rule

Only Person 3's machine currently has the GPU required for the real Qwen execution.

Therefore:

```text
Person 1
    ↓
CPU development + tests

Person 2
    ↓
CPU/mock development + tests

Person 3
    ↓
integration + actual Qwen execution + benchmarks
```

Person 1 and Person 2 must not make the project dependent on access to the GPU for ordinary unit tests.

---

# 16. Dataset / Model / Elasticsearch Data

Large files must NOT be committed to Git.

Do not commit:

```text
*.tsv
*.tsv.gz
model weights
Hugging Face cache
Elasticsearch data
benchmark outputs
large logs
```

Use `.gitignore`.

Each developer should keep local data outside the tracked source tree where practical.

---

# 17. Git Ownership Rules

Each developer works primarily on their own branch.

Suggested branches:

```text
person1-retrieval
person2-qwen
person3-flare
```

Do not directly push another person's branch.

Before changing a file owned by another person:

1. Notify the owner.
2. Explain why the change is required.
3. Agree on the change.
4. Make the change in coordination with the owner.

---

# 18. Shared File Rule

The following files are considered high-conflict:

```text
src/templates.py
src/utils.py
configs/*.json
README.md
pyproject.toml
requirements files
```

AI coding agents are NOT allowed to modify these files casually.

Before an AI agent changes a shared file, the developer must verify:

```text
1. Why is this change necessary?
2. Does it violate another component's interface?
3. Can the change be isolated to the developer's own module?
4. Will another branch be broken?
```

Prefer adding a new adapter/module over modifying a shared file.

---

# 19. AI Coding Agent Rules

AI agents must follow these rules.

### Rule 1 — Read before modifying

An AI agent must inspect the existing implementation before modifying it.

It must not assume that a file works the way its filename suggests.

---

### Rule 2 — Do not change public interfaces silently

An agent must not change:

```text
function names
argument names
return structures
class interfaces
configuration keys
```

if those interfaces are already part of this agreement.

---

### Rule 3 — Do not refactor unrelated code

If asked to implement retrieval:

```text
DO:
    modify retrieval code

DO NOT:
    refactor Qwen
    rewrite FLARE
    change configuration
    rename unrelated modules
```

---

### Rule 4 — Do not install dependencies casually

AI agents must not add dependencies without documenting:

```text
package
version/range
reason
which module requires it
```

---

### Rule 5 — Do not replace working implementations unnecessarily

If an existing implementation works, do not rewrite it merely for style.

---

### Rule 6 — Do not alter experimental parameters silently

Changing FLARE parameters changes the experiment.

Any change must be explicit and documented.

---

### Rule 7 — Tests are mandatory

Every new component must include tests for its public interface.

---

### Rule 8 — Preserve the original implementation

The original FLARE implementation is our reference.

When modifying behavior, the agent should identify:

```text
Original behavior:
...

New behavior:
...

Reason:
...
```

---

# 20. Integration Contract

A component is considered ready for integration only when:

```text
[ ] Public interface documented
[ ] Unit tests included
[ ] No unrelated files modified
[ ] No hidden dependencies
[ ] No hard-coded local paths
[ ] No GPU requirement unless explicitly documented
[ ] Existing tests still pass
```

---

# 21. Definition of Done

## Person 1

Done when:

```text
DPR Wikipedia
    ↓
Elasticsearch
    ↓
BM25
    ↓
RetrieverResult
```

works independently and is tested.

---

## Person 2

Done when:

```text
Qwen
    ↓
GeneratorResult
    ↓
tokens + token IDs + probabilities
```

works independently and is tested.

The real model must work on the integration machine.

---

## Person 3

Done when:

```text
Generator
     +
Retriever
     +
FLARE controller
```

produces:

```text
final answer
+
retrieval history
+
confidence information
+
generation history
```

and passes deterministic mock tests.

---

# 22. Integration Sequence

Integration happens in this order:

```text
1. Person 1 completes Retriever interface
                ↓
2. Person 2 completes Generator interface
                ↓
3. Person 3 integrates both
                ↓
4. Mock FLARE tests
                ↓
5. Real Elasticsearch
                ↓
6. Real Qwen
                ↓
7. One-example debugging
                ↓
8. Small dataset experiment
                ↓
9. Full benchmark
```

Do not skip directly to full benchmarking.

---

# 23. Experimental Baselines

The final system should distinguish at least:

```text
Baseline 1:
Qwen without retrieval

Baseline 2:
Qwen + standard one-shot RAG

System:
Qwen + FLARE
```

The purpose is to determine whether FLARE itself provides a benefit.

---

# 24. Scientific Reproducibility Rule

When reporting results, explicitly state:

```text
Original FLARE:
    Generator = text-davinci-003

Our implementation:
    Generator = Qwen3-4B-Instruct-2507
```

We are reproducing the **FLARE methodology/algorithm**, not claiming numerical reproduction of the original paper's results.

Any deviation from the original paper/repository must be documented.

---

# 25. Conflict Resolution

If two branches require incompatible changes:

```text
1. Do not force-merge.
2. Identify the interface causing the conflict.
3. Discuss the desired contract.
4. Prefer the smallest interface change.
5. Update this document if the contract changes.
6. Update tests.
7. Then merge.
```

The interface document is the source of truth.

Code must conform to the interface, not the other way around.

---

# 26. Golden Rule

> **A developer owns implementation; the team owns interfaces.**

No individual developer or AI coding agent may unilaterally change an agreed interface.

If an interface genuinely needs to change, change the agreement first, then change the code.

---

# 27. Current Component Map

```text
                         FLARE
                           │
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
               GENERATOR      RETRIEVER
                    │             │
                    │             │
                  Qwen        Elasticsearch
                    │             │
                    ▼             ▼
             GeneratorResult  RetrievalResult
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                     FLARE Agent
                           │
                           ▼
                       FLAREResult
```

This architecture is the team's shared contract.