# FLARE Reimplementation --- Step-by-Step Implementation Guide

## Goal

Reproduce the FLARE paper's experimental behavior and obtain numbers
reasonably close to the paper, while initially replacing the obsolete
`text-davinci-003` backend with a modern local causal LM.

Official sources:

-   FLARE repository: https://github.com/jzbjyb/FLARE
-   Paper: https://aclanthology.org/2023.emnlp-main.495/
-   2WikiMultihopQA: https://github.com/Alab-NII/2wikimultihop
-   Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
-   Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

The core rule is:

> Do not copy the original repository file-by-file. Read it as the
> specification, understand the execution path, and reimplement the
> required behavior while changing only the model backend.

------------------------------------------------------------------------

# 1. Current repository

Keep the current structure:

``` text
.
├── 2023.emnlp-main.495-flare.pdf
├── configs
├── data
├── notebooks
├── README.md
├── requirements.txt
├── results
├── scripts
├── src
│   └── flare
│       └── __init__.py
└── tests
```

Gradually add:

``` text
src/flare/
├── __init__.py
├── model.py
├── retriever.py
├── datasets.py
├── prompts.py
├── flare.py
└── evaluation.py

scripts/
├── build_index.py
├── test_model.py
├── test_retrieval.py
├── run_flare.py
└── evaluate.py

configs/
└── flare.yaml

docker/
└── elasticsearch/
    └── compose.yaml
```

Do not create empty files just for appearance. Create each as you reach
that phase.

------------------------------------------------------------------------

# 2. Architecture

Your final system should be:

``` text
2WikiMultihopQA
       |
       v
   prompt builder
       |
       v
 modern local causal LM
       |
       | generated text + token probabilities
       v
 FLARE confidence logic
       |
       +---- confident ------> accept prediction
       |
       +---- uncertain
                 |
                 v
        predicted future text
                 |
                 v
        Elasticsearch / BM25
                 |
                 v
          Wikipedia passages
                 |
                 v
             regenerate
                 |
                 v
              continue
```

The original FLARE implementation uses Elasticsearch/BM25 over the DPR
Wikipedia passage collection and `text-davinci-003` for generation. The
released `openai.sh` configures the 2WikiMultihopQA experiment with
Elasticsearch index `wikipedia_dpr`, 8 few-shot examples, 500 examples,
temperature 0, and maximum generation length 256.

------------------------------------------------------------------------

# 3. Important original files and what each teaches you

Read the original repository in this order.

## 3.1 `README.md`

Understand:

-   required data
-   Wikipedia corpus
-   Elasticsearch setup
-   datasets
-   original model
-   commands used to run experiments

Do not copy its environment blindly; it was written for the 2023
software stack.

## 3.2 `openai.sh`

This is extremely useful because it tells you the actual experiment
configuration.

For `2wikihop`, the released script uses approximately:

``` text
model              = text-davinci-003
temperature        = 0
search engine      = elasticsearch
index              = wikipedia_dpr
fewshot            = 8
max examples       = 500
max generation     = 256
```

Source:

https://github.com/jzbjyb/FLARE/blob/main/openai.sh

## 3.3 `configs/2wikihop_flare_config.json`

Read this carefully.

Extract the FLARE-specific parameters such as confidence thresholds,
look-ahead behavior, retrieval settings, and any prompt/configuration
parameters.

Do not guess them.

## 3.4 `src/openai_api.py`

This is the most important source file.

Read it deeply.

Find:

-   OpenAI generation calls
-   token log-probability handling
-   look-ahead generation
-   confidence calculation
-   low-confidence detection
-   retrieval query construction
-   retrieval calls
-   prompt/context construction
-   regeneration
-   generation loop termination

This file contains the core behavior you need to reproduce.

## 3.5 `src/retriever.py`

Understand:

``` text
query -> Elasticsearch -> document IDs + text
```

## 3.6 `prep.py`

Understand:

``` text
Wikipedia TSV -> Elasticsearch index
```

## 3.7 `src/datasets.py`

Find `WikiMultiHopQA`.

Understand:

-   dataset loading
-   few-shot examples
-   input prompt template
-   output template
-   answer normalization
-   evaluation

------------------------------------------------------------------------

# 4. Python environment

From the repository root:

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the modern equivalents of the old stack:

``` bash
pip install torch
pip install transformers
pip install accelerate
pip install datasets
pip install tqdm
pip install numpy
pip install pyyaml
pip install requests
pip install beir
pip install spacy
pip install elasticsearch
```

Install the English spaCy model:

``` bash
python -m spacy download en_core_web_sm
```

Check:

``` bash
python - <<'PY'
import torch
import transformers
import datasets
import beir
import spacy

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:",
          torch.cuda.get_device_properties(0).total_memory / 1024**3,
          "GB")
PY
```

------------------------------------------------------------------------

# 5. Model selection

First inspect your GPU:

``` bash
nvidia-smi
```

For the first local experiment, use a causal model that exposes logits.

Recommended:

``` text
Qwen/Qwen2.5-7B-Instruct
```

if your hardware can run it comfortably.

Fallback:

``` text
Qwen/Qwen2.5-3B-Instruct
```

The Hugging Face model cards provide direct Transformers loading
instructions.

Do not choose a random model merely because it is popular. FLARE
requires token-level confidence, so you need access to the model's
logits.

For confidence fidelity:

-   BF16/FP16 is preferable if hardware permits.
-   4-bit quantization is a fallback when memory requires it.
-   Record the exact model and precision in your experiment config.

------------------------------------------------------------------------

# 6. Isolate model caches

Do not let this experiment fill your normal Hugging Face cache.

Create:

``` bash
mkdir -p "$HOME/research/flare-storage/huggingface"
export FLARE_STORAGE="$HOME/research/flare-storage"
export HF_HOME="$FLARE_STORAGE/huggingface"
```

If you want this permanently for the project, put those exports in your
shell config.

The model files will then live under:

``` text
~/research/flare-storage/
```

When the project is over:

``` bash
rm -rf "$HOME/research/flare-storage"
```

------------------------------------------------------------------------

# 7. Phase 1 --- Implement the model backend

Create:

``` text
src/flare/model.py
```

Before doing anything with FLARE, make this work:

``` text
prompt
  |
  v
tokenizer
  |
  v
causal LM
  |
  +--> generated token IDs
  |
  +--> logits
          |
          v
       softmax
          |
          v
 token probability of selected token
```

Your wrapper conceptually needs to return:

``` python
{
    "text": "...",
    "tokens": [...],
    "token_ids": [...],
    "logprobs": [...]
}
```

Do not just call `model.generate()` and discard the scores.

The original OpenAI implementation gets token-level log probabilities
from the completion API. Your local implementation must provide the
equivalent information from model logits.

------------------------------------------------------------------------

# 8. Understand generation probabilities

For a generated token `t`:

``` text
logits
   |
   v
softmax(logits)
   |
   v
P(vocabulary token)
   |
   v
P(selected token)
```

Then:

``` python
token_logprob = torch.log(token_probability)
```

The important probability is the probability of the token that was
actually generated, not the maximum vocabulary probability.

FLARE uses this information for confidence decisions.

------------------------------------------------------------------------

# 9. Model test

Create:

``` text
scripts/test_model.py
```

Run one prompt and print:

``` text
PROMPT
...

GENERATED TEXT
...

TOKEN        PROBABILITY
------------------------
The          ...
capital      ...
of           ...
France       ...
...
```

Only proceed once you can obtain token probabilities.

------------------------------------------------------------------------

# 10. Phase 2 --- Wikipedia retrieval corpus

The original FLARE README uses the DPR Wikipedia passage collection:

``` text
psgs_w100.tsv.gz
```

Official download:

``` bash
mkdir -p data/dpr

wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Then:

``` bash
cd data/dpr
gzip -d psgs_w100.tsv.gz
cd ../..
```

You should have:

``` text
data/dpr/psgs_w100.tsv
```

Do not commit this file.

Your `.gitignore` should include:

``` gitignore
data/*
!data/.gitkeep
```

------------------------------------------------------------------------

# 11. Check disk space before downloading

Run:

``` bash
df -h
docker system df
```

The raw Wikipedia TSV plus Elasticsearch index can consume a lot of disk
space.

Your intended lifecycle is:

``` text
compressed TSV
      |
      v
decompress
      |
      v
build Elasticsearch index
      |
      v
test retrieval
      |
      v
delete raw TSV
```

Do not delete the raw TSV before verifying the index.

------------------------------------------------------------------------

# 12. Elasticsearch

You already have Elasticsearch running in Docker.

Verify:

``` bash
curl http://localhost:9200
```

Then:

``` bash
curl http://localhost:9200/_cat/indices?v
```

The target index is:

``` text
wikipedia_dpr
```

The original repository uses Elasticsearch 7.17.9.

Because your Elasticsearch is in Docker, your Python code should connect
to:

``` text
http://localhost:9200
```

rather than starting a local Elasticsearch process.

------------------------------------------------------------------------

# 13. Read `prep.py` before building the index

Open the original:

``` text
prep.py
```

Find the `build_elasticsearch` task.

Understand:

1.  How the TSV is read.
2.  Which columns are document ID/title/text.
3.  How the Elasticsearch mapping is created.
4.  How bulk insertion works.
5.  What index name is used.

The original command is:

``` bash
python prep.py     --task build_elasticsearch     --inp data/dpr/psgs_w100.tsv     wikipedia_dpr
```

Do not blindly run it because your Elasticsearch setup is different.

Instead, implement an equivalent local builder.

------------------------------------------------------------------------

# 14. Implement the index builder

Create:

``` text
scripts/build_index.py
```

It should:

``` text
read TSV
   |
   v
create `wikipedia_dpr`
   |
   v
bulk insert passages
   |
   v
refresh/check index
```

Keep the original document fields and BM25 behavior as closely as
possible.

Do not replace BM25 with:

-   embeddings
-   FAISS
-   Chroma
-   Qdrant
-   hybrid search
-   reranking

Those would change the experiment.

------------------------------------------------------------------------

# 15. Verify the Wikipedia index

Create:

``` text
scripts/test_retrieval.py
```

Try:

``` text
Who was the first president of the United States?
```

Expected behavior:

``` text
QUERY
...

RESULT 1
George Washington ...

RESULT 2
...

RESULT 3
...
```

Then inspect:

``` bash
curl http://localhost:9200/_cat/indices?v
```

Make sure `wikipedia_dpr` exists and contains a large number of
documents.

------------------------------------------------------------------------

# 16. Delete the raw Wikipedia file

After retrieval has been verified:

``` bash
rm data/dpr/psgs_w100.tsv
```

Now you have:

``` text
raw Wikipedia corpus: deleted
Elasticsearch index: retained inside Docker volume
```

This avoids keeping two copies.

------------------------------------------------------------------------

# 17. Phase 3 --- 2WikiMultihopQA

Do not implement all FLARE datasets yet.

Start with:

``` text
2WikiMultihopQA
```

The original FLARE script uses:

``` text
data/2wikimultihopqa
```

and runs 500 examples with eight few-shot examples.

The official dataset repository is:

https://github.com/Alab-NII/2wikimultihop

Put the required files under:

``` text
data/2wikimultihopqa/
```

The original FLARE dataset code expects files including the development
data and alias information.

------------------------------------------------------------------------

# 18. Read `src/datasets.py`

Find:

``` python
class WikiMultiHopQA
```

Understand:

``` text
raw JSON
   |
   v
question
   |
   +--> few-shot examples
   |
   v
prompt
   |
   v
model
```

Also understand how the expected answer is extracted and normalized.

The original implementation contains hard-coded demonstration examples.
Keep them aligned with the original for reproduction.

------------------------------------------------------------------------

# 19. Implement `datasets.py`

Create:

``` text
src/flare/datasets.py
```

Initially implement only:

``` text
WikiMultiHopQA
```

Return a simple structure such as:

``` python
{
    "id": ...,
    "question": ...,
    "answer": ...,
    "gold_output": ...,
    "ctxs": ...
}
```

Do not build support for StrategyQA, ASQA, WikiAsp, etc. yet.

------------------------------------------------------------------------

# 20. Phase 4 --- Prompts

Create:

``` text
src/flare/prompts.py
```

Reproduce the original prompt templates and few-shot formatting.

Do not improve the prompts.

For a reproduction project, prompt changes are experimental changes.

Keep:

-   wording
-   ordering
-   separators
-   few-shot examples
-   answer formatting
-   retrieval-context formatting

as close to the original as possible.

------------------------------------------------------------------------

# 21. Phase 5 --- Retriever

Read:

``` text
original src/retriever.py
```

The conceptual interface is:

``` python
documents = retriever.retrieve(query, top_k)
```

with results like:

``` python
[
    {
        "id": "...",
        "text": "..."
    },
    ...
]
```

Implement:

``` text
src/flare/retriever.py
```

Configuration:

``` yaml
elasticsearch:
  url: http://localhost:9200
  index: wikipedia_dpr
```

The rest of your code should not depend on Elasticsearch's raw response
format.

------------------------------------------------------------------------

# 22. Phase 6 --- Understand FLARE before implementing it

Now return to:

``` text
original src/openai_api.py
```

Trace the main generation path.

Answer these questions from the actual code:

1.  What is the initial prompt?
2.  How is the next content generated?
3.  How many tokens/sentences are looked ahead?
4.  Where are probabilities obtained?
5.  How is confidence computed?
6.  What threshold triggers retrieval?
7.  How is the uncertain region selected?
8.  How is the retrieval query created?
9.  How many documents are retrieved?
10. How are documents inserted into the prompt?
11. How is regeneration performed?
12. What text is accepted after regeneration?
13. How does the loop continue?
14. What stops generation?

Do not invent answers from the paper if the released implementation does
something more specific. For reproduction, the released code is the
implementation reference.

------------------------------------------------------------------------

# 23. FLARE control flow

Your final controller should implement this behavior:

``` text
current context
      |
      v
look-ahead generation
      |
      v
token probabilities
      |
      v
confidence check
      |
      +---------- confident ----------> accept
      |
      v
    uncertain
      |
      v
predicted future sentence
      |
      v
retrieval query
      |
      v
BM25 / Wikipedia
      |
      v
retrieved passages
      |
      v
regenerate
      |
      v
accept regenerated content
      |
      v
continue
```

The paper describes this forward-looking prediction +
low-confidence-triggered retrieval mechanism as the central FLARE idea.

------------------------------------------------------------------------

# 24. Implement `flare.py`

Create:

``` text
src/flare/flare.py
```

Conceptually:

``` python
class FLARE:
    def __init__(self, model, retriever, config):
        ...

    def generate(self, question):
        ...
```

Internally separate these responsibilities:

``` text
lookahead generation
confidence calculation
retrieval decision
query construction
retrieval
regeneration
loop control
```

The exact behavior must come from the original `openai_api.py`.

Do not merely implement the simplified pseudocode if it differs from the
original code.

------------------------------------------------------------------------

# 25. Confidence logic is critical

Do not replace the original confidence calculation with your own idea.

If the original implementation checks token probability against a
threshold, reproduce that.

If it aggregates probabilities over a sentence, reproduce that.

If it identifies sentence boundaries in a particular way, reproduce
that.

The local model may have very different probability calibration from
GPT-3, so the absolute confidence behavior may differ. Record the exact
model and thresholds.

------------------------------------------------------------------------

# 26. Query construction is also critical

Do not turn FLARE into:

``` python
query = original_question
```

unless the original implementation actually does so.

The key FLARE idea is that predicted future content provides the
retrieval query when the model is uncertain.

Your implementation should preserve:

``` text
predicted future content
        |
        v
retrieval query
```

as the original code does.

------------------------------------------------------------------------

# 27. Retrieval context must match the original

When documents are retrieved, reproduce as closely as possible:

-   top-k
-   document ordering
-   separators
-   labels
-   context placement
-   prompt wording

Do not add a reranker or summarizer.

------------------------------------------------------------------------

# 28. Generation settings

The original runner sets:

``` text
temperature = 0
```

Start your local model deterministically.

For Transformers, use deterministic generation rather than sampling.

Do not start with:

``` text
temperature = 0.7
top_p = 0.9
```

because then repeated runs can differ and your experiment becomes harder
to debug.

------------------------------------------------------------------------

# 29. Chat-model issue

`text-davinci-003` is a completion model.

A modern model such as Qwen2.5-7B-Instruct is an instruction/chat model.

Therefore you must explicitly decide how the original textual prompt is
presented to the new model.

Do not silently make a large prompt transformation.

Document:

``` text
original FLARE prompt
        |
        v
local model input formatting
```

If you use the model's chat template, record that fact in the experiment
configuration.

This is one legitimate source of difference from the original paper.

------------------------------------------------------------------------

# 30. Phase 7 --- Single-example runner

Create:

``` text
scripts/run_flare.py
```

It should initially support one question.

Print:

``` text
QUESTION
--------------------------------------------------

LOOKAHEAD
--------------------------------------------------

TOKEN CONFIDENCE
--------------------------------------------------

RETRIEVAL TRIGGER
--------------------------------------------------

RETRIEVAL QUERY
--------------------------------------------------

TOP-K DOCUMENTS
--------------------------------------------------

REGENERATION
--------------------------------------------------

FINAL ANSWER
--------------------------------------------------
```

This trace is extremely valuable.

You want to be able to see immediately if:

``` text
retrieval never happens
```

or:

``` text
retrieval happens every sentence
```

or:

``` text
retrieval query is nonsense
```

or:

``` text
documents are good but never enter the prompt
```

------------------------------------------------------------------------

# 31. Debug one question before running 500

Use one real 2WikiMultihopQA example.

Verify:

``` text
dataset loaded
      ↓
prompt correct
      ↓
model generates
      ↓
probabilities available
      ↓
confidence works
      ↓
retrieval works
      ↓
documents enter prompt
      ↓
regeneration works
      ↓
loop continues/stops correctly
```

Do not launch 500 examples before this works.

------------------------------------------------------------------------

# 32. Phase 8 --- Evaluation

Create:

``` text
src/flare/evaluation.py
scripts/evaluate.py
```

Use the same answer normalization/scoring logic as the original
implementation.

For 2WikiMultihopQA, understand the original `datasets.py` scoring code
before writing your evaluator.

Do not use an arbitrary LLM judge.

------------------------------------------------------------------------

# 33. Experiment configuration

Create:

``` text
configs/flare.yaml
```

Put all important parameters there:

``` yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  temperature: 0
  precision: ...

elasticsearch:
  url: http://localhost:9200
  index: wikipedia_dpr

dataset:
  name: 2wikihop
  path: data/2wikimultihopqa
  fewshot: 8
  max_examples: 500

generation:
  max_new_tokens: 256

flare:
  # Fill these from the original config/code.
  threshold: ...
  lookahead: ...
  top_k: ...
```

Do not guess the FLARE values. Extract them from the original
configuration/code.

------------------------------------------------------------------------

# 34. Phase 9 --- Run the 500-example benchmark

Once one example works:

``` bash
python scripts/run_flare.py     --config configs/flare.yaml
```

Save:

``` text
results/
├── predictions.jsonl
├── metrics.json
└── traces.jsonl
```

If traces are huge, you can later delete them after debugging.

------------------------------------------------------------------------

# 35. Baselines

Run these in order.

## Baseline A --- Local model, no retrieval

``` text
question -> local model -> answer
```

Purpose: measure model strength.

## Baseline B --- Standard RAG

``` text
question -> BM25 -> passages -> local model -> answer
```

Purpose: verify retrieval pipeline.

## Main --- FLARE

``` text
question -> FLARE -> local model + BM25 -> answer
```

Purpose: reproduce the method.

Only after these work should you spend time on another model.

------------------------------------------------------------------------

# 36. Do not immediately switch to text-davinci-003

Your original plan is:

``` text
local model first
      |
      +-- similar result -> done
      |
      +-- substantially worse -> investigate
```

That is sensible.

If local FLARE is substantially worse, check:

``` text
1. model capability
2. prompt formatting
3. chat-vs-completion formatting
4. confidence calculation
5. token probability extraction
6. threshold
7. retrieval quality
8. retrieved context formatting
9. generation length
10. evaluation
```

Only after those are correct should you consider reproducing with the
original OpenAI backend.

------------------------------------------------------------------------

# 37. WikiAsp comes later

Do not start with WikiAsp.

The original FLARE README says WikiAsp requires a Bing search API and a
local cached Bing search server.

That introduces:

``` text
Bing API key
+
external search
+
cache server
```

For your time constraint, use:

``` text
2WikiMultihopQA
```

first.

------------------------------------------------------------------------

# 38. Exact chronological implementation checklist

## Stage 1 --- Environment

``` text
[ ] create/activate venv
[ ] install PyTorch
[ ] install Transformers
[ ] install datasets
[ ] install BEIR
[ ] install spaCy
[ ] install Elasticsearch client
[ ] verify CUDA
```

## Stage 2 --- Local model

``` text
[ ] inspect openai_api.py model calls
[ ] implement model.py
[ ] load local model
[ ] generate text
[ ] expose token IDs
[ ] expose logits
[ ] calculate selected-token probabilities
[ ] test on one prompt
```

## Stage 3 --- Wikipedia

``` text
[ ] check disk space
[ ] download psgs_w100.tsv.gz
[ ] decompress
[ ] read prep.py
[ ] implement build_index.py
[ ] build wikipedia_dpr
[ ] test BM25
[ ] delete raw TSV
```

## Stage 4 --- Dataset

``` text
[ ] download 2WikiMultihopQA
[ ] place under data/2wikimultihopqa
[ ] read WikiMultiHopQA in datasets.py
[ ] implement datasets.py
[ ] verify one example
```

## Stage 5 --- Prompts

``` text
[ ] read original templates
[ ] reproduce few-shot examples
[ ] reproduce input formatting
[ ] reproduce output formatting
```

## Stage 6 --- Retriever

``` text
[ ] read original retriever.py
[ ] implement retriever.py
[ ] query wikipedia_dpr
[ ] return top-k text
```

## Stage 7 --- FLARE

``` text
[ ] read openai_api.py main control path
[ ] implement look-ahead generation
[ ] implement confidence logic
[ ] implement low-confidence detection
[ ] implement query construction
[ ] implement retrieval
[ ] implement retrieval-context formatting
[ ] implement regeneration
[ ] implement loop
```

## Stage 8 --- Debugging

``` text
[ ] run one example
[ ] print full trace
[ ] verify retrieval trigger
[ ] verify documents
[ ] verify regeneration
[ ] verify termination
```

## Stage 9 --- Benchmark

``` text
[ ] run 500 examples
[ ] save predictions
[ ] evaluate
[ ] compare to paper
```

## Stage 10 --- Only if needed

``` text
[ ] try another local model
[ ] investigate major result discrepancy
[ ] optionally test text-davinci-003
```

------------------------------------------------------------------------

# 39. Final repository structure

A finished version should look approximately like:

``` text
.
├── 2023.emnlp-main.495-flare.pdf
├── configs
│   └── flare.yaml
├── data
│   ├── 2wikimultihopqa
│   └── .gitkeep
├── docker
│   └── elasticsearch
│       └── compose.yaml
├── notebooks
├── results
│   ├── predictions.jsonl
│   └── metrics.json
├── scripts
│   ├── build_index.py
│   ├── evaluate.py
│   ├── run_flare.py
│   ├── test_model.py
│   └── test_retrieval.py
├── src
│   └── flare
│       ├── __init__.py
│       ├── datasets.py
│       ├── evaluation.py
│       ├── flare.py
│       ├── model.py
│       ├── prompts.py
│       └── retriever.py
├── tests
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# 40. Git commit sequence

Use commits that correspond to real milestones:

``` text
chore: set up Python environment

feat: add local causal model backend

feat: expose token log probabilities

feat: add Elasticsearch retriever

feat: add Wikipedia index builder

feat: add 2WikiMultihopQA loader

feat: reproduce FLARE prompts

feat: implement confidence detection

feat: implement FLARE retrieval loop

feat: add single-example debug runner

feat: add 2WikiMultihopQA evaluation

feat: run local FLARE benchmark
```

------------------------------------------------------------------------

# 41. Daily Docker workflow

You already have Elasticsearch in Docker.

Start work:

``` bash
cd /path/to/flare-reimplementation
source .venv/bin/activate

docker compose -f docker/elasticsearch/compose.yaml start
```

Verify:

``` bash
curl http://localhost:9200
```

Work normally.

At the end of the day:

``` bash
docker compose -f docker/elasticsearch/compose.yaml stop
```

The index remains.

Next day:

``` bash
docker compose -f docker/elasticsearch/compose.yaml start
```

Do NOT use `down -v` every day.

------------------------------------------------------------------------

# 42. Final cleanup

When the paper is completely finished:

``` bash
docker compose -f docker/elasticsearch/compose.yaml down -v
```

This removes:

``` text
Elasticsearch container
Elasticsearch volume
Wikipedia BM25 index
```

Then check:

``` bash
docker volume ls
docker system df
```

Delete the project-specific model cache:

``` bash
rm -rf "$HOME/research/flare-storage"
```

Delete any remaining temporary dataset files if desired.

Keep:

``` text
source code
configs
README
paper
final results
```

------------------------------------------------------------------------

# 43. What NOT to spend time on

Do not build:

``` text
vector database
embedding retriever
reranker
web UI
FastAPI service
distributed inference
training
fine-tuning
generic plugin architecture
production RAG framework
experiment dashboard
```

None of these are required for reproducing this paper.

------------------------------------------------------------------------

# 44. The single most important implementation principle

You want to transform:

``` text
ORIGINAL

text-davinci-003
      |
      v
token probabilities
      |
      v
FLARE algorithm
      |
      v
Elasticsearch/BM25
```

into:

``` text
YOUR VERSION

modern local causal LM
      |
      v
token probabilities from logits
      |
      v
SAME FLARE algorithm
      |
      v
SAME Elasticsearch/BM25
```

Do not simultaneously change:

``` text
model
retriever
prompt
confidence metric
dataset
evaluation
```

If all six change and the number is different, you cannot diagnose the
reason.

Change the model backend first and preserve everything else as closely
as practical.

------------------------------------------------------------------------

# 45. Your immediate next action

Do not download Wikipedia yet.

First:

``` bash
nvidia-smi
```

and determine your GPU/VRAM.

Then implement:

``` text
src/flare/model.py
```

while reading the model-related section of:

``` text
original FLARE/src/openai_api.py
```

The first concrete milestone is:

``` text
Qwen/local causal LM
        |
        v
generated tokens
        |
        v
token probabilities/log-probabilities
```

Once that works, proceed to the Wikipedia index.

This order minimizes the chance of spending hours downloading and
indexing a huge corpus before discovering that the local model backend
cannot provide the information FLARE needs.
