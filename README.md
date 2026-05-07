# Mediterranean Culinary RAG

A retrieval-augmented generation system specialised in Mediterranean cuisine. Given a question, it retrieves relevant passages from a curated corpus of 161 cuisine documents and generates a grounded answer.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

Originally developed as university coursework (COMP64702 — Transforming Text Into Meaning) and restructured here as a deployable Python package with a public web demo.

## Architecture

```
question → [Hybrid Retrieval: BGE dense + BM25] → top-5 chunks → [LLM + grounded prompt] → answer
```

| Component | Choice | Why |
|---|---|---|
| Chunking | Overlapping (200 tokens, 50 overlap) | Best Hit@5 — handles answers near chunk boundaries |
| Embeddings | `BAAI/bge-small-en-v1.5` | Best Hit@1 / MRR among models tested; retrieval-trained |
| Retrieval | Hybrid dense+BM25 (α=0.6) | Beats either method alone across 6 alpha settings |
| Generation | Gemini 2.5 Flash (default) / Llama 3.3 70B / Qwen 0.5B | Configurable; original constraint was Qwen 0.5B |
| Prompt | Instructed (grounded) | Highest ROUGE-L; reduces hallucination |

## Results on the 748-question benchmark

Retrieval (hybrid BGE+BM25 vs dense-only):

| Metric | Hybrid | Dense |
|---|---|---|
| Hit@1 | 0.92 | 0.86 |
| Hit@5 | 0.98 | 0.96 |
| MRR | 0.95 | 0.91 |

Generation (with the same retrieved context, varying the LLM):

| Model | ROUGE-L | BERTScore F1 |
|---|---|---|
| Qwen 2.5-0.5B (coursework) | 0.39 | 0.91 |
| Gemini 2.5 Flash (demo) | _fill in after eval_ | _fill in_ |

## Quick start (local)

```bash
# Install just the slim runtime (no torch/transformers — uses Gemini API)
pip install -e .

# Get a free Gemini key at aistudio.google.com/apikey
export GEMINI_API_KEY=AIza_your_key_here

# Build the index once (uses raw corpus)
python scripts/build_index.py

# Run the Streamlit demo
streamlit run app/app.py
```

## Deploying to a public URL

See [DEPLOY.md](DEPLOY.md) for a 15-minute walkthrough of putting this on Streamlit Cloud.

## Install variants

The package has optional dependency groups so you don't pay for what you don't use:

```bash
# Slim demo runtime — Gemini/Groq backends, no PyTorch
pip install -e .

# + the heavy ML stack — needed for offline benchmark evaluation
# (BERTScore) and the local Qwen backend
pip install -e ".[eval]"

# + dev tools (pytest, ruff)
pip install -e ".[dev]"
```

## Configuration

All knobs live in `config.yaml`. Switch a strategy by editing one line — no code changes:

```yaml
chunking:
  strategy: overlapping  # fixed_200 | fixed_400 | sentence | overlapping

retrieval:
  strategy: hybrid       # dense | bm25 | hybrid
  hybrid_alpha: 0.6

generation:
  backend: gemini        # gemini | groq | local
  model: gemini-2.5-flash
  prompt_style: instructed  # basic | instructed | role_based | cot
```

## Running the benchmark

```bash
# Retrieval only — fast, runs in seconds
python scripts/evaluate.py

# With generation — slower, calls the LLM 748 times
pip install -e ".[eval]"   # one-off, for the BERTScore dependency
python scripts/evaluate.py --generation
```

The script reports retrieval metrics in both **lenient** mode (counts a hit if any chunk from the gold source file is retrieved) and **strict** mode (gold answer must actually appear in the retrieved chunk text). The lenient version saturates near 1.0 because the corpus has roughly one chunk per document; the strict numbers are the more meaningful comparison across configurations.

## Project layout

```
src/rag_culinary/      # importable package
  config.py            # YAML config loader
  corpus.py            # load corpus + benchmark
  chunking.py          # 4 strategies (fixed/sentence/overlapping)
  embedding.py         # SentenceTransformers wrapper
  retrieval.py         # Dense / BM25 / Hybrid retrievers
  generation.py        # 3 LLM backends + 4 prompt strategies
  evaluation.py        # Hit@K, MRR, ROUGE-L, BERTScore (lazy-loaded)
  pipeline.py          # end-to-end RAGPipeline
  io_utils.py          # JSON I/O + format parsing

app/app.py             # Streamlit demo
scripts/               # CLI entry points (build_index, evaluate, run_inference)
tests/                 # 37 tests, no GPU/network needed
artifacts/             # prebuilt chunks + embeddings (committed to git)
config.yaml            # all tuneable parameters
DEPLOY.md              # cloud deployment guide
```

## License

MIT.
