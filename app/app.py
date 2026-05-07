"""Streamlit demo for the Mediterranean Culinary RAG.

Run with:
    streamlit run app/app.py

Make sure you've already run `python scripts/build_index.py` once and that
GROQ_API_KEY is set in your environment.
"""
from __future__ import annotations

import os
import time

import streamlit as st

from rag_culinary.config import load_config
from rag_culinary.pipeline import RAGPipeline


# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mediterranean Culinary RAG",
    page_icon="🫒",
    layout="wide",
)


# ── Cached pipeline (loaded once per session) ───────────────────────────────

@st.cache_resource(show_spinner="Loading retriever and embedding model...")
def load_pipeline():
    return RAGPipeline.from_artifacts(_cfg)


# ── Header ──────────────────────────────────────────────────────────────────

st.title("🫒 Mediterranean Culinary RAG")
st.caption(
    "Ask anything about Mediterranean cuisine — Italian, Greek, Spanish, "
    "Moroccan, Lebanese, Turkish and more. Answers are grounded in a curated "
    "corpus of 161 cuisine documents using hybrid retrieval (BGE + BM25)."
)


# ── Load config first so we know which backend's key to check ──────────────

_cfg = load_config("config.yaml")

_BACKEND_INSTRUCTIONS = {
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "name": "Google Gemini",
        "url": "https://aistudio.google.com/apikey",
        "url_label": "aistudio.google.com/apikey",
        "key_prefix": "AIza...",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "name": "Groq",
        "url": "https://console.groq.com",
        "url_label": "console.groq.com",
        "key_prefix": "gsk_...",
    },
    "local": {
        "env_var": None,   # no key needed
        "name": "local Qwen",
        "url": None,
        "url_label": None,
        "key_prefix": None,
    },
}


def _check_api_key():
    """Block the app with a helpful error if the configured backend's key is missing.

    Resolution order: st.secrets (Streamlit Cloud) -> os.environ (local).
    Setting via st.secrets also exports it to os.environ so downstream code
    (the Generator classes) finds it without any changes.
    """
    info = _BACKEND_INSTRUCTIONS.get(_cfg.generation.backend)
    if info is None or info["env_var"] is None:
        return  # local backend or unknown — nothing to check

    env_var = info["env_var"]

    # If running on Streamlit Cloud, the key lives in st.secrets. Copy it to
    # os.environ so the Generator (which reads from os.environ) finds it.
    if env_var not in os.environ:
        try:
            if env_var in st.secrets:
                os.environ[env_var] = st.secrets[env_var]
        except (FileNotFoundError, AttributeError):
            # st.secrets isn't configured locally — that's fine, env var path handles it
            pass

    if not os.environ.get(env_var):
        st.error(
            f"**{env_var} not found.** This demo is configured to use "
            f"the {info['name']} API.\n\n"
            f"**Local development:**\n"
            f"1. Sign up at [{info['url_label']}]({info['url']}) (no credit card)\n"
            f"2. Create a key and copy it\n"
            f"3. Set it before launching: `export {env_var}={info['key_prefix']}`\n"
            f"4. Restart this app\n\n"
            f"**Streamlit Cloud:** add `{env_var}` to the app's Secrets in "
            f"`Settings -> Secrets`.\n\n"
            f"To switch backends, edit `backend:` in `config.yaml` "
            f"(options: gemini, groq, local)."
        )
        st.stop()


_check_api_key()


# ── Sidebar: configuration display ──────────────────────────────────────────

rag = load_pipeline()
cfg = rag.config

with st.sidebar:
    st.header("Configuration")
    st.markdown(f"""
    **Retrieval pipeline**
    - Chunking: `{cfg.chunking.strategy}` ({cfg.chunking.size} tok, {cfg.chunking.overlap} overlap)
    - Embeddings: `{cfg.embedding.model.split('/')[-1]}`
    - Retrieval: `{cfg.retrieval.strategy}` (α={cfg.retrieval.hybrid_alpha})
    - Top-K: `{cfg.retrieval.top_k}`

    **Generation**
    - Backend: `{cfg.generation.backend}`
    - Model: `{cfg.generation.model}`
    - Prompt: `{cfg.generation.prompt_style}`
    """)

    st.divider()
    st.header("Try an example")
    examples = [
        "What is hummus made from?",
        "Where did paella originate?",
        "How is Moroccan tagine traditionally cooked?",
        "What's the difference between Greek tzatziki and Lebanese labneh?",
        "Which herbs are typical in Provençal cuisine?",
    ]
    example_choice = st.radio(
        "Pick a question:",
        options=["—"] + examples,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "Built from a university coursework project — see "
        "[GitHub repo](https://github.com/) for the full pipeline."
    )


# ── Main interaction ────────────────────────────────────────────────────────

# Use the example as the default text if the user picked one
default_q = example_choice if example_choice != "—" else ""

question = st.text_input(
    "Ask a question:",
    value=default_q,
    placeholder="What is hummus made from?",
)

ask = st.button("Ask", type="primary", disabled=not question.strip())


if ask:
    # Retrieval phase
    with st.spinner("Retrieving relevant passages..."):
        t0 = time.time()
        retrieved = rag.retrieve(question)
        retrieval_ms = (time.time() - t0) * 1000

    # Generation phase
    with st.spinner(f"Generating answer with {cfg.generation.model}..."):
        t0 = time.time()
        try:
            answer = rag.generator.generate(question, retrieved)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()
        generation_ms = (time.time() - t0) * 1000

    # ── Layout: answer on the left, retrieved chunks on the right ──────────
    col_answer, col_chunks = st.columns([3, 2])

    with col_answer:
        st.subheader("Answer")
        st.markdown(answer)
        st.caption(
            f"Retrieval: {retrieval_ms:.0f} ms · "
            f"Generation: {generation_ms:.0f} ms"
        )

    with col_chunks:
        st.subheader(f"Retrieved chunks (top {len(retrieved)})")
        for r in retrieved:
            with st.expander(
                f"#{r.rank} · {r.chunk.source} · score {r.score:.3f}",
                expanded=(r.rank == 1),
            ):
                st.write(r.chunk.text)


# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Note: answers are grounded in the corpus only. If the model says "
    "\"I don't know\", the answer wasn't in the retrieved passages."
)
