"""Generation: prompt strategies + three backend implementations.

Three generator backends share the same interface:

  - LocalQwenGenerator: runs Qwen 2.5-0.5B locally via transformers. Slow on CPU,
    quality limited by model size (the coursework constraint). Kept as a baseline.

  - GroqGenerator: calls Groq's free OpenAI-compatible API. Default model is
    Llama 3.3 70B. Free tier: 30 req/min, no card required.
    Get a key at https://console.groq.com and set:
        export GROQ_API_KEY=gsk_...

  - GeminiGenerator: calls Google's free Gemini API. Default model is
    Gemini 2.5 Flash. Free tier: 1500 req/day, no card required.
    Get a key at https://aistudio.google.com/apikey and set:
        export GEMINI_API_KEY=AIza...

The factory `build_generator(cfg)` picks the right one based on
`cfg.generation.backend` ("local", "groq", or "gemini").
"""
from __future__ import annotations

import os
from typing import Callable, Protocol

from rag_culinary.config import GenerationConfig
from rag_culinary.retrieval import RetrievalResult


# ── Prompt strategies (shared across backends) ──────────────────────────────

PromptFn = Callable[[str, str], str]   # (question, context) -> prompt text


def _basic(question: str, context: str) -> str:
    return (
        f"Use the context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def _instructed(question: str, context: str) -> str:
    return (
        f"Answer only from the context provided. "
        f"If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def _role_based(cuisine: str) -> PromptFn:
    def inner(question: str, context: str) -> str:
        return (
            f"You are an expert culinary assistant specialising in {cuisine} cuisine. "
            f"Use only the context provided to answer accurately and concisely.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    return inner


def _cot(question: str, context: str) -> str:
    return (
        f"Think step by step. First identify what the question is asking, "
        f"then find the relevant information in the context, "
        f"then give a clear concise answer. Use only the context provided. "
        f"If the answer is not present, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nStep-by-step answer:"
    )


def get_prompt_fn(style: str, cuisine: str = "Mediterranean") -> PromptFn:
    if style == "basic":      return _basic
    if style == "instructed": return _instructed
    if style == "role_based": return _role_based(cuisine)
    if style == "cot":        return _cot
    raise ValueError(
        f"Unknown prompt style: {style!r}. "
        f"Valid: basic, instructed, role_based, cot"
    )


# ── Context formatting (shared) ──────────────────────────────────────────────

def format_context(retrieved: list[RetrievalResult]) -> str:
    """Concatenate retrieved chunks into a labelled context string."""
    return "\n\n".join(
        f"[Chunk {r.rank} | Source: {r.chunk.source}]\n{r.chunk.text.strip()}"
        for r in retrieved
    )


# ── Common interface ────────────────────────────────────────────────────────

class Generator(Protocol):
    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str: ...


# ── Local backend (Qwen via transformers) ────────────────────────────────────

class LocalQwenGenerator:
    """Loads the LLM once and exposes a generate() call.

    Slow on CPU. Known to segfault on some macOS / transformers 5.x combinations
    during weight materialisation — use the Groq backend instead in that case.
    """

    def __init__(self, cfg: GenerationConfig, cuisine: str = "Mediterranean"):
        # Heavy imports kept inside __init__ so this module loads fast for
        # users who only want the Groq backend.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.cuisine = cuisine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        self._torch = torch
        self.prompt_fn = get_prompt_fn(cfg.prompt_style, cuisine)

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = format_context(retrieved)
        prompt = self.prompt_fn(question, context)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Groq backend (OpenAI-compatible API, free tier) ─────────────────────────

class GroqGenerator:
    """Calls the Groq API via the OpenAI-compatible endpoint.

    Free tier: 30 req/min on Llama 3.3 70B. Get a key (no credit card) at
    https://console.groq.com and set GROQ_API_KEY in your environment.
    """

    def __init__(self, cfg: GenerationConfig, cuisine: str = "Mediterranean"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Groq backend requires the openai package. "
                "Install with: pip install openai"
            ) from e

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set.\n"
                "Get a free key at https://console.groq.com (no credit card), "
                "then run:\n    export GROQ_API_KEY=gsk_..."
            )

        self.cfg = cfg
        self.cuisine = cuisine
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.prompt_fn = get_prompt_fn(cfg.prompt_style, cuisine)

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = format_context(retrieved)
        prompt = self.prompt_fn(question, context)

        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.cfg.max_new_tokens,
            temperature=0.0 if not self.cfg.do_sample else 0.7,
        )
        return (response.choices[0].message.content or "").strip()


# ── Gemini backend (Google AI Studio, free tier) ────────────────────────────

class GeminiGenerator:
    """Calls Google's Gemini API via the official google-genai SDK.

    Free tier: 1500 requests/day on Gemini 2.5 Flash, no credit card. Get a key
    at https://aistudio.google.com/apikey and set GEMINI_API_KEY in your
    environment.

    Note: the legacy `google-generativeai` package conflicts with the current
    `google-genai` package. If you see `ModuleNotFoundError: No module named
    'google.genai'`, run `pip uninstall google-generativeai` first.
    """

    def __init__(self, cfg: GenerationConfig, cuisine: str = "Mediterranean"):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Gemini backend requires the google-genai package. "
                "Install with: pip install google-genai\n"
                "(If you previously installed google-generativeai, uninstall it first: "
                "pip uninstall google-generativeai)"
            ) from e

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set.\n"
                "Get a free key at https://aistudio.google.com/apikey "
                "(no credit card), then run:\n    export GEMINI_API_KEY=AIza..."
            )

        self.cfg = cfg
        self.cuisine = cuisine
        self.client = genai.Client(api_key=api_key)
        self._types = types
        self.prompt_fn = get_prompt_fn(cfg.prompt_style, cuisine)

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        context = format_context(retrieved)
        prompt = self.prompt_fn(question, context)

        response = self.client.models.generate_content(
            model=self.cfg.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                max_output_tokens=self.cfg.max_new_tokens,
                temperature=0.0 if not self.cfg.do_sample else 0.7,
            ),
        )
        return (response.text or "").strip()


# ── Factory ─────────────────────────────────────────────────────────────────

def build_generator(cfg: GenerationConfig, cuisine: str = "Mediterranean") -> Generator:
    """Build the configured generation backend."""
    backend = cfg.backend
    if backend == "local":
        return LocalQwenGenerator(cfg, cuisine)
    if backend == "groq":
        return GroqGenerator(cfg, cuisine)
    if backend == "gemini":
        return GeminiGenerator(cfg, cuisine)
    raise ValueError(
        f"Unknown generation backend: {backend!r}. Valid: local, groq, gemini"
    )
