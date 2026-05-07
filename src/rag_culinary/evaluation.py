"""Evaluation metrics for retrieval and generation.

Retrieval: Hit@K and MRR over a benchmark of (question, gold_answer, source).
Generation: Exact match, ROUGE-L, BERTScore F1.

Two hit modes are supported:

  - "lenient" (default, matches the original notebook): a chunk counts as a
    hit if it comes from the gold source file OR if the gold answer is a
    substring of the chunk OR if there is high keyword overlap. With short
    documents (~one chunk per file), the source-filename rule dominates and
    metrics tend toward 1.0 — which makes this mode an upper bound on
    retrieval quality.

  - "strict": a chunk counts as a hit only when the gold answer text actually
    appears in the chunk (substring match) or has high keyword overlap. This
    is harder and gives more discriminating numbers across configurations.

Use strict mode when comparing different retrievers against each other; use
lenient mode when reporting "did we find the right document?" performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rag_culinary.chunking import Chunk
from rag_culinary.corpus import QAPair
from rag_culinary.retrieval import RetrievalResult

HitMode = Literal["lenient", "strict"]


# ── Retrieval ────────────────────────────────────────────────────────────────

def is_hit(
    gold_answer: str,
    gold_source: str,
    chunk: Chunk,
    mode: HitMode = "lenient",
    keyword_overlap_threshold: float = 0.6,
) -> bool:
    """Was this chunk a valid answer source for the gold QA pair?

    lenient mode (default, matches notebook):
        1. Same source filename                                  -> hit
        2. Gold answer appears as substring of chunk             -> hit
        3. >= threshold of gold-answer keywords appear in chunk  -> hit

    strict mode:
        Skips the source-filename shortcut. A hit requires the chunk to
        actually contain the gold answer text (or a high keyword overlap).
        This is the correct measure when chunks ≈ documents in size, because
        the filename check otherwise lets you "hit" without actually
        retrieving the answer text.
    """
    if mode == "lenient" and chunk.source == gold_source:
        return True

    if gold_answer.lower().strip() in chunk.text.lower():
        return True

    gold_words = {w for w in gold_answer.lower().split() if len(w) > 3}
    if not gold_words:
        return False
    chunk_words = set(chunk.text.lower().split())
    overlap = len(gold_words & chunk_words) / len(gold_words)
    return overlap >= keyword_overlap_threshold


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float

    def to_dict(self) -> dict:
        return {
            "Hit@1": round(self.hit_at_1, 3),
            "Hit@3": round(self.hit_at_3, 3),
            "Hit@5": round(self.hit_at_5, 3),
            "MRR":   round(self.mrr, 3),
        }


def evaluate_retrieval(
    benchmark: list[QAPair],
    retrieved_per_query: list[list[RetrievalResult]],
    mode: HitMode = "lenient",
) -> RetrievalMetrics:
    if len(benchmark) != len(retrieved_per_query):
        raise ValueError("benchmark and retrieved_per_query must align 1:1")

    hits = {1: 0, 3: 0, 5: 0}
    rrs: list[float] = []
    for qa, results in zip(benchmark, retrieved_per_query):
        first_hit_rank = None
        for r in results:
            if is_hit(qa.answer, qa.source_file, r.chunk, mode=mode):
                first_hit_rank = r.rank
                break
        if first_hit_rank is not None:
            for k in hits:
                if first_hit_rank <= k:
                    hits[k] += 1
            rrs.append(1.0 / first_hit_rank)
        else:
            rrs.append(0.0)

    n = len(benchmark)
    return RetrievalMetrics(
        hit_at_1=hits[1] / n,
        hit_at_3=hits[3] / n,
        hit_at_5=hits[5] / n,
        mrr=float(np.mean(rrs)),
    )


# ── Generation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationMetrics:
    exact_match: float
    rouge_l: float
    bertscore_f1: float

    def to_dict(self) -> dict:
        return {
            "ExactMatch":   round(self.exact_match, 3),
            "ROUGE-L":      round(self.rouge_l, 3),
            "BERTScore-F1": round(self.bertscore_f1, 3),
        }


def _exact_match(pred: str, gold: str) -> int:
    return 1 if gold.lower().strip() in pred.lower() else 0


def evaluate_generation(predictions: list[str], golds: list[str]) -> GenerationMetrics:
    """Compute Exact Match, ROUGE-L F1, and BERTScore F1.

    Imports rouge_score and bert_score lazily — they are heavy and not always
    needed (e.g. inference-only runs).
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must align 1:1")
    if not predictions:
        return GenerationMetrics(0.0, 0.0, 0.0)

    from rouge_score import rouge_scorer
    from bert_score import score as bertscore

    em = [_exact_match(p, g) for p, g in zip(predictions, golds)]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(predictions, golds)]

    _, _, f1 = bertscore(predictions, golds, lang="en", verbose=False)
    bert_f1 = [float(x) for x in f1]

    return GenerationMetrics(
        exact_match=float(np.mean(em)),
        rouge_l=float(np.mean(rouge)),
        bertscore_f1=float(np.mean(bert_f1)),
    )
