"""Run the full benchmark evaluation: retrieval metrics + (optionally) generation.

Reports retrieval metrics in BOTH lenient and strict modes, because the
lenient definition (which counts any chunk from the gold source file as a hit)
saturates to 1.0 on this corpus, where most documents fit in a single chunk.
The strict mode (gold answer must actually appear in the retrieved chunk) is
more discriminating across configurations.

Usage:
    python scripts/evaluate.py                          # all 748 questions
    python scripts/evaluate.py --limit 50               # quick subset
    python scripts/evaluate.py --generation             # add LLM metrics
    python scripts/evaluate.py --strict-only            # skip lenient mode
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_culinary.config import load_config
from rag_culinary.corpus import load_benchmark
from rag_culinary.evaluation import evaluate_generation, evaluate_retrieval
from rag_culinary.pipeline import RAGPipeline


def _print_metrics(label: str, metrics_dict: dict) -> None:
    print(f"      {label}:")
    for k, v in metrics_dict.items():
        print(f"        {k:<6} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark evaluation.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--generation", action="store_true",
                    help="Also run LLM generation and report ROUGE/BERTScore")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate on first N benchmark items (for quick runs)")
    ap.add_argument("--out-dir", default="evaluation",
                    help="Directory to write metrics JSON files")
    ap.add_argument("--strict-only", action="store_true",
                    help="Skip lenient mode and only report strict-mode metrics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(cfg.paths.benchmark)
    if args.limit:
        benchmark = benchmark[: args.limit]
    print(f"Evaluating on {len(benchmark)} benchmark items")

    rag = RAGPipeline.from_artifacts(cfg)

    # ── Retrieval (run once, score twice) ────────────────────────────────────
    print("\n[1/2] Retrieval...")
    t0 = time.time()
    retrieved_per_query = [rag.retrieve(qa.question) for qa in benchmark]
    print(f"      ({time.time() - t0:.1f}s)")

    summary: dict = {}
    if not args.strict_only:
        lenient = evaluate_retrieval(benchmark, retrieved_per_query, mode="lenient")
        _print_metrics("Lenient (source-match counts as hit)", lenient.to_dict())
        summary["lenient"] = lenient.to_dict()

    strict = evaluate_retrieval(benchmark, retrieved_per_query, mode="strict")
    _print_metrics("Strict (gold answer must appear in chunk)", strict.to_dict())
    summary["strict"] = strict.to_dict()

    with open(out_dir / "retrieval_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Generation (optional) ───────────────────────────────────────────────
    if args.generation:
        print("\n[2/2] Generation...")
        t0 = time.time()
        preds, golds = [], []
        for i, (qa, retrieved) in enumerate(zip(benchmark, retrieved_per_query), 1):
            ans = rag.generator.generate(qa.question, retrieved)
            preds.append(ans)
            golds.append(qa.answer)
            if i % 25 == 0 or i == len(benchmark):
                print(f"      [{i}/{len(benchmark)}] {time.time() - t0:.0f}s elapsed")

        gen_metrics = evaluate_generation(preds, golds)
        print("\n      Generation metrics:")
        for k, v in gen_metrics.to_dict().items():
            print(f"        {k:<14} {v}")

        with open(out_dir / "generation_metrics.json", "w") as f:
            json.dump(gen_metrics.to_dict(), f, indent=2)
    else:
        print("\n(Skipping generation — pass --generation to include)")

    print(f"\nWrote metrics to {out_dir}/")


if __name__ == "__main__":
    main()
