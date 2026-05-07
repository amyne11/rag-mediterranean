"""Run inference on a query JSON file.

Reads the query file, runs the full RAG pipeline for each query, and writes
the outputs in the spec-required schema:

    {"results": [{"query_id": "0", "query": "...", "response": "...",
                  "retrieved_context": [{"doc_id": "000", "text": "..."}]}, ...]}

Usage:
    python scripts/run_inference.py --input queries.json --output results.json
    python scripts/run_inference.py --input queries.json --output results.json \
        --gold gold_answers.json     # also run evaluation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_culinary.config import load_config
from rag_culinary.evaluation import evaluate_generation
from rag_culinary.io_utils import (
    parse_gold_answers, parse_input_queries, write_outputs,
)
from rag_culinary.pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAG inference on a query JSON.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", required=True, help="Input query JSON")
    ap.add_argument("--output", default="test_outputs.json", help="Output JSON path")
    ap.add_argument("--gold", default=None, help="Optional gold-answer JSON for evaluation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Loading pipeline (chunking={cfg.chunking.strategy}, "
          f"retrieval={cfg.retrieval.strategy}, "
          f"prompt={cfg.generation.prompt_style})")
    rag = RAGPipeline.from_artifacts(cfg)

    queries = parse_input_queries(Path(args.input))
    print(f"Loaded {len(queries)} queries from {args.input}")

    results = []
    t0 = time.time()
    for i, q in enumerate(queries, start=1):
        ans = rag.answer_with_sources(q.question)
        results.append({
            "query_id": q.query_id,
            "query": q.question,
            "response": ans.answer,
            "retrieved_context": [
                {"doc_id": str(j).zfill(3), "text": r.chunk.text}
                for j, r in enumerate(ans.retrieved)
            ],
        })
        if i % 10 == 0 or i == len(queries):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(queries)}] {elapsed:.0f}s elapsed")

    write_outputs(results, Path(args.output))
    print(f"\nWrote {len(results)} answers -> {args.output}")

    # Optional evaluation
    if args.gold:
        print(f"\nEvaluating against {args.gold}...")
        gold = parse_gold_answers(Path(args.gold))
        preds, golds = [], []
        for r in results:
            qid = str(r["query_id"])
            if qid in gold:
                preds.append(r["response"])
                golds.append(gold[qid])
        print(f"Matched {len(preds)}/{len(results)} predictions to gold answers")

        metrics = evaluate_generation(preds, golds)
        print("\nEvaluation results:")
        for k, v in metrics.to_dict().items():
            print(f"  {k:<14} {v}")

        eval_path = Path(args.output).with_name("evaluation_results.json")
        with open(eval_path, "w") as f:
            json.dump({"num_evaluated": len(preds), **metrics.to_dict()}, f, indent=2)
        print(f"\n-> {eval_path}")


if __name__ == "__main__":
    main()
