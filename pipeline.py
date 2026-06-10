"""
Pipeline Runner — End-to-End Contradiction Detection
====================================================
Orchestrates all 5 phases in sequence:
  1. Load pre-extracted clauses (or run extraction)
  2. Semantic alignment via FAISS
  3. NLI contradiction detection via DeBERTa
  4. Severity scoring
  5. Report generation

Usage:
    python pipeline.py --doc-a contract.txt --doc-b policy.txt --output report.json
    python pipeline.py --evaluate --model models/nli_deberta
"""

import json
import time
import argparse
from pathlib import Path

from loguru import logger

from clause_extractor import ClauseExtractor, clauses_to_json
from semantic_aligner import SemanticAligner, evaluate_alignment
from severity_scorer  import SeverityScorer


def run_pipeline(
    text_a:    str,
    text_b:    str,
    doc_a_name: str = "Document A",
    doc_b_name: str = "Document B",
    nli_model_dir: str = "models/nli_deberta",
    output_path: str = "output/conflict_report.json",
    top_k: int = 5,
) -> dict:
    """
    Full contradiction detection pipeline.
    Returns the structured conflict report dict.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("LEGAL-FINANCIAL CONFLICT RESOLVER — PIPELINE START")
    logger.info("=" * 60)

    # ── Phase 2: Clause Extraction ──────────────────────────────────
    logger.info("[Phase 2] Clause Extraction")
    extractor = ClauseExtractor()
    clauses_a = extractor.extract(text_a, source_doc=doc_a_name)
    clauses_b = extractor.extract(text_b, source_doc=doc_b_name)
    logger.info(f"  Doc A: {len(clauses_a)} clauses | Doc B: {len(clauses_b)} clauses")

    # ── Phase 3: Semantic Alignment ─────────────────────────────────
    logger.info("[Phase 3] Semantic Alignment")
    aligner = SemanticAligner()

    clauses_b_dicts = [{"clause_id": c.clause_id, "text": c.text, "clause_type": c.clause_type} for c in clauses_b]
    clauses_a_dicts = [{"clause_id": c.clause_id, "text": c.text, "clause_type": c.clause_type} for c in clauses_a]

    aligner.index_doc_b(clauses_b_dicts)
    pairs = aligner.align(clauses_a_dicts, top_k=top_k)
    logger.info(f"  {len(pairs)} candidate pairs aligned")

    # ── Phase 4: NLI Contradiction Detection ────────────────────────
    logger.info("[Phase 4] NLI Contradiction Detection")
    from nli_model import NLIInferencer
    
    model_id = "cross-encoder/nli-deberta-v3-base"
    nli_model_path = Path(nli_model_dir)
    target = str(nli_model_path) if nli_model_path.exists() else model_id
    
    logger.info(f"  Loading model: {target}")
    nli = NLIInferencer.from_pretrained(target)
    pairs_as_dicts = [
        {
            "pair_id":       p.pair_id,
            "clause_a_id":   p.clause_a_id,
            "clause_b_id":   p.clause_b_id,
            "clause_a_text": p.clause_a_text,
            "clause_b_text": p.clause_b_text,
            "clause_a_type": p.clause_a_type,
            "clause_b_type": p.clause_b_type,
        }
        for p in pairs
    ]
    nli_results_obj = nli.predict_batch(pairs_as_dicts)
    nli_results = [
        {
            **r.__dict__,
            "clause_a_type": next((p["clause_a_type"] for p in pairs_as_dicts if p["pair_id"] == r.pair_id), "OTHER"),
            "clause_b_type": next((p["clause_b_type"] for p in pairs_as_dicts if p["pair_id"] == r.pair_id), "OTHER"),
        }
        for r in nli_results_obj
    ]

    # ── Phase 5: Severity Scoring ────────────────────────────────────
    logger.info("[Phase 5] Severity Scoring")
    scorer  = SeverityScorer()
    reports = scorer.score(nli_results)
    report  = scorer.save_report(reports, output_path)

    elapsed = round(time.time() - start, 2)
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed}s")
    logger.info(f"  Total conflicts: {len(reports)}")
    logger.info(f"  CRITICAL: {report['summary']['CRITICAL']} | HIGH: {report['summary']['HIGH']}")
    logger.info("=" * 60)

    report["processing_time"] = elapsed
    return report


def run_evaluation(model_dir: str, test_data_path: str = "data/raw/contract_nli/test.parquet"):
    """
    Evaluates the full pipeline on the ContractNLI test set.
    Reports Accuracy, Precision, Recall, F1 (four-pillar metrics).
    """
    import pandas as pd
    from nli_model import NLIInferencer

    logger.info("Running evaluation on ContractNLI test set...")
    if not Path(test_data_path).exists():
        logger.error(f"Test data not found: {test_data_path}. Run data_ingestion.py first.")
        return

    df = pd.read_parquet(test_data_path)
    logger.info(f"Test set: {len(df)} samples")

    # Prepare as NLI pairs
    pairs = [
        {
            "pair_id":       f"test-{i}",
            "clause_a_id":   f"a-{i}",
            "clause_b_id":   f"b-{i}",
            "clause_a_text": row.get("hypothesis", row.get("premise", "")),
            "clause_b_text": row.get("premise", ""),
            "clause_a_type": "OTHER",
            "clause_b_type": "OTHER",
        }
        for i, (_, row) in enumerate(df.iterrows())
    ]

    ground_truth = [
        {"pair_id": f"test-{i}", "label": row.get("label", "NOT_MENTIONED")}
        for i, (_, row) in enumerate(df.iterrows())
    ]

    nli     = NLIInferencer.from_pretrained(model_dir)
    results = nli.predict_batch(pairs)
    metrics = nli.evaluate(results, ground_truth)

    logger.info("=" * 40)
    logger.info("EVALUATION RESULTS (Four-Pillar Metrics)")
    logger.info("=" * 40)
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key:20s}: {val:.4f}")

    # Save metrics
    Path("output").mkdir(exist_ok=True)
    with open("output/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success("Metrics saved → output/eval_metrics.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal-Financial Conflict Resolver Pipeline")
    parser.add_argument("--doc-a",    help="Path to Document A (txt)")
    parser.add_argument("--doc-b",    help="Path to Document B (txt)")
    parser.add_argument("--output",   default="output/conflict_report.json")
    parser.add_argument("--model",    default="models/nli_deberta")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on ContractNLI test set")
    parser.add_argument("--top-k",    type=int, default=5)
    args = parser.parse_args()

    if args.evaluate:
        run_evaluation(args.model)
    elif args.doc_a and args.doc_b:
        text_a = Path(args.doc_a).read_text(encoding="utf-8")
        text_b = Path(args.doc_b).read_text(encoding="utf-8")
        run_pipeline(text_a, text_b, args.doc_a, args.doc_b, args.model, args.output, args.top_k)
    else:
        parser.print_help()
