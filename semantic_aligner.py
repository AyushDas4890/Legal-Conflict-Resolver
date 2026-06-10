"""
Phase 3: Semantic Alignment (Cross-Document Retrieval)
=======================================================
Encodes all clauses from both documents using sentence-transformers,
builds a FAISS index over Doc B, and retrieves top-K candidates per
clause in Doc A for downstream NLI contradiction checking.

This is the "hard-attention" pre-filter that keeps complexity at O(N·K)
instead of an intractable O(N·M) cross-product.

Usage:
    aligner = SemanticAligner()
    aligner.index_doc_b(clauses_b)
    pairs   = aligner.align(clauses_a, top_k=5)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ── Config ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "law-ai/InLegalBERT"      # Legal-domain BERT for embeddings
FALLBACK_MODEL  = "all-MiniLM-L6-v2"        # Lightweight fallback
SIM_THRESHOLD   = 0.45                       # Cosine similarity cutoff for "related"
TOP_K           = 5


@dataclass
class ClausePair:
    """A candidate pair of aligned clauses for NLI evaluation."""
    pair_id:        str
    clause_a_id:    str
    clause_a_text:  str
    clause_a_type:  str
    clause_b_id:    str
    clause_b_text:  str
    clause_b_type:  str
    similarity:     float
    rank:           int


class SemanticAligner:
    """
    Bi-Encoder Semantic Alignment using sentence-transformers + FAISS.

    Architecture:
      Clause A ──► Bi-Encoder ──► Vector A ──┐
      Clause B ──► Bi-Encoder ──► Vector B ──┤
                                              └──► FAISS Index ──► Top-K matches
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        logger.info(f"Loading sentence encoder: {model_name}")
        try:
            self.encoder = SentenceTransformer(model_name, device=device)
            logger.success(f"Loaded {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Using fallback.")
            self.encoder = SentenceTransformer(FALLBACK_MODEL, device=device)

        self.index:         Optional[faiss.Index] = None
        self.doc_b_clauses: list[dict]            = []
        self.doc_b_vecs:    Optional[np.ndarray]  = None

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encodes a list of texts into L2-normalised embeddings."""
        vecs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def index_doc_b(self, clauses_b: list[dict]):
        """
        Build a FAISS Flat L2 index over Doc B's clause embeddings.
        clauses_b: list of Clause dicts (with 'text', 'clause_id', 'clause_type')
        """
        self.doc_b_clauses = clauses_b
        texts = [c["text"] for c in clauses_b]

        logger.info(f"Encoding {len(texts)} clauses from Doc B...")
        self.doc_b_vecs = self._encode(texts)

        dim   = self.doc_b_vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner-Product == cosine sim (normalized)
        self.index.add(self.doc_b_vecs)
        logger.success(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def align(
        self,
        clauses_a: list[dict],
        top_k: int = TOP_K,
        threshold: float = SIM_THRESHOLD,
    ) -> list[ClausePair]:
        """
        For each clause in Doc A, retrieve top-K semantically similar clauses
        from Doc B that exceed the similarity threshold.

        Returns a list of ClausePair objects for NLI evaluation.
        """
        if self.index is None:
            raise RuntimeError("Call index_doc_b() first.")

        texts_a = [c["text"] for c in clauses_a]
        logger.info(f"Encoding {len(texts_a)} clauses from Doc A...")
        vecs_a  = self._encode(texts_a)

        logger.info(f"Retrieving top-{top_k} matches per clause...")
        sims, indices = self.index.search(vecs_a, top_k)

        pairs     = []
        pair_idx  = 0

        for i, (clause_a, sim_row, idx_row) in enumerate(
            zip(clauses_a, sims, indices)
        ):
            for rank, (sim_score, b_idx) in enumerate(zip(sim_row, idx_row)):
                if sim_score < threshold:
                    continue          # Below relevance threshold — skip
                if b_idx < 0 or b_idx >= len(self.doc_b_clauses):
                    continue

                clause_b = self.doc_b_clauses[b_idx]
                pairs.append(ClausePair(
                    pair_id       = f"P-{pair_idx:05d}",
                    clause_a_id   = clause_a.get("clause_id", f"A-{i}"),
                    clause_a_text = clause_a["text"],
                    clause_a_type = clause_a.get("clause_type", "OTHER"),
                    clause_b_id   = clause_b.get("clause_id", f"B-{b_idx}"),
                    clause_b_text = clause_b["text"],
                    clause_b_type = clause_b.get("clause_type", "OTHER"),
                    similarity    = round(float(sim_score), 4),
                    rank          = rank + 1,
                ))
                pair_idx += 1

        logger.info(f"Alignment complete: {len(pairs)} candidate pairs (threshold={threshold})")
        return pairs

    def save_pairs(self, pairs: list[ClausePair], output_path: str | Path):
        data = [asdict(p) for p in pairs]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.success(f"Saved {len(pairs)} aligned pairs → {output_path}")


# ── Evaluation (Retrieval Recall@K, MAP, Precision, Recall, F1) ─────────────────
def evaluate_alignment(
    pairs: list[ClausePair],
    ground_truth: list[dict],
    k: int = 5,
) -> dict:
    """
    Evaluates the alignment step using:
      - Recall@K: Does the true contradicting clause appear in top-K?
      - MAP (Mean Average Precision)
      - Accuracy, Precision, Recall, F1 (four-pillar metrics)

    ground_truth: [{"clause_a_id": "...", "clause_b_id": "...", "label": "CONTRADICTION"}, ...]
    """
    # Build lookup of true contradicting B clauses per A clause
    true_pairs = {
        (g["clause_a_id"], g["clause_b_id"])
        for g in ground_truth
        if g.get("label") == "CONTRADICTION"
    }

    # Group retrieved pairs by clause_a_id
    retrieved: dict[str, list[str]] = {}
    for p in pairs:
        retrieved.setdefault(p.clause_a_id, []).append(p.clause_b_id)

    # Recall@K
    recall_hits = 0
    total_true  = len({a for a, b in true_pairs})

    for a_id, b_ids in retrieved.items():
        top_k_b = b_ids[:k]
        if any((a_id, b_id) in true_pairs for b_id in top_k_b):
            recall_hits += 1

    recall_at_k = recall_hits / total_true if total_true > 0 else 0.0

    # Binary classification metrics across all pairs
    y_true, y_pred = [], []
    for p in pairs:
        is_true = (p.clause_a_id, p.clause_b_id) in true_pairs
        y_true.append(int(is_true))
        y_pred.append(1)   # All retrieved pairs are "positive" predictions

    # Add false negatives (true pairs NOT retrieved)
    retrieved_set = {(p.clause_a_id, p.clause_b_id) for p in pairs}
    for a_id, b_id in true_pairs:
        if (a_id, b_id) not in retrieved_set:
            y_true.append(1)
            y_pred.append(0)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return {
        "recall_at_k":   round(recall_at_k, 4),
        "map":           round(float(np.mean([1.0 if y else 0.0 for y in y_true[:len(pairs)]])), 4),
        "accuracy":      round(acc, 4),
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1_score":      round(f1, 4),
        "total_pairs":   len(pairs),
        "total_true":    len(true_pairs),
        "k":             k,
    }
