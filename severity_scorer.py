"""
Phase 5: Conflict Severity Scoring & Report Generation
=======================================================
Scores each detected contradiction on a 0–1 scale using:
  1. NLI contradiction confidence (from DeBERTa)
  2. Clause type weight matrix (OBLIGATION vs OBLIGATION = highest severity)
  3. Financial keyword density (amounts, dates, penalties boost severity)

Produces structured JSON reports with human-readable recommendations.

Usage:
    scorer  = SeverityScorer()
    reports = scorer.score(nli_results)
    scorer.save_report(reports, "output/conflict_report.json")
"""

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ── Severity Weight Matrix ───────────────────────────────────────────────────────
# Each (clause_type_A, clause_type_B) pair has an inherent conflict severity multiplier.
# OBLIGATION vs PROHIBITION is the most severe legal conflict.
SEVERITY_MATRIX = {
    ("OBLIGATION",  "PROHIBITION"):  1.00,   # Highest — direct conflict
    ("OBLIGATION",  "OBLIGATION"):   0.90,   # Conflicting duties
    ("OBLIGATION",  "PERMISSION"):   0.75,   # Duty vs discretionary
    ("PERMISSION",  "PROHIBITION"):  0.85,   # Right vs ban
    ("PROHIBITION", "PROHIBITION"):  0.70,   # Conflicting bans
    ("CONDITION",   "OBLIGATION"):   0.65,   # Conditional obligation conflict
    ("CONDITION",   "PROHIBITION"):  0.60,
    ("PENALTY",     "OBLIGATION"):   0.80,   # Penalty triggers obligation conflict
    ("PENALTY",     "CONDITION"):    0.55,
    ("DEFINITION",  "OBLIGATION"):   0.50,   # Definitional mismatch
    ("OTHER",       "OTHER"):        0.30,   # Low-severity default
}

SEVERITY_MATRIX_DEFAULT = 0.40

# ── Financial Keyword Severity Boost ────────────────────────────────────────────
FINANCIAL_BOOST_KEYWORDS = {
    "shall pay":     0.08,
    "payment":       0.06,
    "interest rate": 0.07,
    "principal":     0.07,
    "default":       0.09,
    "covenant":      0.07,
    "force majeure": 0.06,
    "indemnify":     0.08,
    "termination":   0.06,
    "breach":        0.09,
    "penalty":       0.08,
    "ebitda":        0.06,
    "collateral":    0.07,
    "acceleration":  0.07,
    "waiver":        0.05,
}

# ── Conflict Type Labels ─────────────────────────────────────────────────────────
CONFLICT_TYPES = {
    ("OBLIGATION",  "PROHIBITION"):  "OBLIGATION_vs_PROHIBITION",
    ("OBLIGATION",  "OBLIGATION"):   "CONFLICTING_OBLIGATIONS",
    ("OBLIGATION",  "PERMISSION"):   "OBLIGATION_vs_DISCRETION",
    ("PERMISSION",  "PROHIBITION"):  "PERMISSION_vs_PROHIBITION",
    ("PROHIBITION", "PROHIBITION"):  "CONFLICTING_PROHIBITIONS",
    ("CONDITION",   "OBLIGATION"):   "CONDITIONAL_OBLIGATION_CONFLICT",
    ("PENALTY",     "OBLIGATION"):   "PENALTY_OBLIGATION_CONFLICT",
    ("CONDITION",   "PROHIBITION"):  "CONDITIONAL_PROHIBITION_CONFLICT",
    ("DEFINITION",  "OBLIGATION"):   "DEFINITIONAL_MISMATCH",
    ("OTHER",       "OTHER"):        "GENERAL_CONFLICT",
}


def _get_financial_boost(text_a: str, text_b: str) -> tuple[float, list[str]]:
    """Returns financial severity boost and matching keywords."""
    combined = (text_a + " " + text_b).lower()
    boost     = 0.0
    found_kws = []
    for kw, weight in FINANCIAL_BOOST_KEYWORDS.items():
        if kw in combined:
            boost    += weight
            found_kws.append(kw)
    return min(boost, 0.20), found_kws   # Cap financial boost at 0.20


def _get_clause_type_weight(type_a: str, type_b: str) -> tuple[float, str]:
    """Looks up severity weight and conflict type from the type matrix."""
    key   = (type_a, type_b)
    rev   = (type_b, type_a)
    weight = SEVERITY_MATRIX.get(key, SEVERITY_MATRIX.get(rev, SEVERITY_MATRIX_DEFAULT))
    label  = CONFLICT_TYPES.get(key, CONFLICT_TYPES.get(rev, "GENERAL_CONFLICT"))
    return weight, label


def _generate_recommendation(
    conflict_type: str,
    clause_a: str,
    clause_b: str,
    financial_kws: list[str],
) -> str:
    """Generates a concrete human-readable recommendation for each conflict."""
    base_recs = {
        "OBLIGATION_vs_PROHIBITION":    "Revise the prohibition clause or add an explicit exception for the stated obligation.",
        "CONFLICTING_OBLIGATIONS":      "Clarify priority order of the two obligations or merge into a single harmonized clause.",
        "OBLIGATION_vs_DISCRETION":     "Confirm whether the discretionary right supersedes or is subordinate to the stated obligation.",
        "PERMISSION_vs_PROHIBITION":    "Resolve whether the permission or prohibition takes precedence; add a governing clause.",
        "CONFLICTING_PROHIBITIONS":     "Reconcile the scope of the two prohibitions and define exceptions explicitly.",
        "CONDITIONAL_OBLIGATION_CONFLICT": "Define the conditions precedent to each obligation to avoid ambiguity.",
        "PENALTY_OBLIGATION_CONFLICT":  "Clarify trigger conditions for penalty vis-à-vis the stated obligation.",
        "DEFINITIONAL_MISMATCH":        "Align definitions used in both documents to ensure consistent interpretation.",
        "GENERAL_CONFLICT":             "Review both clauses with legal counsel to resolve the inconsistency.",
    }
    rec = base_recs.get(conflict_type, base_recs["GENERAL_CONFLICT"])

    if financial_kws:
        kw_str = ", ".join(f'"{k}"' for k in financial_kws[:3])
        rec += f" Special attention required for financial terms: {kw_str}."

    return rec


@dataclass
class ConflictReport:
    """Structured output for a single detected conflict."""
    conflict_id:        str
    pair_id:            str
    doc_a_clause_id:    str
    doc_b_clause_id:    str
    doc_a_clause:       str
    doc_b_clause:       str
    conflict_type:      str
    severity:           float          # 0.0 – 1.0
    severity_label:     str            # LOW | MEDIUM | HIGH | CRITICAL
    nli_confidence:     float
    clause_type_weight: float
    financial_boost:    float
    financial_keywords: list[str]      = field(default_factory=list)
    attention_tokens_a: list[str]      = field(default_factory=list)
    attention_tokens_b: list[str]      = field(default_factory=list)
    recommendation:     str            = ""
    clause_a_type:      str            = "OTHER"
    clause_b_type:      str            = "OTHER"


def _severity_label(score: float) -> str:
    if score >= 0.87:   return "CRITICAL"
    elif score >= 0.70: return "HIGH"
    elif score >= 0.50: return "MEDIUM"
    else:               return "LOW"


class SeverityScorer:
    """
    Scores all flagged NLI contradictions into structured ConflictReports.
    Severity = weighted combination of:
      (a) NLI contradiction confidence   [weight: 0.50]
      (b) Clause type conflict weight    [weight: 0.30]
      (c) Financial keyword boost        [weight: 0.20, capped]
    """

    def score(self, nli_results: list[dict]) -> list[ConflictReport]:
        """
        Input: list of NLIResult dicts (from nli_model.py)
        Output: sorted list of ConflictReport (highest severity first)
        """
        reports   = []
        report_id = 0

        for result in nli_results:
            if not result.get("is_flagged", False):
                continue

            type_a = result.get("clause_a_type", "OTHER").upper()
            type_b = result.get("clause_b_type", "OTHER").upper()

            # ── Component scores ──────────────────────────────────────────────
            nli_conf    = result.get("contradiction_prob", result.get("confidence", 0.5))
            type_weight, conflict_type = _get_clause_type_weight(type_a, type_b)
            fin_boost, fin_kws = _get_financial_boost(
                result.get("clause_a_text", ""),
                result.get("clause_b_text", ""),
            )

            # Weighted severity formula
            severity = (
                0.50 * nli_conf +
                0.30 * type_weight +
                0.20 * (fin_boost / 0.20)   # Normalize fin_boost to [0, 1]
            )
            severity = round(min(max(severity, 0.0), 1.0), 4)

            rec = _generate_recommendation(
                conflict_type,
                result.get("clause_a_text", ""),
                result.get("clause_b_text", ""),
                fin_kws,
            )

            reports.append(ConflictReport(
                conflict_id         = f"C-{report_id:04d}",
                pair_id             = result.get("pair_id", ""),
                doc_a_clause_id     = result.get("clause_a_id", ""),
                doc_b_clause_id     = result.get("clause_b_id", ""),
                doc_a_clause        = result.get("clause_a_text", ""),
                doc_b_clause        = result.get("clause_b_text", ""),
                conflict_type       = conflict_type,
                severity            = severity,
                severity_label      = _severity_label(severity),
                nli_confidence      = round(nli_conf, 4),
                clause_type_weight  = type_weight,
                financial_boost     = round(fin_boost, 4),
                financial_keywords  = fin_kws,
                attention_tokens_a  = result.get("attention_tokens_a", []),
                attention_tokens_b  = result.get("attention_tokens_b", []),
                recommendation      = rec,
                clause_a_type       = type_a,
                clause_b_type       = type_b,
            ))
            report_id += 1

        # Sort by severity (highest first)
        reports.sort(key=lambda r: r.severity, reverse=True)

        n_critical = sum(1 for r in reports if r.severity_label == "CRITICAL")
        n_high     = sum(1 for r in reports if r.severity_label == "HIGH")
        logger.info(
            f"Severity scoring: {len(reports)} conflicts "
            f"({n_critical} CRITICAL, {n_high} HIGH)"
        )
        return reports

    def save_report(self, reports: list[ConflictReport], output_path: str | Path):
        """Saves all conflict reports as a structured JSON."""
        data = {
            "total_conflicts": len(reports),
            "summary": {
                "CRITICAL": sum(1 for r in reports if r.severity_label == "CRITICAL"),
                "HIGH":     sum(1 for r in reports if r.severity_label == "HIGH"),
                "MEDIUM":   sum(1 for r in reports if r.severity_label == "MEDIUM"),
                "LOW":      sum(1 for r in reports if r.severity_label == "LOW"),
            },
            "conflicts": [asdict(r) for r in reports],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.success(f"Conflict report saved → {output_path}")
        return data

    def evaluate_scoring(
        self,
        reports: list[ConflictReport],
        human_ratings: list[dict],
    ) -> dict:
        """
        Compare system severity labels vs human expert ratings.
        Returns four-pillar metrics: Accuracy, Precision, Recall, F1.

        human_ratings: [{"conflict_id": "C-0001", "severity_label": "HIGH"}, ...]
        """
        LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

        id_to_report = {r.conflict_id: r for r in reports}
        y_pred, y_true = [], []

        for hr in human_ratings:
            cid = hr.get("conflict_id")
            if cid in id_to_report:
                y_true.append(LABEL_MAP.get(hr["severity_label"].upper(), 0))
                y_pred.append(LABEL_MAP.get(id_to_report[cid].severity_label, 0))

        if not y_true:
            return {"error": "No matched human ratings"}

        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "evaluated": len(y_true),
        }
