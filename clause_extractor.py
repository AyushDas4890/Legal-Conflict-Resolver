"""
Phase 2: Clause Extraction & Segmentation
==========================================
Detects clause boundaries and classifies each clause into:
  OBLIGATION | PERMISSION | PROHIBITION | CONDITION | PENALTY | OTHER

Architecture:
  1. spaCy rule-based boundary detector (fast, high-recall)
  2. BERT token classifier fine-tuned on CUAD spans (high-precision)
  3. Clause-type tagger using legal keyword heuristics + model

Usage:
    python clause_extractor.py --input docs/contract.txt --output extracted.json
"""

import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from loguru import logger

# ── Types ───────────────────────────────────────────────────────────────────────
CLAUSE_TYPES = [
    "OBLIGATION",    # "shall", "must", "is required to"
    "PERMISSION",    # "may", "is permitted to", "has the right to"
    "PROHIBITION",   # "shall not", "must not", "is prohibited"
    "CONDITION",     # "if", "unless", "provided that", "subject to"
    "PENALTY",       # "liable", "damages", "penalty", "indemnify"
    "DEFINITION",    # "means", "defined as", "refers to"
    "OTHER",
]

# ── Legal keyword patterns (heuristic pre-classifier) ───────────────────────────
KEYWORD_PATTERNS = {
    "OBLIGATION":  [
        r"\bshall\b", r"\bmust\b", r"\bis required to\b", r"\bhereby agrees to\b",
        r"\bwill be responsible\b", r"\bobligated to\b",
    ],
    "PERMISSION": [
        r"\bmay\b", r"\bis permitted\b", r"\bhas the right to\b",
        r"\bhas the option\b", r"\bat its discretion\b",
    ],
    "PROHIBITION": [
        r"\bshall not\b", r"\bmust not\b", r"\bis prohibited\b",
        r"\bwill not\b", r"\bnot be permitted\b", r"\bforbidden\b",
    ],
    "CONDITION": [
        r"\bif\b", r"\bunless\b", r"\bprovided that\b", r"\bsubject to\b",
        r"\bin the event that\b", r"\bcontingent upon\b",
    ],
    "PENALTY": [
        r"\bpenalty\b", r"\bdamages\b", r"\bindemnify\b", r"\bliable\b",
        r"\bbreach\b", r"\bdefault\b", r"\bforfeiture\b",
    ],
    "DEFINITION": [
        r"\bmeans\b", r"\bdefined as\b", r"\brefers to\b", r"\bshall mean\b",
        r'"[A-Z][^"]+"\s+means',
    ],
}

# Compile once for performance
COMPILED_PATTERNS = {
    ctype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ctype, patterns in KEYWORD_PATTERNS.items()
}


@dataclass
class Clause:
    """A single extracted clause with metadata."""
    clause_id:    str
    text:         str
    clause_type:  str
    confidence:   float          # Heuristic or model confidence 0–1
    start_char:   int
    end_char:     int
    source_doc:   str = ""
    section:      str = ""
    financial_keywords: list[str] = None

    def __post_init__(self):
        if self.financial_keywords is None:
            self.financial_keywords = []


# ── Financial covenant keywords (boost severity score in Phase 5) ────────────────
FINANCIAL_KEYWORDS = [
    "payment", "interest rate", "principal", "covenant", "collateral",
    "maturity", "default", "amortization", "dividend", "leverage ratio",
    "ebitda", "debt service", "prepayment", "acceleration", "remedy",
]
FINANCIAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in FINANCIAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class ClauseExtractor:
    """
    Two-stage clause extractor:
    Stage 1 — spaCy sentence boundary detection + rule-based splitting
    Stage 2 — Heuristic clause-type classification (+ optional BERT NER)
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        bert_model: Optional[str] = None,
        device: int = -1,
    ):
        logger.info(f"Loading spaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
            self.nlp = None

        self.bert_pipeline = None
        if bert_model:
            logger.info(f"Loading BERT NER pipeline: {bert_model}")
            tokenizer = AutoTokenizer.from_pretrained(bert_model)
            model     = AutoModelForTokenClassification.from_pretrained(bert_model)
            self.bert_pipeline = pipeline(
                "ner", model=model, tokenizer=tokenizer,
                aggregation_strategy="simple", device=device
            )

    def segment_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (sentence_text, start_char, end_char).
        Uses spaCy's dependency parser for accurate legal text splitting.
        """
        if self.nlp is None:
            # Fallback: split on sentence-ending punctuation
            sentences = []
            for m in re.finditer(r'[^.!?]+[.!?]+', text):
                sentences.append((m.group().strip(), m.start(), m.end()))
            return sentences

        doc = self.nlp(text)
        return [(sent.text.strip(), sent.start_char, sent.end_char) for sent in doc.sents]

    def classify_clause_type(self, text: str) -> tuple[str, float]:
        """
        Heuristic multi-label clause type classification.
        Priority order: PROHIBITION > OBLIGATION > PERMISSION > CONDITION > PENALTY > DEFINITION
        Returns (clause_type, confidence_score).
        """
        scores = {}
        for ctype, patterns in COMPILED_PATTERNS.items():
            matches = sum(1 for p in patterns if p.search(text))
            scores[ctype] = matches

        if all(v == 0 for v in scores.values()):
            return "OTHER", 0.1

        best_type = max(scores, key=scores.get)
        total     = sum(scores.values())
        confidence = scores[best_type] / total if total > 0 else 0.0

        # Boost confidence for strong single-match types
        confidence = min(0.95, confidence + 0.15) if scores[best_type] >= 2 else confidence

        return best_type, round(confidence, 3)

    def extract_financial_keywords(self, text: str) -> list[str]:
        """Returns financial keywords found in a clause."""
        return list({m.group().lower() for m in FINANCIAL_RE.finditer(text)})

    def extract(
        self,
        text: str,
        source_doc: str = "",
        min_length: int = 20,
        max_length: int = 2000,
    ) -> list[Clause]:
        """
        Main extraction method.
        Returns a list of Clause objects from the input text.
        """
        sentences = self.segment_sentences(text)
        clauses = []
        clause_idx = 0

        # ── Group short consecutive sentences into clause units ─────────────────
        buffer_text  = ""
        buffer_start = 0
        buffer_end   = 0

        def flush_buffer():
            nonlocal buffer_text, buffer_start, buffer_end, clause_idx
            if len(buffer_text.strip()) < min_length:
                buffer_text = ""
                return

            ctype, conf = self.classify_clause_type(buffer_text)
            fin_kw      = self.extract_financial_keywords(buffer_text)

            clauses.append(Clause(
                clause_id         = f"C-{clause_idx:04d}",
                text              = buffer_text.strip(),
                clause_type       = ctype,
                confidence        = conf,
                start_char        = buffer_start,
                end_char          = buffer_end,
                source_doc        = source_doc,
                financial_keywords= fin_kw,
            ))
            clause_idx += 1
            buffer_text = ""

        for sent_text, start, end in sentences:
            # Start new buffer or extend current one
            if not buffer_text:
                buffer_start = start

            buffer_text += " " + sent_text
            buffer_end   = end

            # Flush if buffer is long enough or hit a "period paragraph" break
            if len(buffer_text) >= max_length:
                flush_buffer()

        flush_buffer()  # Flush remaining

        logger.info(
            f"Extracted {len(clauses)} clauses from '{source_doc}' "
            f"({sum(1 for c in clauses if c.clause_type != 'OTHER')} typed)"
        )
        return clauses

    def extract_from_file(self, file_path: str | Path) -> list[Clause]:
        """Convenience wrapper to extract clauses from a text file."""
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return self.extract(text, source_doc=str(file_path))


def clauses_to_json(clauses: list[Clause], output_path: str | Path):
    """Serialize extracted clauses to JSON."""
    data = [asdict(c) for c in clauses]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.success(f"Saved {len(clauses)} clauses → {output_path}")


def evaluate_extraction(
    predicted: list[Clause],
    gold_spans: list[dict],
) -> dict:
    """
    Evaluate extraction against CUAD gold spans.
    Returns Accuracy, Precision, Recall, F1 (four-pillar metrics).

    gold_spans: [{"text": "...", "clause_type": "OBLIGATION", ...}, ...]
    """
    from sklearn.metrics import classification_report, accuracy_score

    pred_types = [c.clause_type for c in predicted]
    gold_types = [g.get("clause_type", "OTHER") for g in gold_spans]

    # Align lengths (trim to shorter list for evaluation)
    min_len    = min(len(pred_types), len(gold_types))
    pred_types = pred_types[:min_len]
    gold_types = gold_types[:min_len]

    if not pred_types:
        return {"error": "No predictions to evaluate"}

    report = classification_report(gold_types, pred_types, output_dict=True, zero_division=0)
    acc    = accuracy_score(gold_types, pred_types)

    macro = report.get("macro avg", {})
    return {
        "accuracy":  round(acc, 4),
        "precision": round(macro.get("precision", 0.0), 4),
        "recall":    round(macro.get("recall", 0.0), 4),
        "f1_score":  round(macro.get("f1-score", 0.0), 4),
        "per_class": {
            k: v for k, v in report.items()
            if k not in ["accuracy", "macro avg", "weighted avg"]
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Legal Clause Extractor")
    parser.add_argument("--input",  required=True,  help="Path to input text file")
    parser.add_argument("--output", required=True,  help="Path to output JSON file")
    parser.add_argument("--bert",   default=None,   help="Optional BERT NER model path")
    args = parser.parse_args()

    extractor = ClauseExtractor(bert_model=args.bert)
    clauses   = extractor.extract_from_file(args.input)
    clauses_to_json(clauses, args.output)
