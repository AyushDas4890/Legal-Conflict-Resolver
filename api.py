"""
FastAPI Backend: Legal-Financial Conflict Resolver API
======================================================
Serves the full pipeline as a REST API consumed by the Vite frontend.

Endpoints:
  POST /api/analyze       → Upload two docs, run full pipeline, return conflicts
  GET  /api/results/{id}  → Retrieve cached results by analysis ID
  GET  /api/health        → Health check + model status
  GET  /api/metrics       → Latest evaluation metrics

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# ── Document text extraction (PDF, DOCX, plain text) ──────────────────────────
def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from uploaded bytes based on file extension."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages  = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            logger.warning("pypdf not installed — treating PDF as raw text")
        except Exception as e:
            logger.warning(f"PDF extraction failed ({e}) — treating as raw text")

    elif ext == ".docx":
        try:
            import docx
            import io
            doc    = docx.Document(io.BytesIO(data))
            paras  = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paras)
        except ImportError:
            logger.warning("python-docx not installed — treating DOCX as raw text")
        except Exception as e:
            logger.warning(f"DOCX extraction failed ({e}) — treating as raw text")

    # Fallback: plain text
    return data.decode("utf-8", errors="replace")


from clause_extractor import ClauseExtractor, clauses_to_json, Clause
from semantic_aligner import SemanticAligner
from severity_scorer import SeverityScorer, ConflictReport

# ── Lazy-load NLI (heavy model) ─────────────────────────────────────────────────
_nli_inferencer = None

def get_nli_inferencer():
    global _nli_inferencer
    if _nli_inferencer is None:
        from nli_model import NLIInferencer
        
        # Use a tiny model on Render to fit in 512MB RAM
        if os.getenv("RENDER"):
            model_id = "cross-encoder/nli-miniLM-L6-v2"
            logger.info("Low-RAM mode: Using miniLM-L6 cross-encoder")
        else:
            model_id = "cross-encoder/nli-deberta-v3-base"
        
        model_dir = Path("models/nli_deberta")
        
        # Prefer local if exists, else load from hub
        target = str(model_dir) if model_dir.exists() else model_id
        logger.info(f"Initializing NLI Inferencer with: {target}")
        _nli_inferencer = NLIInferencer.from_pretrained(target)
    return _nli_inferencer


# ── App Setup ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Legal-Financial Conflict Resolver API",
    description = "Multi-Document Cross-Attention Logic for Contradiction Detection",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(","),
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── In-memory cache (replace with Redis in production) ──────────────────────────
ANALYSIS_CACHE: dict[str, dict] = {}

# ── Shared pipeline components ───────────────────────────────────────────────────
extractor = ClauseExtractor()
aligner   = SemanticAligner()
scorer    = SeverityScorer()

RESULTS_DIR = Path("output")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Pydantic Models ─────────────────────────────────────────────────────────────
class AnalysisStatus(BaseModel):
    analysis_id:  str
    status:       str    # pending | processing | done | error
    progress:     int    # 0–100
    message:      str    = ""


class ConflictSummary(BaseModel):
    conflict_id:      str
    conflict_type:    str
    severity:         float
    severity_label:   str
    doc_a_clause:     str
    doc_b_clause:     str
    clause_a_type:    str
    clause_b_type:    str
    attention_tokens_a: list[str]
    attention_tokens_b: list[str]
    recommendation:   str
    nli_confidence:   float


class AnalysisResult(BaseModel):
    analysis_id:    str
    status:         str
    total_conflicts: int
    summary: dict[str, int]
    conflicts:      list[ConflictSummary]
    metrics:        dict
    processing_time: float


# ── Background Analysis Task ─────────────────────────────────────────────────────
async def run_analysis(
    analysis_id: str,
    text_a: str,
    text_b: str,
    doc_a_name: str,
    doc_b_name: str,
):
    """Full pipeline: extract → align → nli → score."""
    start = time.time()
    ANALYSIS_CACHE[analysis_id] = {
        "status": "processing", "progress": 0,
        "message": "Extracting clauses from Document A..."
    }

    try:
        # ── Phase 2: Clause Extraction ─────────────────────────────────────────
        clauses_a = extractor.extract(text_a, source_doc=doc_a_name)
        clauses_b = extractor.extract(text_b, source_doc=doc_b_name)
        ANALYSIS_CACHE[analysis_id].update({"progress": 25, "message": "Clauses extracted. Aligning..."})

        # ── Phase 3: Semantic Alignment ────────────────────────────────────────
        clauses_b_dicts = [
            {"clause_id": c.clause_id, "text": c.text, "clause_type": c.clause_type}
            for c in clauses_b
        ]
        clauses_a_dicts = [
            {"clause_id": c.clause_id, "text": c.text, "clause_type": c.clause_type}
            for c in clauses_a
        ]

        aligner.index_doc_b(clauses_b_dicts)
        pairs = aligner.align(clauses_a_dicts, top_k=5)
        ANALYSIS_CACHE[analysis_id].update({"progress": 50, "message": "Alignment complete. Running NLI..."})

        # ── Phase 4: NLI Contradiction Detection ───────────────────────────────
        nli   = get_nli_inferencer()
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

        if nli:
            nli_results_obj = nli.predict_batch(pairs_as_dicts)
            nli_results     = [
                {**r.__dict__,
                 "clause_a_type": next(
                    (p["clause_a_type"] for p in pairs_as_dicts if p["pair_id"] == r.pair_id), "OTHER"),
                 "clause_b_type": next(
                    (p["clause_b_type"] for p in pairs_as_dicts if p["pair_id"] == r.pair_id), "OTHER"),
                }
                for r in nli_results_obj
            ]
        else:
            # Mock mode: flag all pairs with similarity > 0.75 as contradictions
            nli_results = []
            for p in pairs:
                is_flagged = p.similarity > 0.75
                nli_results.append({
                    "pair_id":            p.pair_id,
                    "clause_a_id":        p.clause_a_id,
                    "clause_b_id":        p.clause_b_id,
                    "clause_a_text":      p.clause_a_text,
                    "clause_b_text":      p.clause_b_text,
                    "clause_a_type":      p.clause_a_type,
                    "clause_b_type":      p.clause_b_type,
                    "predicted_label":    "CONTRADICTION" if is_flagged else "NOT_MENTIONED",
                    "confidence":         p.similarity,
                    "contradiction_prob": p.similarity if is_flagged else 0.0,
                    "is_flagged":         is_flagged,
                    "attention_tokens_a": [],
                    "attention_tokens_b": [],
                })

        ANALYSIS_CACHE[analysis_id].update({"progress": 75, "message": "NLI done. Scoring conflicts..."})

        # ── Phase 5: Severity Scoring ──────────────────────────────────────────
        reports = scorer.score(nli_results)

        elapsed = round(time.time() - start, 2)
        result_data = {
            "analysis_id":    analysis_id,
            "status":         "done",
            "total_conflicts": len(reports),
            "summary": {
                "CRITICAL": sum(1 for r in reports if r.severity_label == "CRITICAL"),
                "HIGH":     sum(1 for r in reports if r.severity_label == "HIGH"),
                "MEDIUM":   sum(1 for r in reports if r.severity_label == "MEDIUM"),
                "LOW":      sum(1 for r in reports if r.severity_label == "LOW"),
            },
            "conflicts": [
                {
                    "conflict_id":       r.conflict_id,
                    "conflict_type":     r.conflict_type,
                    "severity":          r.severity,
                    "severity_label":    r.severity_label,
                    "doc_a_clause":      r.doc_a_clause,
                    "doc_b_clause":      r.doc_b_clause,
                    "clause_a_type":     r.clause_a_type,
                    "clause_b_type":     r.clause_b_type,
                    "attention_tokens_a": r.attention_tokens_a,
                    "attention_tokens_b": r.attention_tokens_b,
                    "recommendation":    r.recommendation,
                    "nli_confidence":    r.nli_confidence,
                }
                for r in reports
            ],
            "metrics": {
                "clauses_extracted_a": len(clauses_a),
                "clauses_extracted_b": len(clauses_b),
                "candidate_pairs":     len(pairs),
                "contradictions":      len(reports),
            },
            "processing_time": elapsed,
        }

        # Cache and save
        ANALYSIS_CACHE[analysis_id] = {**result_data, "progress": 100}
        scorer.save_report(reports, RESULTS_DIR / f"{analysis_id}.json")

    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        ANALYSIS_CACHE[analysis_id] = {
            "status": "error", "progress": 0, "message": str(e)
        }


# ── API Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    nli = get_nli_inferencer()
    return {
        "status":      "ok",
        "nli_loaded":  nli is not None,
        "model":       "DeBERTa-v3-large (ContractNLI fine-tuned)" if nli else "Mock mode",
        "version":     "1.0.0",
    }


@app.post("/api/analyze", response_model=AnalysisStatus)
async def analyze(
    background_tasks: BackgroundTasks,
    doc_a: UploadFile = File(..., description="Contract or agreement document (.txt, .pdf, .docx)"),
    doc_b: UploadFile = File(..., description="Policy or regulation document (.txt, .pdf, .docx)"),
):
    """Upload two documents; returns an analysis_id to poll for results."""
    analysis_id = str(uuid.uuid4())

    raw_a  = await doc_a.read()
    raw_b  = await doc_b.read()
    text_a = extract_text_from_bytes(raw_a, doc_a.filename or "doc_a.txt")
    text_b = extract_text_from_bytes(raw_b, doc_b.filename or "doc_b.txt")

    MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    if len(raw_a) > MAX_UPLOAD_BYTES or len(raw_b) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // 1024 // 1024} MB per document.")
    if len(text_a) < 50 or len(text_b) < 50:
        raise HTTPException(status_code=400, detail="Documents too short (minimum 50 characters)")

    ANALYSIS_CACHE[analysis_id] = {
        "status": "pending", "progress": 0, "message": "Queued for analysis"
    }

    background_tasks.add_task(
        run_analysis, analysis_id, text_a, text_b,
        doc_a.filename or "Document A", doc_b.filename or "Document B"
    )

    return AnalysisStatus(
        analysis_id = analysis_id,
        status      = "pending",
        progress    = 0,
        message     = "Analysis started",
    )


@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Poll for analysis results. Returns status + results when done."""
    data = ANALYSIS_CACHE.get(analysis_id)
    if data is None:
        # Try loading from disk
        saved = RESULTS_DIR / f"{analysis_id}.json"
        if saved.exists():
            with open(saved) as f:
                return JSONResponse(content=json.load(f))
        raise HTTPException(status_code=404, detail="Analysis ID not found")
    return JSONResponse(content=data)


@app.get("/api/metrics")
async def get_metrics():
    """Returns evaluation metrics from the most recent validation run."""
    metrics_path = Path("output/eval_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {
        "message": "No evaluation metrics available yet. Run evaluation first.",
        "accuracy":  None,
        "precision": None,
        "recall":    None,
        "f1_score":  None,
    }
