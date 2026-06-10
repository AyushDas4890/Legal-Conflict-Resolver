<div align="center">

```
██╗     ███████╗ ██████╗  █████╗ ██╗      ██████╗ ██████╗ ███╗   ██╗███████╗██╗     ██╗ ██████╗████████╗
██║     ██╔════╝██╔════╝ ██╔══██╗██║     ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║     ██║██╔════╝╚══██╔══╝
██║     █████╗  ██║  ███╗███████║██║     ██║     ██║   ██║██╔██╗ ██║█████╗  ██║     ██║██║        ██║   
██║     ██╔══╝  ██║   ██║██╔══██║██║     ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║     ██║██║        ██║   
███████╗███████╗╚██████╔╝██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚████║██║     ███████╗██║╚██████╗   ██║   
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝ ╚═════╝   ╚═╝   
                                                                                                         
██████╗ ███████╗███████╗ ██████╗ ██╗    ██╗   ██╗███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝██╔═══██╗██║    ██║   ██║██╔════╝██╔══██╗
██████╔╝█████╗  ███████╗██║   ██║██║    ██║   ██║█████╗  ██████╔╝
██╔══██╗██╔══╝  ╚════██║██║   ██║██║    ╚██╗ ██╔╝██╔══╝  ██╔══██╗
██║  ██║███████╗███████║╚██████╔╝███████╗╚████╔╝ ███████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝ ╚═══╝  ╚══════╝╚═╝  ╚═╝
```

**AI-powered contradiction detection between legal & financial documents**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-website--orpin--chi--25.vercel.app-f59e0b?style=for-the-badge&logo=vercel&logoColor=white)](https://website-orpin-chi-25.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

---

> Drop two legal documents. Get back every contradiction, ranked by severity, with explainable evidence — in under 3 seconds.

</div>

---

## ⚡ What It Does

Contracts and policies contradict each other all the time. Lawyers miss it. LLMs hallucinate it. This system **finds contradictions with mathematical precision** using Natural Language Inference.

Upload a **contract** + a **policy/regulation**. The pipeline:

1. Extracts every legal clause from both documents (spaCy + rule-based)
2. Semantically aligns related clauses across documents (FAISS vector search)  
3. Runs NLI contradiction detection on each aligned pair (DeBERTa cross-encoder)
4. Scores every conflict: **CRITICAL → HIGH → MEDIUM → LOW**
5. Returns structured JSON with attention-highlighted evidence

No summaries. No guesswork. Pure discriminative classification.

---

## 🚀 Try It Live

**→ [website-orpin-chi-25.vercel.app](https://website-orpin-chi-25.vercel.app)**

Upload two `.txt`, `.pdf`, or `.docx` files and watch the analysis run in real time.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Next.js 14)                          │
│           Drag & Drop  ──►  Progress Poll  ──►  Conflict View       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ POST /api/analyze (multipart)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                                │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   Phase 2    │   │   Phase 3    │   │       Phase 4         │   │
│  │   Clause     │──►│  Semantic    │──►│  NLI Contradiction    │   │
│  │  Extraction  │   │  Alignment   │   │     Detection         │   │
│  │  (spaCy +    │   │  (FAISS +    │   │ (DeBERTa cross-       │   │
│  │  rule-based) │   │  sentence-   │   │  encoder, top_k=5)    │   │
│  └──────────────┘   │ transformers)│   └──────────┬───────────┘   │
│                     └──────────────┘               │               │
│                                                     ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       Phase 5                               │   │
│  │               Severity Scoring Engine                       │   │
│  │   CRITICAL (0.90+) │ HIGH (0.75+) │ MEDIUM (0.60+) │ LOW   │   │
│  └─────────────────────────────┬───────────────────────────────┘   │
│                                 │                                   │
│              GET /api/results/{id}  (polling)                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ JSON conflict report
                                  ▼
                         ┌─────────────────┐
                         │  output/*.json  │
                         │  (persisted)    │
                         └─────────────────┘
```

<details>
<summary><b>📦 Module Breakdown</b> (click to expand)</summary>

| File | Role | Key Dependency |
|------|------|----------------|
| `api.py` | FastAPI server, REST endpoints, async pipeline dispatch | fastapi, uvicorn |
| `clause_extractor.py` | Legal clause segmentation + type tagging | spaCy en_core_web_sm |
| `semantic_aligner.py` | FAISS-indexed semantic search, top-k pair generation | sentence-transformers, faiss-cpu |
| `nli_model.py` | DeBERTa fine-tuning + inference, attention extraction | transformers, torch |
| `severity_scorer.py` | Conflict severity classification + recommendations | — |
| `pipeline.py` | CLI entrypoint, orchestrates all 5 phases | all above |
| `data_ingestion.py` | PDF/DOCX/TXT text extraction | pypdf, python-docx |
| `threshold_tune.py` | NLI confidence threshold calibration | scikit-learn |
| `website/` | Next.js 14 frontend (App Router) | framer-motion, Tailwind |

</details>

---

## 🧠 The Model

The NLI brain is a **cross-encoder** — both clauses go in together, so the model sees their interaction directly (unlike bi-encoders that encode separately).

```
┌──────────────────────────────────────────────┐
│  [CLS] Clause A [SEP] Clause B [SEP]         │
│                    │                          │
│            DeBERTa Cross-Encoder             │
│         (cross-encoder/nli-deberta-v3-base)   │
│                    │                          │
│     ┌──────────────┼──────────────┐           │
│     ▼              ▼              ▼           │
│  CONTRADICTION  ENTAILMENT    NEUTRAL         │
│   (conflict!)   (consistent)  (unrelated)     │
└──────────────────────────────────────────────┘
```

**Anti-hallucination design:**
- Minimum confidence threshold: **0.70** for contradiction labeling
- Purely **discriminative** — no text generation, no hallucination risk
- Label smoothing (ε=0.1) during fine-tuning for calibrated probabilities
- Low-RAM fallback: `cross-encoder/nli-miniLM-L6-v2` on Render/constrained envs

---

## 🎯 Conflict Severity Levels

| Severity | Score | What It Means | Example |
|----------|-------|---------------|---------|
| 🔴 **CRITICAL** | ≥ 0.90 | Direct legal contradiction, compliance risk | Contract allows X; policy prohibits X |
| 🟠 **HIGH** | ≥ 0.75 | Significant conflict requiring review | Conflicting liability caps |
| 🟡 **MEDIUM** | ≥ 0.60 | Potential ambiguity or partial contradiction | Overlapping jurisdiction clauses |
| 🟢 **LOW** | < 0.60 | Minor inconsistency, likely harmless | Different terminology for same concept |

---

## 🛠️ Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/AyushDas4890/Legal-Conflict-Resolver.git
cd Legal-Conflict-Resolver
docker build -t legal-resolver .
docker run -p 8000:8000 legal-resolver
```

The API is now live at `http://localhost:8000`.

### Option 2 — Manual Setup

```bash
# 1. Clone
git clone https://github.com/AyushDas4890/Legal-Conflict-Resolver.git
cd Legal-Conflict-Resolver

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Start the API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3 — Frontend Only (point at existing API)

```bash
cd website
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# → http://localhost:3000
```

---

## 📡 API Reference

### `POST /api/analyze`
Upload two documents to start analysis.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "doc_a=@contract.pdf" \
  -F "doc_b=@policy.docx"
```

**Response:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "progress": 0,
  "message": "Analysis started"
}
```

---

### `GET /api/results/{analysis_id}`
Poll for results. Returns live progress (0–100) until complete.

```bash
curl http://localhost:8000/api/results/550e8400-e29b-41d4-a716-446655440000
```

<details>
<summary><b>Full response shape</b></summary>

```json
{
  "analysis_id": "550e8400-...",
  "status": "done",
  "total_conflicts": 3,
  "summary": {
    "CRITICAL": 1,
    "HIGH": 1,
    "MEDIUM": 1,
    "LOW": 0
  },
  "conflicts": [
    {
      "conflict_id": "conflict_001",
      "conflict_type": "CONTRADICTION",
      "severity": 0.94,
      "severity_label": "CRITICAL",
      "doc_a_clause": "The contractor shall not be liable for any damages...",
      "doc_b_clause": "All parties are fully liable for direct and indirect damages...",
      "clause_a_type": "LIABILITY",
      "clause_b_type": "LIABILITY",
      "attention_tokens_a": ["not", "liable", "damages"],
      "attention_tokens_b": ["fully", "liable", "direct", "indirect"],
      "recommendation": "Reconcile liability clauses — direct contradiction detected.",
      "nli_confidence": 0.94
    }
  ],
  "metrics": {
    "clauses_extracted_a": 24,
    "clauses_extracted_b": 31,
    "candidate_pairs": 87,
    "contradictions": 3
  },
  "processing_time": 2.47
}
```

</details>

---

### `GET /api/health`
Check model load status.

```bash
curl http://localhost:8000/api/health
# {"status":"ok","nli_loaded":true,"model":"DeBERTa-v3-large (ContractNLI fine-tuned)","version":"1.0.0"}
```

---

## 🔬 How the Pipeline Works

<details>
<summary><b>Phase 2: Clause Extraction</b></summary>

spaCy + custom rule-based extractor segments each document into atomic legal clauses. Each clause is tagged with its type (LIABILITY, PAYMENT, TERMINATION, INDEMNIFICATION, JURISDICTION, etc.).

```python
clauses_a = extractor.extract(text_a, source_doc="contract.pdf")
# Returns: [Clause(clause_id="A_001", text="...", clause_type="LIABILITY"), ...]
```

</details>

<details>
<summary><b>Phase 3: Semantic Alignment</b></summary>

FAISS-indexed vector search finds the `top_k=5` most semantically similar clause pairs across both documents. This ensures we only run expensive NLI on relevant pairs — not every possible combination.

```
Doc A: 24 clauses × Doc B: 31 clauses = 744 possible pairs
After FAISS alignment: 87 candidate pairs (88% reduction)
```

</details>

<details>
<summary><b>Phase 4: NLI Contradiction Detection</b></summary>

Each aligned pair is fed to the DeBERTa cross-encoder. The model outputs probabilities across three labels:

| Label | Meaning |
|-------|---------|
| CONTRADICTION | The clauses conflict — one cannot be true if the other is |
| ENTAILMENT | The clauses are consistent |
| NOT_MENTIONED | The clauses are unrelated |

Only pairs with `contradiction_prob ≥ 0.70` are flagged as conflicts.

</details>

<details>
<summary><b>Phase 5: Severity Scoring</b></summary>

Flagged contradictions are scored using a multi-factor model combining:
- NLI confidence score
- Clause type criticality (LIABILITY > PAYMENT > TERMINATION > ...)
- Semantic similarity of the conflicting terms

Each conflict gets a recommendation string for legal review.

</details>

---

## 📁 Supported File Formats

| Format | Extension | Library |
|--------|-----------|---------|
| PDF | `.pdf` | pypdf 4.x |
| Word Document | `.docx` | python-docx |
| Plain Text | `.txt` | built-in |

Maximum upload size: **10 MB per document** (configurable via `MAX_UPLOAD_MB` env var).

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Allowed frontend origins |
| `MAX_UPLOAD_MB` | `10` | Max document size in MB |
| `RENDER` | — | Set to `1` on Render.com for low-RAM model |
| `PORT` | `8000` | API server port |

---

## 🛣️ Roadmap

- [ ] **Batch analysis** — compare one contract against multiple policies at once
- [ ] **Clause-level highlighting** — visual diff view in the frontend
- [ ] **Export to PDF** — download conflict report with annotations
- [ ] **Fine-tuned model** on ContractNLI / CUAD dataset
- [ ] **Webhook support** — push results to Slack/email when analysis completes
- [ ] **Multi-language** — extend beyond English legal text

---

## 🤝 Contributing

```bash
# Fork → clone → branch
git checkout -b feature/your-feature

# Make changes, then
git commit -m "feat: your feature description"
git push origin feature/your-feature
# Open a PR on GitHub
```

Issues and PRs are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT © [AyushDas4890](https://github.com/AyushDas4890)

---

<div align="center">

**Built with DeBERTa · FastAPI · Next.js · FAISS · spaCy**

[Live Demo](https://website-orpin-chi-25.vercel.app) · [Report Bug](https://github.com/AyushDas4890/Legal-Conflict-Resolver/issues) · [Request Feature](https://github.com/AyushDas4890/Legal-Conflict-Resolver/issues)

</div>
