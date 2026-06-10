# ── Legal Conflict Resolver — Docker Build ─────────────────────────────────────
# Usage:
#   docker build -t legal-conflict-resolver .
#   docker run -p 8000:8000 -e CORS_ORIGINS="http://localhost:5173" legal-conflict-resolver

FROM python:3.11-slim AS base

# System deps for spaCy / torch / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Build frontend (optional — requires Node) ─────────────────────────────────
# Uncomment if you want to serve the React dashboard from FastAPI:
# FROM node:20-slim AS frontend
# WORKDIR /app/dashboard
# COPY dashboard/package*.json ./
# RUN npm ci
# COPY dashboard/ .
# RUN npm run build
# COPY --from=frontend /app/dashboard/dist /app/dashboard/dist

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# CORS_ORIGINS: comma-separated list of allowed frontend origins
# MAX_UPLOAD_MB: max file size per document (default 10)
# MODEL_DIR: path to fine-tuned DeBERTa checkpoint (optional, uses mini model otherwise)
ENV CORS_ORIGINS="http://localhost:5173,http://localhost:3000" \
    MAX_UPLOAD_MB="10" \
    PYTHONUNBUFFERED="1"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
