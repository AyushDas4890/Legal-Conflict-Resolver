"""
Phase 1: Data Acquisition
=========================
Downloads CUAD, ContractNLI, MAUD from HuggingFace
and scrapes SEC EDGAR 10-K risk factor sections.

Usage:
    python data_ingestion.py --all
    python data_ingestion.py --cuad --contract-nli
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger

# ── SEC EDGAR ──────────────────────────────────────────────────────────────────
try:
    from sec_edgar_downloader import Downloader
    SEC_AVAILABLE = True
except ImportError:
    SEC_AVAILABLE = False
    logger.warning("sec-edgar-downloader not installed. SEC data unavailable.")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"
EDA_DIR  = DATA_DIR / "eda"

DATASETS_CONFIG = {
    "cuad":         {"hf_path": "cuad",                           "subset": None},
    "contract_nli": {"hf_path": "lcampillos/contract-nli",        "subset": None},
    "maud":         {"hf_path": "theatticusproject/maud",          "subset": None},
}

SEC_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "BAC",
    "TSLA", "META", "NVDA"
]


def setup_dirs():
    for d in [RAW_DIR, EDA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories ready: {DATA_DIR.resolve()}")


def download_hf_dataset(name: str, config: dict) -> dict:
    """Download a HuggingFace dataset and save splits as parquet files."""
    out_dir = RAW_DIR / name
    out_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading {name} from HuggingFace...")
    try:
        ds = load_dataset(config["hf_path"], config["subset"], trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        return {}

    split_info = {}
    for split_name, split_data in ds.items():
        out_path = out_dir / f"{split_name}.parquet"
        split_data.to_pandas().to_parquet(out_path, index=False)
        split_info[split_name] = {
            "rows": len(split_data),
            "columns": split_data.column_names,
            "path": str(out_path)
        }
        logger.success(f"  [{name}] {split_name}: {len(split_data):,} rows → {out_path}")

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(split_info, f, indent=2)

    return split_info


def download_sec_edgar(tickers: list[str], form_type: str = "10-K", limit: int = 3):
    """Download SEC EDGAR filings for a list of tickers."""
    if not SEC_AVAILABLE:
        logger.warning("Skipping SEC EDGAR download (library not installed).")
        return

    sec_dir = RAW_DIR / "sec_edgar"
    sec_dir.mkdir(exist_ok=True)

    dl = Downloader("NLP Project", "nlp@project.edu", str(sec_dir))
    logger.info(f"Downloading {form_type} filings for {len(tickers)} tickers...")

    for ticker in tqdm(tickers, desc="SEC EDGAR"):
        try:
            dl.get(form_type, ticker, limit=limit, download_details=True)
            logger.success(f"  {ticker}: downloaded {limit} {form_type} filing(s)")
        except Exception as e:
            logger.warning(f"  {ticker}: failed — {e}")


def compute_eda_stats(name: str, split: str = "train") -> dict:
    """
    Compute EDA statistics for a dataset split.
    Reports the four key metrics basis:
      - Label distribution (class frequencies)
      - Text length distribution (clause/sentence level)
      - Vocabulary size
      - Class balance ratio
    """
    parquet_path = RAW_DIR / name / f"{split}.parquet"
    if not parquet_path.exists():
        logger.warning(f"EDA skipped for {name}/{split}: file not found.")
        return {}

    df = pd.read_parquet(parquet_path)
    stats = {
        "dataset": name,
        "split": split,
        "total_rows": len(df),
        "columns": list(df.columns),
    }

    # ── Text length stats ───────────────────────────────────────────────────────
    text_cols = [c for c in df.columns if df[c].dtype == object]
    for col in text_cols[:3]:  # Limit to first 3 text columns
        lengths = df[col].dropna().str.split().str.len()
        stats[f"{col}_word_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "std":  float(lengths.std()),
            "min":  int(lengths.min()),
            "max":  int(lengths.max()),
        }

    # ── Label distribution ──────────────────────────────────────────────────────
    label_cols = [c for c in df.columns if "label" in c.lower() or "answer" in c.lower()]
    for col in label_cols[:3]:
        dist = df[col].value_counts(normalize=True).to_dict()
        stats[f"{col}_distribution"] = {str(k): round(v, 4) for k, v in dist.items()}

    eda_path = EDA_DIR / f"{name}_{split}_stats.json"
    with open(eda_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.success(f"EDA stats saved: {eda_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Legal-Financial Conflict Resolver — Data Ingestion")
    parser.add_argument("--all",          action="store_true", help="Download all datasets")
    parser.add_argument("--cuad",         action="store_true", help="Download CUAD dataset")
    parser.add_argument("--contract-nli", action="store_true", help="Download ContractNLI dataset")
    parser.add_argument("--maud",         action="store_true", help="Download MAUD dataset")
    parser.add_argument("--sec",          action="store_true", help="Download SEC EDGAR filings")
    parser.add_argument("--eda",          action="store_true", help="Run EDA on downloaded data")
    args = parser.parse_args()

    setup_dirs()
    summary = {}

    targets = {
        "cuad":         args.all or args.cuad,
        "contract_nli": args.all or args.contract_nli,
        "maud":         args.all or args.maud,
    }

    for name, should_download in targets.items():
        if should_download:
            info = download_hf_dataset(name, DATASETS_CONFIG[name])
            summary[name] = info

    if args.all or args.sec:
        download_sec_edgar(SEC_TICKERS)

    if args.all or args.eda:
        logger.info("Running EDA...")
        for name in DATASETS_CONFIG:
            stats = compute_eda_stats(name)
            if stats:
                logger.info(
                    f"  {name} — {stats.get('total_rows', 0):,} rows, "
                    f"columns: {stats.get('columns', [])}"
                )

    # ── Final summary ───────────────────────────────────────────────────────────
    summary_path = DATA_DIR / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Download complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
