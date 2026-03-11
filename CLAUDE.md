# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for analyzing corporate earnings call transcripts. Detects topics (via spaCy pattern matching + sentence-transformer semantic similarity), classifies speaker roles and interaction types (custom HuggingFace models), and performs sentiment analysis (VADER). Reads from and writes to Google BigQuery.

## Setup & Running

```bash
# First-time setup (creates venv, installs deps, downloads models, configures GCP auth)
python setup.py

# Local analysis (default: Fortune 100, last 90 days, test mode = 50 records)
python cli_analysis.py --test

# Full run for specific companies
python cli_analysis.py --companies AAPL,MSFT --mode full --start-date 2025-01-01

# Cloud delegation
python cli_analysis.py --cloud --companies AAPL --test

# Regenerate topics.json from raw keyword config
python cli_analysis.py --parse-raw

# Custom keyword file
python cli_analysis.py --keyword-file inputs/anti-us-sentiment-key-terms.csv
```

## Running Tests

Tests live in `scripts/`. No test framework — each is a standalone Python script:

```bash
python scripts/test_local_pipeline.py      # End-to-end local pipeline
python scripts/test_cloud_pipeline.py      # Cloud Run endpoint checks
python scripts/test_classification_order.py # Operator intro stripping
python scripts/test_edge_cases.py          # Interaction classification edges
python scripts/test_regex_patterns.py      # Session boundary regex
python scripts/validate_changes.py         # Answer logic, session detection
python scripts/benchmark_parallel.py       # Sequential vs parallel speedup
```

## Architecture

### Entry Points
- **`cli_analysis.py`** — Main CLI (argparse). Handles company selection, date ranges, BigQuery queries, orchestrates analysis, writes output CSV and optionally to BigQuery.
- **`analysis.py`** — Cloud Run HTTP server (Flask-style). Exposes `GET /` (health) and `POST|GET /run` for cloud-triggered analysis.

### Core Pipeline
- **`analyzer.py`** — `IssueAnalyzer` class. Loads `topics.json`, builds spaCy Matcher patterns, pre-computes anchor embeddings, detects topics in text via dual-mode matching (exact patterns + semantic similarity at 0.7 threshold).
- **`parallel_analyzer.py`** — `ParallelAnalyzer` wraps classification and topic detection with multiprocessing. Auto-detects worker count (CPUs - 1). Disable with `--no-parallel`.
- **`generate_topics.py`** — Transforms keyword CSVs into `topics.json`. Two input paths: raw (`issue_config_inputs_raw.csv` with `--parse-raw`) or intermediate (`updated_issue_config_inputs.csv`).

### Data Flow
1. Load companies from CSV (`tickers.csv` default) or `--companies` flag
2. Query BigQuery (`sri-benchmarking-databases.pressure_monitoring`) with date/company filters
3. Rejoin fragmented transcript segments
4. Classify interaction type (Question/Answer/Admin) and speaker role (Analyst/Executive/Operator/Admin)
5. Detect topics per segment (pattern match + semantic similarity, with exclusion filtering)
6. Group Q&A sessions by regex-detected boundaries
7. Compute VADER sentiment
8. Output: CSV to `outputs/` and optionally BigQuery (`earnings_call_transcript_enriched_local`)

### Keyword Configuration Precedence
1. `--keyword-file path.csv` (runtime override)
2. `updated_issue_config_inputs.csv` (default intermediate format)
3. `issue_config_inputs_raw.csv` + `--parse-raw` (regenerate from human-readable source)

### ML Models (in `models/`)
- `all-MiniLM-L6-v2` — Sentence embeddings (384-dim) for semantic similarity
- `eng_type_class_v1` — Interaction type classifier (custom)
- `role_class_v1` — Speaker role classifier (custom)

All models run CPU-only (`CUDA_VISIBLE_DEVICES=-1`, torch limited to 2 threads).

## BigQuery Tables

Project: `sri-benchmarking-databases`, Dataset: `pressure_monitoring`
- `earnings_call_transcript_content` — Source transcripts
- `earnings_call_transcript_metadata` — Earnings dates, company info
- `earnings_call_transcript_enriched_local` — Output results
- `corporation_reference` — Company names, sectors

GCP auth requires both Cloud Platform and Drive scopes (some BQ tables are Drive-backed).

## Deployment

Cloud Build (`cloudbuild.yaml`): downloads models from GCS → builds Docker image (Python 3.11-slim) → deploys to Cloud Run (4Gi memory, 2 CPU, 3600s timeout).
