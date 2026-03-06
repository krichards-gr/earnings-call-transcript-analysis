# Earnings Call Transcript Analysis

Analyzes earnings call transcripts from BigQuery, detecting topics, assessing sentiment, and classifying speaker roles and interaction types. Outputs enriched CSV files suitable for downstream analysis.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Setup](#3-setup)
4. [Quick Start](#4-quick-start)
5. [CLI Reference](#5-cli-reference)
6. [Configuration](#6-configuration)
7. [Cloud Deployment](#7-cloud-deployment)
8. [Output Schema](#8-output-schema)
9. [Development](#9-development)

---

## 1. Overview

**What it does:**
- Pulls earnings call transcript segments from BigQuery
- Classifies each segment by speaker role (Analyst, Executive, Operator, Admin) and interaction type (Question, Answer, Admin)
- Detects relevant topics using exact keyword matching (spaCy) and semantic similarity (sentence-transformers)
- Scores sentiment per topic using VADER
- Groups Q&A exchanges into numbered sessions
- Writes enriched results to a local CSV (and optionally to BigQuery)

**Data source:** Google BigQuery — `sri-benchmarking-databases.pressure_monitoring`
- Source table: `earnings_call_transcript_content`
- Metadata table: `earnings_call_transcript_metadata`
- Output table: `earnings_call_transcript_enriched_local`

**Outputs:** Timestamped CSV files in `outputs/`

---

## 2. Architecture

```
issue_config_inputs_raw.csv
        │
        ▼
 generate_topics.py  ──────► topics.json
                                  │
                                  ▼
                            analyzer.py          (IssueAnalyzer: topic detection engine)
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
            cli_analysis.py              analysis.py
          (local CLI runner)         (Cloud Run HTTP server)
                    │
                    ▼
          parallel_analyzer.py      (multi-core processing wrapper)
```

| File | Role |
|---|---|
| `cli_analysis.py` | Main entry point for local and cloud-triggered analysis. Parses CLI args, queries BigQuery, orchestrates the pipeline, writes CSV output. |
| `analysis.py` | Cloud Run HTTP server. Exposes `/` (health check) and `/run` (POST/GET) endpoints. Same pipeline logic as `cli_analysis.py`. |
| `analyzer.py` | Core `IssueAnalyzer` class. Loads topic config, builds spaCy Matcher patterns, pre-computes anchor embeddings, detects topics in text. |
| `parallel_analyzer.py` | `ParallelAnalyzer` wrapper. Distributes classification and topic detection across CPU cores. Auto-detects optimal worker/batch settings. |
| `generate_topics.py` | Reads keyword inputs (from `--keyword-file`, `updated_issue_config_inputs.csv`, or `issue_config_inputs_raw.csv` via `--parse-raw`) and writes `topics.json`. Called automatically on startup. |
| `setup.py` | Interactive setup script. Creates virtualenv, installs deps, downloads models, configures GCP auth. |

---

## 3. Setup

### Prerequisites

- Python 3.8+
- Google Cloud SDK (`gcloud`) — for BigQuery access
- GCP project with BigQuery access to `sri-benchmarking-databases`

### Automated Setup (Recommended)

```bash
python setup.py
```

This will:
1. Check Python version
2. Create `.venv` virtual environment
3. Install dependencies from `requirements.txt`
4. Download spaCy `en_core_web_sm` model
5. Download ML models via `scripts/download_models.py`
6. Prompt for `gcloud auth application-default login`
7. Verify required core files exist

### Manual Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_models.py
gcloud auth application-default login
```

---

## 4. Quick Start

```bash
# 1. Test run — 50 records, last 90 days, Fortune 100 companies
python cli_analysis.py --test

# 2. Specific companies, last quarter
python cli_analysis.py --companies AAPL,MSFT,GOOGL --days-back 90

# 3. Full production run, specific date range
python cli_analysis.py --mode full --start-date 2025-01-01 --end-date 2025-03-31
```

Results are written to `outputs/cli_analysis_results_YYYYMMDD_HHMMSS.csv`.

---

## 5. CLI Reference

Run `python cli_analysis.py --help` for the full list. Key arguments:

### Company Selection

| Argument | Description | Default |
|---|---|---|
| `--companies AAPL,MSFT` | Comma-separated ticker symbols | — |
| `--company-file path.csv` | CSV file with a `symbol` column | — |
| *(neither)* | Uses all symbols from `tickers.csv` | Fortune 100 |

### Date Range

| Argument | Description | Default |
|---|---|---|
| `--start-date YYYY-MM-DD` | Start of date range | — |
| `--end-date YYYY-MM-DD` | End of date range | — |
| `--days-back N` | Last N days from today | `90` |
| `--latest N` | Pull the N most-recent complete transcripts | — |
| `--earliest N` | Pull the N oldest complete transcripts | — |

If neither `--start-date` nor `--end-date` is provided, defaults to the last `--days-back` days.

### Volume / Mode

| Argument | Description | Default |
|---|---|---|
| `--mode [test\|full]` | `test` = 50 records; `full` = all records | `test` |
| `--test` | Shorthand for `--mode test` | — |
| `--limit N` | Custom record cap (returns complete transcripts only) | — |

`--latest` / `--earliest` override `--limit` and `--mode`.

### Output

| Argument | Description |
|---|---|
| `--output path.csv` | Custom output path (default: `outputs/cli_analysis_results_TIMESTAMP.csv`) |
| `--write-to-bq` | Also write results to BigQuery destination table |
| `--no-content` | Omit transcript text from output (smaller files) |

### Execution Target

| Argument | Description |
|---|---|
| `--cloud` | Send request to Cloud Run instead of running locally |
| `--cloud-url URL` | Override default Cloud Run service URL |

### Parallelization (local only)

| Argument | Description |
|---|---|
| `--no-parallel` | Disable multi-core processing |
| `--workers N` | Number of worker processes (default: auto) |
| `--sentiment-batch-size N` | Batch size for VADER sentiment (default: auto) |
| `--classification-batch-size N` | Batch size for role/interaction classifiers (default: auto) |

---

## 6. Configuration

### Topic / Keyword Configuration

The pipeline supports three ways to supply keyword inputs, in order of precedence:

#### 1. `--keyword-file` (highest priority)

Point at any CSV in the intermediate format (`issue_area,topic,term,type`) to use it as the keyword config for that run:

```bash
python cli_analysis.py --keyword-file inputs/anti-us-sentiment-key-terms.csv --companies MCD,SBUX
```

The file must have the columns `issue_area`, `topic`, `term`, and `type` (same format as `updated_issue_config_inputs.csv`). This overrides the default config without touching any project files.

#### 2. `updated_issue_config_inputs.csv` (default intermediate format)

The default intermediate CSV used when `--keyword-file` is not supplied. Columns: `issue_area`, `topic`, `term`, `type`. Edit this file directly, then re-run without any extra flags.

#### 3. `issue_config_inputs_raw.csv` + `--parse-raw` (raw human-friendly format)

The human-readable source of truth. Each row defines one topic with:
- `issue_area` — broad category (e.g., "Tariffs & Trade")
- `issue_subtopic` — specific topic label
- `pattern` — semicolon-separated keyword patterns for exact matching
- `anchor_phrases` — semicolon-separated phrases for semantic similarity matching
- `exclusionary_term` — semicolon-separated terms to suppress false positives

Pass `--parse-raw` to transform this file into `updated_issue_config_inputs.csv` and regenerate `topics.json`:

```bash
python cli_analysis.py --parse-raw
# or regenerate standalone:
python generate_topics.py
```

`topics.json` is auto-regenerated on every startup of `cli_analysis.py` and `analysis.py`.

### Company List (`tickers.csv`)

Default company universe. Must have a `symbol` column. Replace or extend to target different companies. Pass `--company-file` to use an alternative file without modifying `tickers.csv`.

### Similarity Threshold

Defined in `cli_analysis.py` at the top:

```python
SIMILARITY_THRESHOLD = 0.7
```

Segments with cosine similarity to an anchor phrase below this value are not tagged with that topic.

---

## 7. Cloud Deployment

### Cloud Run Service

The `analysis.py` server exposes:
- `GET /` — health check, returns `200 OK`
- `GET /run?companies=AAPL,MSFT&mode=test` — trigger analysis via query params
- `POST /run` — trigger analysis via JSON body

**POST body parameters:**

| Parameter | Type | Description |
|---|---|---|
| `companies` | string | Comma-separated symbols (default: Fortune 100) |
| `start_date` | string | YYYY-MM-DD |
| `end_date` | string | YYYY-MM-DD |
| `mode` | string | `"test"` or `"full"` |
| `limit` | int | Max records |
| `latest` | int | N most-recent complete transcripts |
| `earliest` | int | N oldest complete transcripts |
| `return_csv` | bool | Return CSV in response body (default: false) |

### Build & Deploy

```bash
# Build and push image
gcloud builds submit --tag gcr.io/[PROJECT-ID]/earnings-analysis

# Deploy to Cloud Run
gcloud run deploy earnings-analysis \
  --image gcr.io/[PROJECT-ID]/earnings-analysis \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars PRODUCTION_MODE=true,BATCH_SIZE=500
```

The `cloudbuild.yaml` file automates build + deploy via Cloud Build triggers.

### Triggering from `cli_analysis.py`

```bash
# Use default Cloud Run URL (hardcoded in cli_analysis.py)
python cli_analysis.py --companies AAPL,MSFT --cloud

# Use a custom URL
python cli_analysis.py --companies AAPL,MSFT --cloud --cloud-url https://my-service.run.app
```

Authentication uses `gcloud auth print-identity-token` automatically.

---

## 8. Output Schema

Each row in the output CSV represents one transcript segment × one detected topic. Segments with no detected topic produce one row with null topic fields.

| Column | Description |
|---|---|
| `transcript_id` | Unique identifier for the earnings call transcript |
| `paragraph_number` | Sequential segment number within the transcript |
| `speaker` | Speaker name as it appears in the transcript |
| `qa_session_id` | Integer session counter; increments at each Q&A exchange boundary |
| `interaction_type` | `Question`, `Answer`, or `Admin` |
| `role` | `Analyst`, `Executive`, `Operator`, or `Admin` |
| `issue_area` | Broad topic category (from `issue_config_inputs_raw.csv`) |
| `issue_subtopic` | Specific topic label |
| `sentiment_label` | `positive`, `negative`, or `neutral` (VADER compound score) |
| `sentiment_score` | Absolute VADER compound score (0–1) |
| `all_scores` | Full VADER breakdown: `pos`, `neu`, `neg`, `compound` |
| `similarity_score` | Cosine similarity score (vector-match rows only; null for pattern-match rows) |
| `matched_anchor` | The anchor phrase that triggered the match (vector-match rows only) |
| `report_date` | Earnings call date from BigQuery metadata |
| `symbol` | Company ticker symbol |
| `content` | Original transcript text (omitted if `--no-content`) |

Additional columns from the BigQuery metadata table are appended as-is.

---

## 9. Development

Utility and test scripts live in `scripts/`. Run them from the project root:

```bash
python scripts/verify_parallel.py
python scripts/benchmark_parallel.py --companies AAPL --limit 100
python scripts/test_local_pipeline.py
```

See [`scripts/README.md`](scripts/README.md) for a full index of available scripts.

### ML Models

Downloaded to `models/` (not tracked in git):
- `models/all-MiniLM-L6-v2` — sentence embeddings
- `models/eng_type_class_v1` — interaction type classifier (custom)
- `models/role_class_v1` — speaker role classifier (custom)

Re-download with:
```bash
python scripts/download_models.py
```
