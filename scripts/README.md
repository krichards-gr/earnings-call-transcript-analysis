# scripts/

Utility, setup, diagnostic, and test scripts. Run all of them from the **project root**, not from inside this directory.

---

## Setup & Installation

| Script | Purpose | Example |
|---|---|---|
| `setup_utils.py` | Auto-installs missing Python dependencies and spaCy/embedding models if not present. Called by other scripts to ensure the environment is ready. | `python scripts/setup_utils.py` |
| `download_models.py` | Downloads ML models from GCS and Hugging Face into `models/`. Handles rate-limit retries. | `python scripts/download_models.py` |

---

## Verification & Diagnostics

| Script | Purpose | Example |
|---|---|---|
| `verify_parallel.py` | Checks that parallel processing is properly configured after installation. Quick sanity check. | `python scripts/verify_parallel.py` |
| `diagnose_transcripts.py` | Queries BigQuery to examine transcript content and classification results. Helps identify issues with Q&A session detection. | `python scripts/diagnose_transcripts.py` |
| `check_data.py` | Queries BigQuery to check what transcripts exist in a given date range. Used for troubleshooting data availability. | `python scripts/check_data.py` |

---

## Benchmarking

| Script | Purpose | Example |
|---|---|---|
| `benchmark_parallel.py` | Runs sequential vs. parallel processing back-to-back and reports the speedup ratio on your hardware. | `python scripts/benchmark_parallel.py --companies AAPL --limit 100` |

---

## Validation

| Script | Purpose | Example |
|---|---|---|
| `validate_changes.py` | Verifies Answer validation logic, session detection, and interaction type distribution in output data. | `python scripts/validate_changes.py` |

---

## Tests

| Script | Purpose | Example |
|---|---|---|
| `test_local_pipeline.py` | End-to-end test of the local analysis pipeline. Runs a small query and checks that output is generated. | `python scripts/test_local_pipeline.py` |
| `test_cloud_pipeline.py` | Tests the Cloud Run service health check and `/run` endpoint. | `python scripts/test_cloud_pipeline.py https://your-service.run.app` |
| `test_classification_order.py` | Unit test for the operator-intro stripping logic that runs before classification. | `python scripts/test_classification_order.py` |
| `test_edge_cases.py` | Unit tests for edge cases in interaction classification (e.g., operator intro merged with analyst question). | `python scripts/test_edge_cases.py` |
| `test_regex_patterns.py` | Tests session-boundary regex patterns against real transcript samples. | `python scripts/test_regex_patterns.py` |
| `test_cloud_api.sh` | Bash script for testing Cloud Run endpoints (health check, GET, POST). | `./scripts/test_cloud_api.sh https://your-service.run.app` |
| `test_cloud_api.ps1` | PowerShell equivalent of `test_cloud_api.sh`. | `.\scripts\test_cloud_api.ps1 https://your-service.run.app` |
