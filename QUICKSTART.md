# Quick Start Guide

## Installation

Run the automated setup script to get started:

```bash
python setup.py
```

This will install all dependencies, download models, and configure authentication.

## Overview

This tool analyzes earnings call transcripts with the ability to:
- Select specific companies or use your custom tickers.csv list
- Filter by date ranges
- Run in test mode (50 records) or full production mode
- Execute locally via CLI or leverage Google Cloud's compute via Cloud Run
- Get comprehensive output including ALL metadata from BigQuery

## Quick Examples

### 1. Test the Tool (Local CLI)

Run a quick test with 50 records from your tickers.csv:
```bash
python cli_analysis.py --test
```

### 1b. Test Using Cloud Run (Recommended for Large Datasets)

Execute on Google Cloud and download results:
```bash
python cli_analysis.py --test --cloud
```

### 2. Analyze Specific Companies

Analyze Apple, Microsoft, and Google for the last 90 days:
```bash
python cli_analysis.py --companies AAPL,MSFT,GOOGL
```

### 3. Date Range Analysis (Local)

Analyze all companies in tickers.csv for Q1 2024:
```bash
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-03-31 --mode full
```

### 3b. Date Range Analysis (Cloud)

Same analysis using Cloud Run for better performance:
```bash
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-03-31 --mode full --cloud
```

### 4. Custom Company List

Create a CSV file with your companies:
```csv
symbol,company_name
AAPL,Apple Inc.
MSFT,Microsoft Corporation
GOOGL,Alphabet Inc.
```

Then run:
```bash
python cli_analysis.py --company-file my_companies.csv --mode full
```

## Cloud Run Deployment

### Deploy to Google Cloud

1. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/earnings-analysis
   gcloud run deploy earnings-analysis \
     --image gcr.io/[PROJECT-ID]/earnings-analysis \
     --region us-central1 \
     --memory 4Gi \
     --timeout 3600
   ```

2. **Get the service URL:**
   ```bash
   SERVICE_URL=$(gcloud run services describe earnings-analysis --region us-central1 --format 'value(status.url)')
   echo $SERVICE_URL
   ```

### Trigger Analysis via Cloud Run

**Test mode (GET request):**
```bash
curl "$SERVICE_URL/run?mode=test"
```

**Specific companies (GET request):**
```bash
curl "$SERVICE_URL/run?companies=AAPL,MSFT,GOOGL&mode=test"
```

**Full configuration (POST request):**
```bash
curl -X POST "$SERVICE_URL/run" \
  -H "Content-Type: application/json" \
  -d '{
    "companies": "AAPL,MSFT,GOOGL,AMZN,NVDA",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "mode": "test"
  }'
```

**Full production run (POST request):**
```bash
curl -X POST "$SERVICE_URL/run" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "full",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

## Common Workflows

### Workflow 1: Weekly Analysis of Tech Giants

```bash
# Run every Monday for tech giants
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL,AMZN,META,NVDA \
  --days-back 7 \
  --write-to-bq
```

### Workflow 2: Quarterly Full Analysis

```bash
# End of quarter - full Fortune 100
python cli_analysis.py \
  --mode full \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --write-to-bq
```

### Workflow 3: Custom Sector Analysis

Create `energy_sector.csv`:
```csv
symbol,company_name
XOM,Exxon Mobil Corporation
CVX,Chevron Corporation
```

Run:
```bash
python cli_analysis.py \
  --company-file energy_sector.csv \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full
```

### Workflow 4: Cloud-Based Large-Scale Processing

Use Cloud Run for processing all companies (via CLI):
```bash
# Simple - uses default Cloud Run URL
python cli_analysis.py --mode full --start-date 2024-01-01 --end-date 2024-12-31 --cloud
```

Or via direct API call:
```bash
# Via curl
curl -X POST "https://earnings-call-transcript-analysis-434903546449.us-central1.run.app/run" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "full",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "return_csv": true
  }'
```

This leverages Google's compute resources for faster processing of large datasets.

## Parameters Reference

### CLI Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--companies` | Comma-separated stock symbols | `--companies AAPL,MSFT` |
| `--company-file` | CSV file with symbols | `--company-file my_list.csv` |
| `--start-date` | Start date (YYYY-MM-DD) | `--start-date 2024-01-01` |
| `--end-date` | End date (YYYY-MM-DD) | `--end-date 2024-12-31` |
| `--days-back` | Days back from today | `--days-back 30` |
| `--mode` | test (50) or full (all) | `--mode full` |
| `--test` | Shorthand for test mode | `--test` |
| `--limit` | Custom record limit | `--limit 100` |
| `--write-to-bq` | Write to BigQuery (local only) | `--write-to-bq` |
| `--no-content` | Exclude transcript text (local only) | `--no-content` |
| `--cloud` | Execute on Cloud Run (default URL) | `--cloud` |
| `--cloud-url` | Override Cloud Run URL (optional) | `--cloud-url https://...` |

### Cloud Run API Parameters

Same parameters available via:
- **GET**: Query string (`?companies=AAPL&mode=test`)
- **POST**: JSON payload (`{"companies": "AAPL", "mode": "test"}`)

## Output Files

Results are saved to `outputs/` directory:
- `cli_analysis_results_YYYYMMDD_HHMMSS.csv` - Timestamped analysis results
- Contains all detected topics, sentiment, roles, and Q&A clustering

## Tips

1. **Always test first**: Use `--test` or `--mode test` before running full analysis
2. **Date filtering**: Use date ranges to reduce scope and cost
3. **Company filtering**: Analyze specific sectors instead of all Fortune 100
4. **Cloud Run for scale**: Use Cloud Run for processing 100+ companies or large date ranges
5. **Monitor costs**: Check BigQuery costs when processing large datasets

## Troubleshooting

### No data found
- Check that companies exist in BigQuery metadata table
- Verify date range includes earnings calls
- Confirm company symbols are correct (use uppercase)

### Out of memory
- Reduce `--limit` or use smaller date ranges
- For large jobs, use Cloud Run with increased memory (4Gi+)

### Slow performance
- Use `--no-content` to reduce output file size
- Filter by specific companies instead of Fortune 100
- Use Cloud Run for parallel processing

## Next Steps

- Review output CSV to understand the schema
- Customize `fortune_100_companies.csv` for your specific needs
- Set up scheduled Cloud Run jobs for regular analysis
- Integrate with downstream analytics tools (Tableau, PowerBI, etc.)
