# Usage Examples

This document provides practical examples for common use cases.

## Basic Usage

### Test Mode (50 records)

**Local execution:**
```bash
python cli_analysis.py --test
```

**Cloud execution (recommended for consistency):**
```bash
python cli_analysis.py --test --cloud
```

## Company Selection

### Specific Companies

**Single company:**
```bash
python cli_analysis.py --companies AAPL --test
```

**Multiple companies:**
```bash
python cli_analysis.py --companies AAPL,MSFT,GOOGL,AMZN --test
```

**Tech giants with date range:**
```bash
python cli_analysis.py --companies AAPL,MSFT,GOOGL,AMZN,META,NVDA \
  --start-date 2024-01-01 --end-date 2024-12-31 --mode full
```

### Using tickers.csv (Default)

The tool automatically uses all companies in `tickers.csv` when no companies are specified:

```bash
# All companies in tickers.csv, last 90 days
python cli_analysis.py --test

# All companies, specific date range
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-03-31 --mode full
```

### Custom Company File

Create your own CSV with a `symbol` column:

```csv
symbol
AAPL
MSFT
GOOGL
```

Then use it:
```bash
python cli_analysis.py --company-file my_companies.csv --mode full
```

## Date Filtering

### Recent Analysis

**Last 30 days:**
```bash
python cli_analysis.py --days-back 30 --test
```

**Last quarter (90 days):**
```bash
python cli_analysis.py --days-back 90 --mode full
```

### Specific Date Ranges

**Q1 2024:**
```bash
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-03-31 --mode full
```

**Full year 2024:**
```bash
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-12-31 --mode full
```

**Custom range:**
```bash
python cli_analysis.py --start-date 2024-06-01 --end-date 2024-08-31 --mode full
```

## Execution Modes

### Test Mode (50 records)

Quick validation before full runs:
```bash
python cli_analysis.py --test
```

Or with specific settings:
```bash
python cli_analysis.py --companies AAPL,MSFT --start-date 2024-01-01 --test
```

### Full Mode (All matching records)

Process all records matching your criteria:
```bash
python cli_analysis.py --mode full
```

### Custom Limit

Override default limits:
```bash
python cli_analysis.py --limit 100
python cli_analysis.py --limit 1000 --companies AAPL
```

## Local vs Cloud Execution

### Local Execution

Runs on your machine, writes to local CSV:

```bash
# Basic
python cli_analysis.py --test

# With BigQuery output
python cli_analysis.py --test --write-to-bq

# Without content (smaller files)
python cli_analysis.py --test --no-content
```

### Cloud Execution

Runs on Google Cloud Run, downloads CSV to local machine:

```bash
# Basic (uses default Cloud Run URL)
python cli_analysis.py --test --cloud

# Full analysis
python cli_analysis.py --mode full --cloud

# Specific companies
python cli_analysis.py --companies AAPL,MSFT,GOOGL --cloud

# With all parameters
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full \
  --cloud
```

**When to use cloud:**
- Large datasets (100+ companies, full year)
- Faster execution needed
- Local machine resource constraints
- Want to offload compute to Google infrastructure

## Sector Analysis

### Technology Sector

```bash
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL,AMZN,META,NVDA,ORCL,IBM,CSCO,INTC \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full \
  --cloud
```

### Energy Sector

```bash
python cli_analysis.py \
  --companies XOM,CVX,COP \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full
```

### Financial Sector

```bash
python cli_analysis.py \
  --companies JPM,BAC,WFC,C,GS,MS \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full
```

### Healthcare Sector

```bash
python cli_analysis.py \
  --companies UNH,JNJ,PFE,ABBV,MRK,LLY \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --mode full
```

## Common Workflows

### Weekly Update

Run every Monday to analyze last week's earnings calls:

```bash
python cli_analysis.py --days-back 7 --write-to-bq
```

### Monthly Report

First of the month - analyze previous month:

```bash
python cli_analysis.py --days-back 30 --mode full --write-to-bq
```

### Quarterly Deep Dive

End of quarter - full analysis:

```bash
python cli_analysis.py \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --mode full \
  --cloud
```

### Event-Driven Analysis

After major market event, analyze affected companies:

```bash
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL,AMZN \
  --days-back 7 \
  --cloud
```

## Advanced Usage

### Combining Multiple Filters

```bash
# Specific companies, date range, cloud execution, test first
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --test \
  --cloud

# If test looks good, run full mode
python cli_analysis.py \
  --companies AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --mode full \
  --cloud
```

### Batch Processing with Shell Script

Create `batch_analysis.sh`:

```bash
#!/bin/bash

# Tech sector
python cli_analysis.py --companies AAPL,MSFT,GOOGL --mode full --cloud

# Finance sector
python cli_analysis.py --companies JPM,BAC,GS --mode full --cloud

# Energy sector
python cli_analysis.py --companies XOM,CVX --mode full --cloud
```

### PowerShell Batch Processing

Create `batch_analysis.ps1`:

```powershell
# Tech sector
python cli_analysis.py --companies AAPL,MSFT,GOOGL --mode full --cloud

# Finance sector
python cli_analysis.py --companies JPM,BAC,GS --mode full --cloud

# Energy sector
python cli_analysis.py --companies XOM,CVX --mode full --cloud
```

## Output Management

### Local Output

Results saved to `outputs/` directory:
- `cli_analysis_results_YYYYMMDD_HHMMSS.csv` - Local execution
- `cloud_analysis_results_YYYYMMDD_HHMMSS.csv` - Cloud execution

### With BigQuery

Write to both local CSV and BigQuery:

```bash
python cli_analysis.py --test --write-to-bq
```

### Without Content (Smaller Files)

Exclude transcript content from output:

```bash
python cli_analysis.py --test --no-content
```

Good for:
- Quick topic/sentiment overview
- Reducing file size
- Faster processing

## Troubleshooting Examples

### Test Before Full Run

Always test first with new parameters:

```bash
# Test
python cli_analysis.py --companies AAPL --test

# If successful, run full
python cli_analysis.py --companies AAPL --mode full
```

### Verify Cloud Connection

```bash
# Simple cloud test
python cli_analysis.py --test --cloud --limit 10
```

### Check Specific Company Data

```bash
# Single company, recent data
python cli_analysis.py --companies AAPL --days-back 30 --test
```

### Debug Issues

```bash
# Small dataset for debugging
python cli_analysis.py --companies AAPL --limit 10
```

## Performance Tips

### Use Cloud for Large Jobs

Local (slower):
```bash
python cli_analysis.py --mode full --start-date 2024-01-01 --end-date 2024-12-31
```

Cloud (faster):
```bash
python cli_analysis.py --mode full --start-date 2024-01-01 --end-date 2024-12-31 --cloud
```

### Incremental Processing

Instead of processing all at once, break into smaller chunks:

```bash
# Q1
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-03-31 --mode full --cloud

# Q2
python cli_analysis.py --start-date 2024-04-01 --end-date 2024-06-30 --mode full --cloud

# Q3
python cli_analysis.py --start-date 2024-07-01 --end-date 2024-09-30 --mode full --cloud

# Q4
python cli_analysis.py --start-date 2024-10-01 --end-date 2024-12-31 --mode full --cloud
```

## Getting Help

View all available options:
```bash
python cli_analysis.py --help
```

Test your setup:
```bash
python cli_analysis.py --test
```

Check Cloud Run connectivity:
```bash
python cli_analysis.py --test --cloud --limit 5
```
