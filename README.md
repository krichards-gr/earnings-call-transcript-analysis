# Earnings Call Transcript Analysis

This tool performs deep analysis on earnings call transcripts by identifying topics, assessing sentiment, and classifying speaker roles and interaction types. It also clusters question-and-answer sequences for better traceability. It now includes enhanced metadata mapping for **Issue Areas** and **Subtopics**.

## Features

*   **Topic Detection**:
    *   **Exact Match**: Uses spaCy match patterns to find specific keywords.
    *   **Vector Similarity**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic matching.
    *   **Metadata Enrichment**: Automatically maps detected topics to their broader **Issue Area** and **Subtopic** based on the configuration.
*   **Aspect-Based Sentiment Analysis (ABSA)**:
    *   Uses `yangheng/deberta-v3-base-absa-v1.1` to determine sentiment toward detected topics.
*   **Speaker & Role Classification**:
    *   **Role**: Classifies speakers as **Analyst**, **Executive**, **Operator**, or **Admin**.
    *   **Interaction Type**: Classifies segments as **Admin**, **Answer**, or **Question**.
*   **Q&A Clustering (Robust)**:
    *   Automatically groups questions and their corresponding answers into "Sessions" using a resilient regex-based detection system that handles varied operator phrasing and classification noise.
*   **Optimized Batch Processing**:
    *   Both local and cloud pipelines use vectorized batch inference for classification and sentiment analysis, significantly improving performance.
*   **Data Integrity**:
    *   **Fragment Rejoining**: Automatically rejoins segments split by line breaks to ensure models process complete statements.
*   **Company & Date Filtering**:
    *   Select specific companies or use Fortune 100 default list
    *   Filter by date ranges for targeted analysis
    *   Test mode (50 records) or full production mode

## Usage

### Enhanced CLI Tool (`cli_analysis.py`)

The primary interface for running analysis with full control over companies, date ranges, and processing limits. Can execute locally or send requests to Cloud Run.

**Basic Usage:**
```bash
# Test mode with default companies from tickers.csv (50 records)
python cli_analysis.py --test

# Specific companies, last 90 days
python cli_analysis.py --companies AAPL,MSFT,GOOGL

# Full analysis for all companies in tickers.csv, specific date range
python cli_analysis.py --mode full --start-date 2024-01-01 --end-date 2024-12-31

# Custom company file with symbols
python cli_analysis.py --company-file my_companies.csv --mode full

# Write results to both CSV and BigQuery
python cli_analysis.py --companies AAPL,MSFT --write-to-bq

# Execute on Cloud Run and download results (uses default Cloud Run URL)
python cli_analysis.py --companies AAPL,MSFT,GOOGL --cloud

# Or specify a custom Cloud Run URL
python cli_analysis.py --companies AAPL,MSFT,GOOGL --cloud --cloud-url https://custom-service.run.app
```

**CLI Arguments:**
- `--companies AAPL,MSFT,GOOGL` - Comma-separated list of stock symbols
- `--company-file path/to/file.csv` - CSV file with 'symbol' column (defaults to tickers.csv)
- `--start-date YYYY-MM-DD` - Start date for analysis
- `--end-date YYYY-MM-DD` - End date for analysis
- `--days-back N` - Analyze last N days (default: 90)
- `--mode [test|full]` - Processing mode (default: test = 50 records)
- `--test` - Shorthand for `--mode test`
- `--limit N` - Custom record limit
- `--write-to-bq` - Write results to BigQuery in addition to CSV (local execution only)
- `--no-content` - Exclude transcript content from output (smaller files, local execution only)
- `--cloud` - Execute analysis on Cloud Run instead of locally (uses default URL)
- `--cloud-url URL` - Override default Cloud Run service URL (optional)

**Examples:**
```bash
# Tech giants, last quarter (local)
python cli_analysis.py --companies AAPL,MSFT,GOOGL,AMZN,META --days-back 90

# Energy sector, full year 2024 (local)
python cli_analysis.py --companies XOM,CVX --start-date 2024-01-01 --end-date 2024-12-31 --mode full

# Test with custom limit (local)
python cli_analysis.py --companies AAPL --limit 100

# Cloud execution - leverage Google's compute (default URL)
python cli_analysis.py --companies AAPL,MSFT,GOOGL --mode full --cloud

# Cloud execution - all tickers, specific dates
python cli_analysis.py --start-date 2024-01-01 --end-date 2024-12-31 --cloud
```

### Legacy Local Analysis (`local_analysis.py`)

Original local analysis script with hardcoded configuration:

```bash
python local_analysis.py
```

**Testing:**
```bash
python test_local_pipeline.py
```

### Production Deployment (Cloud Run)

The production pipeline (`analysis.py`) is designed for Cloud Run and supports both GET and POST requests with the same parameters as the CLI tool.

**Health Check:**
```bash
curl https://[YOUR-SERVICE-URL]/
```

**Trigger via GET (Query Parameters):**
```bash
# Test mode with default Fortune 100
curl "https://[YOUR-SERVICE-URL]/run?mode=test"

# Specific companies
curl "https://[YOUR-SERVICE-URL]/run?companies=AAPL,MSFT,GOOGL&mode=test"

# Date range
curl "https://[YOUR-SERVICE-URL]/run?start_date=2024-01-01&end_date=2024-12-31&mode=test"
```

**Trigger via POST (JSON Payload):**
```bash
# Full configuration
curl -X POST "https://[YOUR-SERVICE-URL]/run" \
  -H "Content-Type: application/json" \
  -d '{
    "companies": "AAPL,MSFT,GOOGL,AMZN,NVDA",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "mode": "test",
    "limit": 100
  }'

# Full mode - Fortune 100
curl -X POST "https://[YOUR-SERVICE-URL]/run" \
  -H "Content-Type: application/json" \
  -d '{"mode": "full"}'
```

**API Parameters:**
- `companies` - Comma-separated string of stock symbols (default: Fortune 100)
- `start_date` - Start date in YYYY-MM-DD format
- `end_date` - End date in YYYY-MM-DD format
- `mode` - "test" (50 records) or "full" (all records)
- `limit` - Custom record limit (overrides mode)

**Testing Scripts:**
```bash
# Bash/Linux/Mac
./test_cloud_api.sh [YOUR-SERVICE-URL]

# PowerShell/Windows
.\test_cloud_api.ps1 [YOUR-SERVICE-URL]

# Python
python test_cloud_pipeline.py [YOUR-SERVICE-URL]
```

## Configuration Files

### Company Tickers (`tickers.csv`)
Default company list used when no specific companies are provided. Contains stock symbols for companies to analyze. Customize this file to match your specific use case.

### Issue Configuration (`issue_config_inputs_raw.csv`)
Defines topics, patterns, and exclusionary terms for issue detection.

### Topic Definitions (`topics.json`)
Auto-generated from `issue_config_inputs_raw.csv` via `generate_topics.py`.

## Configuration Variables

### Environment Variables (Cloud Run)
- `BATCH_SIZE` - Number of segments to process per BigQuery cycle (default: 500)
- `PRODUCTION_MODE` - Set to "true" for full run, "false" for testing (default: false)
- `PORT` - Server port (default: 8080)

### BigQuery Tables
- `BQ_SOURCE_TABLE` - Source transcript content
- `BQ_METADATA_TABLE` - Transcript metadata (report_date, symbol, etc.)
- `BQ_DEST_TABLE` - Enriched analysis results

## Project Structure

```
├── cli_analysis.py              # Enhanced CLI tool with company/date selection
├── analysis.py                  # Cloud Run production pipeline (supports GET/POST)
├── local_analysis.py            # Legacy local analysis script
├── analyzer.py                  # Core analyzer class for topic detection
├── generate_topics.py           # Topic configuration generator
├── download_models.py           # Model download utility
├── fortune_100_companies.csv    # Default company list
├── issue_config_inputs_raw.csv  # Issue configuration
├── topics.json                  # Auto-generated topic definitions
├── test_cloud_api.sh            # Bash test script for Cloud Run
├── test_cloud_api.ps1           # PowerShell test script for Cloud Run
├── test_cloud_pipeline.py       # Python test script for Cloud Run
├── test_local_pipeline.py       # Test script for local pipeline
├── models/                      # Pre-downloaded ML models
└── outputs/                     # Analysis results (CSV files)
```

## Installation

### Automated Setup (Recommended)

Run the automated setup script which handles all installation steps:

```bash
python setup.py
```

The setup script will:
- Check Python version (3.8+ required)
- Create a virtual environment
- Install all dependencies
- Download spaCy models
- Download ML models
- Configure Google Cloud authentication (optional)
- Verify all project files

### Manual Setup

If you prefer manual installation:

1.  **Create virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **spaCy Language Model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **ML Models**:
    ```bash
    python download_models.py
    ```

5.  **Google Cloud Authentication** (for BigQuery access):
    ```bash
    gcloud auth application-default login
    ```

## Output

Results are saved as CSV files in the `outputs/` directory with timestamps:
- `cli_analysis_results_YYYYMMDD_HHMMSS.csv` - CLI tool output
- `local_analysis_results.csv` - Legacy local analysis output

### Output Schema
- `transcript_id` - Unique transcript identifier
- `paragraph_number` - Segment number within transcript
- `speaker` - Speaker name
- `qa_session_id` - Q&A session number
- `qa_session_label` - Analyst name for session
- `interaction_type` - Admin/Answer/Question
- `role` - Admin/Analyst/Executive/Operator
- `topic` - Detected topic/issue
- `issue_area` - Broader issue category
- `issue_subtopic` - Specific issue subcategory
- `sentiment_label` - Positive/Negative/Neutral
- `sentiment_score` - Confidence score
- `all_scores` - Detailed sentiment scores
- `similarity_score` - Vector similarity score (if applicable)
- `matched_anchor` - Matched anchor phrase (if applicable)
- `report_date` - Earnings report date
- `symbol` - Company stock symbol
- `content` - Original transcript text (if `--no-content` not used)

## Cloud Deployment

### Building and Deploying to Cloud Run

1. **Build Docker image:**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/earnings-analysis
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy earnings-analysis \
     --image gcr.io/[PROJECT-ID]/earnings-analysis \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 3600 \
     --set-env-vars PRODUCTION_MODE=true,BATCH_SIZE=500
   ```

3. **Test deployment:**
   ```bash
   SERVICE_URL=$(gcloud run services describe earnings-analysis --region us-central1 --format 'value(status.url)')
   ./test_cloud_api.sh $SERVICE_URL
   ```

## Performance Tips

- **Test mode first**: Always run with `--test` or `mode=test` to validate configuration before full runs
- **Batch processing**: Cloud Run automatically processes in batches to handle large datasets
- **Date filtering**: Use `--start-date` and `--end-date` to limit scope and improve performance
- **Company filtering**: Analyze specific companies rather than full Fortune 100 when possible
- **BigQuery costs**: Use `--limit` to control query costs during development

## License

See LICENSE file for details.
