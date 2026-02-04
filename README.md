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

## Usage

### Local Analysis (`local_analysis.py`)

Run the script to process data from BigQuery and save results locally:

```bash
python local_analysis.py
```

**Testing**:
Run the automated test script to verify local functionality:
```bash
python test_local_pipeline.py
```

### Production Deployment (Cloud Run)

The production pipeline (`analysis.py`) is designed for Cloud Run and includes a health check and an execution trigger.

**Triggering Execution**:
Send a GET request to the `/run` endpoint:
```bash
curl https://[YOUR-SERVICE-URL]/run
```

**Testing**:
Run the automated test script to verify cloud connectivity and execution:
```bash
python test_cloud_pipeline.py [YOUR-SERVICE-URL]
```

## Configuration

- `BATCH_SIZE`: Number of segments to process in one BigQuery cycle (default: 500).
- `PRODUCTION_MODE`: Set to `true` for full run, `false` for limited testing batches.
- `WRITE_TO_BQ`: (Local only) Enable/disable writing results back to BigQuery.
- `INCLUDE_CONTENT`: Include original text in the output for easier manual verification.

## Project Structure

- `analysis.py`: Main production entry point for Cloud Run.
- `local_analysis.py`: Synchronized local version for testing and development.
- `download_models.py`: Utility to bake models into the Docker image.
- `generate_topics.py`: Utility to transform raw CSV inputs and generate `topics.json`.
- `test_*.py`: Automated testing scripts for local and cloud pipelines.

## Installation

1.  **Dependencies**: `pip install -r requirements.txt`
2.  **spaCy**: `python -m spacy download en_core_web_sm`
3.  **Models**: Run `python download_models.py` to ensure local availability.
