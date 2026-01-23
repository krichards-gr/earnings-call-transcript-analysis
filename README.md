# Earnings Call Transcript Analysis

This tool performs deep analysis on earnings call transcripts by identifying topics, assessing sentiment, and classifying speaker roles and interaction types. It also clusters question-and-answer sequences for better traceability.

## Features

*   **Topic Detection**:
    *   **Exact Match**: Uses spaCy match patterns to find specific keywords.
    *   **Vector Similarity**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for semantic matching.
*   **Aspect-Based Sentiment Analysis (ABSA)**:
    *   Uses `yangheng/deberta-v3-base-absa-v1.1` to determine sentiment toward detected topics.
*   **Speaker & Role Classification**:
    *   **Role**: Classifies speakers as **Analyst**, **Executive**, **Operator**, or **Admin** using the local `role_class_v1` model.
    *   **Interaction Type**: Classifies segments as **Admin**, **Answer**, or **Question** using the `eng_type_class_v1` model.
*   **Q&A Clustering**:
    *   Automatically groups questions and their corresponding answers into "Sessions" based on Operator introductions.
*   **Data Integrity**:
    *   **Fragment Rejoining**: Automatically rejoins segments split by line breaks to ensure models process complete statements.
    *   **Performance Monitoring**: Prints inference time per classification in the console.

## Usage

### Local Analysis (`local_analysis.py`)

Run the script to process data from BigQuery and save results locally:

```bash
python local_analysis.py
```

**Workflow Configuration**:
- `PRODUCTION_TESTING`: Set to `True` (default) to process only 20 segments for verification. Set to `False` for the full run.
- `WRITE_TO_BQ`: Set to `True` to upload results to the cloud.
- `INCLUDE_CONTENT`: Set to `True` to include the original text in the output for verification.

## Deployment (Cloud Run)

### Model Storage (Automatic)
The `Dockerfile` is configured to automatically download and bake all external models into the image during the build process. This includes:
- `en_core_web_sm` (spaCy)
- `all-MiniLM-L6-v2` (SentenceTransformer)
- `deberta-v3-base-absa-v1.1` (Sentiment)
- Local classification models (`eng_type_class_v1`, `role_class_v1`)

This ensures zero runtime latency and removes the need for internet access during execution on Cloud Run.

### Topic Regeneration
If you modify `topic_definitions.csv`, the script will automatically regenerate `topics.json` on the next run.

### Production Output Schema (`earnings_call_transcript_enriched`)

The focused production table contains:

| Column | Type | Description |
| :--- | :--- | :--- |
| `transcript_id` | STRING | Unique ID for the transcript. |
| `paragraph_number` | INTEGER | Original paragraph/segment number. |
| `speaker` | STRING | Name of the speaker. |
| `qa_session_id` | INTEGER | ID grouping a specific analyst exchange. |
| `qa_session_label` | STRING | Name of the analyst in the exchange. |
| `interaction_type` | STRING | Admin, Question, or Answer. |
| `role` | STRING | Analyst, Executive, Operator, or Admin. |
| `topic` | STRING | The detected topic label. |
| `sentiment_label` | STRING | Positive, Negative, or Neutral. |
| `sentiment_score` | FLOAT | Confidence score for the sentiment. |

### Local/Debug Output Schema (`earnings_call_transcript_enriched_local`)

The comprehensive local table includes all production fields plus:

| Column | Type | Description |
| :--- | :--- | :--- |
| `all_scores` | STRING | Full sentiment probability breakdown. |
| `similarity_score` | FLOAT | Score for vector-based topic matches. |
| `matched_anchor` | STRING | The anchor term matched via vector similarity. |
| `content` | STRING | The original segment text (if enabled). |
| `report_date` | DATE | Date of the earnings call. |
| `symbol` | STRING | Ticker symbol. |

## Installation

1.  **Dependencies**: `pip install -r requirements.txt`
2.  **spaCy**: `python -m spacy download en_core_web_sm`


Minor change to test auto-build