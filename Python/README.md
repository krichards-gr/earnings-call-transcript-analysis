# Earnings Call Transcript Analysis

This tool matches text segments from earnings call transcripts to predefined topics and assesses the sentiment specifically toward those topics.

## Features

*   **Topic Detection**:
    *   **Exact Match**: Uses spaCy to find specific keywords defined in `topics.json`.
    *   **Vector Similarity**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to find semantically similar text if no exact matches are found.
*   **Aspect-Based Sentiment Analysis (ABSA)**:
    *   Uses `yangheng/deberta-v3-base-absa-v1.1` to determine the sentiment (Positive, Negative, Neutral) of the text *specifically regarding the detected topic*.
    *   Provides a confidence score and a full breakdown of sentiment probabilities.

## Installation

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download spaCy Model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```

   *Note: On the first run, the script will automatically download the SentenceTransformer model (~90MB) and the DeBERTa ABSA model (~400MB) from Hugging Face.*

## Usage

Run the script on a transcript file:

```bash
python analysis.py inputs/sample_transcript.txt
```

### BigQuery Integration

To read from and write to BigQuery, use the `--bq` flag:

```bash
python analysis.py --bq
```

*   **Source Table**: `sri-benchmarking-databases.pressure_monitoring.earnings_call_transcript_content`
*   **Destination Table**: `sri-benchmarking-databases.pressure_monitoring.earnings_call_transcript_enriched`

**Output Schema (Tidy/Long Format)**:
The enriched table will have one row per detected topic instance:
*   `transcript_id` (STRING)
*   `paragraph_number` (INTEGER)
*   `topic` (STRING)
*   `sentiment_label` (STRING)
*   `sentiment_score` (FLOAT)

### Output Format (Local)

The script outputs the analyzed paragraphs, detected topics, and sentiment:

```text
Paragraph 1: "This is a sample text about DEI..."
  [DEBUG] Found 1 exact matches.
    - Match: 'DEI' -> Topic: DEI
Result: [{'topic': 'DEI', 'sentiment': 'Positive', 'score': 0.63, 'all_scores': 'Pos: 0.63, Neu: 0.36, Neg: 0.01'}]
    -> Topic: DEI
       Sentiment: Positive (0.63)
       Breakdown: [Pos: 0.63, Neu: 0.36, Neg: 0.01]
```

## Configuration

Topics are defined in `topics.json`:

*   **label**: The display name of the topic.
*   **patterns**: List of spaCy match patterns for exact matching.
*   **anchors**: List of phrases used for vector similarity comparison.

Example:
```json
{
  "label": "DEI",
  "patterns": [[{"LOWER": "dei"}]],
  "anchors": ["workforce diversity", "inclusive hiring"]
}
```
