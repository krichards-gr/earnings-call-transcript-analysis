#!/usr/bin/env python3
"""
cli_analysis.py

Enhanced CLI tool for earnings call transcript analysis.
Supports company selection, date range filtering, and test vs. full mode.

Usage:
    # Run with default Fortune 100 companies, last 90 days, test mode (50 records)
    python cli_analysis.py --test

    # Run with specific companies, date range
    python cli_analysis.py --companies AAPL,MSFT,GOOGL --start-date 2024-01-01 --end-date 2024-12-31

    # Run full analysis for all Fortune 100 companies
    python cli_analysis.py --mode full

    # Run with custom company list file
    python cli_analysis.py --company-file my_companies.csv --mode full
"""

import argparse
import json
import os
import sys
import time
import re
import requests
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

from generate_topics import generate_topics_json
from analyzer import IssueAnalyzer

# =================================================================================================
# CONFIGURATION
# =================================================================================================

# Regenerate topics.json from topic_definitions.csv
generate_topics_json()

current_dir = os.getcwd()
TOPICS_FILE = os.path.join(current_dir, 'topics.json')
TICKERS_FILE = os.path.join(current_dir, 'tickers.csv')

# Default Cloud Run URL
DEFAULT_CLOUD_URL = "https://earnings-call-transcript-analysis-434903546449.us-central1.run.app"

SIMILARITY_THRESHOLD = 0.7

# Local Model Paths
INTERACTION_MODEL_PATH = os.path.join(current_dir, "models", "eng_type_class_v1")
ROLE_MODEL_PATH = os.path.join(current_dir, "models", "role_class_v1")
EMBEDDING_MODEL_PATH = os.path.join(current_dir, "models", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")

# Initialize the modular analyzer
issue_analyzer = IssueAnalyzer(
    similarity_threshold=SIMILARITY_THRESHOLD,
    embedding_model=EMBEDDING_MODEL_PATH
)

# Human-Readable Label Mappings
INTERACTION_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Answer",
    "LABEL_2": "Question"
}

ROLE_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Analyst",
    "LABEL_2": "Executive",
    "LABEL_3": "Operator"
}

# BigQuery Configuration
BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched_local"

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")

nlp = issue_analyzer.nlp
embedder = issue_analyzer.embedder

def load_model_safely(model_path, model_type="embedding"):
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print(f"Loading {model_type} model from {model_path}")
    try:
        if model_type == "embedding":
            return SentenceTransformer(model_path)
        else:
            return pipeline("text-classification", model=model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_type} model: {e}")
        sys.exit(1)

sentiment_analyzer = load_model_safely(SENTIMENT_MODEL_PATH, "sentiment")
interaction_classifier = load_model_safely(INTERACTION_MODEL_PATH, "interaction")
role_classifier = load_model_safely(ROLE_MODEL_PATH, "role")

print("Models loaded successfully.")

anchor_embeddings = issue_analyzer.anchor_embeddings
anchor_metadata = issue_analyzer.anchor_metadata
matcher = issue_analyzer.matcher
EXCLUSIONS_MAP = issue_analyzer.exclusions_map
ISSUE_AREA_MAP = issue_analyzer.issue_area_map

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def load_companies(company_file=None, company_symbols=None):
    """
    Load company symbols to analyze.

    Args:
        company_file: Path to CSV file with 'symbol' column
        company_symbols: List of company symbols (e.g., ['AAPL', 'MSFT'])

    Returns:
        List of company symbols
    """
    if company_symbols:
        return [s.strip().upper() for s in company_symbols]

    if company_file and os.path.exists(company_file):
        df = pd.read_csv(company_file)
        if 'symbol' in df.columns:
            return df['symbol'].tolist()
        else:
            print(f"ERROR: Company file must have 'symbol' column")
            sys.exit(1)

    # Default to tickers list
    if os.path.exists(TICKERS_FILE):
        df = pd.read_csv(TICKERS_FILE)
        # Remove any empty rows
        df = df.dropna(subset=['symbol'])
        df = df[df['symbol'].str.strip() != '']
        print(f"Loaded {len(df)} companies from {TICKERS_FILE}")
        return df['symbol'].tolist()
    else:
        print(f"ERROR: Tickers file not found at {TICKERS_FILE}")
        sys.exit(1)

def rejoin_fragments(df):
    """Rejoins segments split by line breaks."""
    if df.empty:
        return df
    rejoined_rows = []
    current_row = df.iloc[0].to_dict()
    for i in range(1, len(df)):
        next_row = df.iloc[i].to_dict()
        msg = str(current_row['content']).strip()
        is_fragment = not any(msg.endswith(p) for p in ['.', '?', '!', '"', '"', '"'])
        if next_row['transcript_id'] == current_row['transcript_id'] and is_fragment:
            current_row['content'] = msg + " " + str(next_row['content']).strip()
        else:
            rejoined_rows.append(current_row)
            current_row = next_row
    rejoined_rows.append(current_row)
    return pd.DataFrame(rejoined_rows)

def analyze_batch(texts):
    """
    Optimized batch analysis for topics and sentiment.
    Identical to analysis.py logic.
    """
    if not texts:
        return []

    query_embeddings = embedder.encode(texts, convert_to_tensor=True)
    all_scores = util.cos_sim(query_embeddings, anchor_embeddings) if anchor_embeddings is not None else None

    results_by_text = []
    sentiment_queue = []

    for i, text in enumerate(texts):
        # 1. spaCy Matcher
        doc = nlp(text)
        matches = matcher(doc)
        found_topics = set()
        if matches:
            for match_id, start, end in matches:
                found_topics.add(nlp.vocab.strings[match_id])

            text_results = []
            for topic in found_topics:
                exclusions = EXCLUSIONS_MAP.get(topic, [])
                if any(ext.lower() in text.lower() for ext in exclusions):
                    print(f"      [EXCLUSION] Dropping topic '{topic}' due to exclusionary term match.")
                    continue

                text_results.append({"topic": topic, "idx": i})
                sentiment_queue.append({"text": text, "text_pair": topic, "text_idx": i, "topic_idx": len(text_results)-1})
            results_by_text.append(text_results)
            continue

        # 2. Vector Similarity Fallback
        if anchor_embeddings is None:
            results_by_text.append([])
            continue

        cos_scores = all_scores[i]
        text_results = []
        for idx, score in enumerate(cos_scores):
            if score.item() >= SIMILARITY_THRESHOLD:
                topic = anchor_metadata[idx][0]
                exclusions = EXCLUSIONS_MAP.get(topic, [])
                if any(ext.lower() in text.lower() for ext in exclusions):
                    continue

                text_results.append({
                    "topic": topic,
                    "score": score.item(),
                    "idx": i,
                    "matched_anchor": anchor_metadata[idx][1]
                })

        text_results.sort(key=lambda x: x['score'], reverse=True)
        unique = {}
        for r in text_results:
            if r['topic'] not in unique: unique[r['topic']] = r
        top_topics = list(unique.values())[:3]

        for r in top_topics:
            sentiment_queue.append({
                "text": text,
                "text_pair": r['topic'],
                "text_idx": i,
                "topic_idx": top_topics.index(r)
            })

        results_by_text.append(top_topics)

    # 3. Batch Sentiment
    if sentiment_queue:
        print(f"      Running batch sentiment analysis for {len(sentiment_queue)} topic pairs...")
        texts_input = [item["text"] for item in sentiment_queue]
        pairs_input = [item["text_pair"] for item in sentiment_queue]

        batch_size = 4
        sent_results_flat = []

        try:
            for i in range(0, len(texts_input), batch_size):
                batch_texts = [str(t) for t in texts_input[i:i+batch_size]]
                batch_pairs = [str(p) for p in pairs_input[i:i+batch_size]]

                inputs = sentiment_analyzer.tokenizer(
                    batch_texts,
                    text_pair=batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )

                device = sentiment_analyzer.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = sentiment_analyzer.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                id2label = sentiment_analyzer.model.config.id2label
                for j in range(len(batch_texts)):
                    score, label_idx = torch.max(probs[j], dim=0)
                    label = id2label[label_idx.item()]
                    scores_str = ", ".join([f"{id2label[k][:3]}: {probs[j][k].item():.2f}" for k in range(len(id2label))])

                    sent_results_flat.append({
                        "label": label,
                        "score": score.item(),
                        "all_scores": scores_str
                    })

            for i, res in enumerate(sent_results_flat):
                meta = sentiment_queue[i]
                target = results_by_text[meta["text_idx"]][meta["topic_idx"]]
                target["sentiment"] = res['label']
                target["sentiment_score"] = res['score']
                target["all_scores"] = res['all_scores']

        except Exception as e:
            print(f"Error during sentiment batch processing: {e}")
            pass

    return results_by_text

# =================================================================================================
# CLOUD EXECUTION
# =================================================================================================

def run_cloud_analysis(cloud_url, companies, start_date=None, end_date=None, limit=None, mode='test', output_path=None):
    """
    Send analysis request to Cloud Run and download CSV results.

    Args:
        cloud_url: URL of the Cloud Run service
        companies: List of company symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of records
        mode: 'test' or 'full'

    Returns:
        Path to downloaded CSV file
    """
    print(f"\n{'='*80}")
    print(f"CLOUD ANALYSIS REQUEST")
    print(f"{'='*80}")
    print(f"Cloud URL: {cloud_url}")
    print(f"Companies: {len(companies)} ({', '.join(companies[:5])}{'...' if len(companies) > 5 else ''})")
    print(f"Date Range: {start_date or 'All'} to {end_date or 'All'}")
    print(f"Mode: {mode}")
    print(f"Limit: {limit or 'Default'}")
    print(f"{'='*80}\n")

    # Build request payload
    payload = {
        'companies': ','.join(companies),
        'mode': mode,
        'return_csv': True  # Request CSV response
    }

    if start_date:
        payload['start_date'] = start_date
    if end_date:
        payload['end_date'] = end_date
    if limit:
        payload['limit'] = limit

    print(f"Sending request to {cloud_url}/run...")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    try:
        # Get ID token for Cloud Run (required for authenticated Cloud Run services)
        headers = {'Content-Type': 'application/json'}
        try:
            # Try using gcloud CLI to get ID token
            import subprocess
            import shutil

            # Find gcloud in PATH
            gcloud_path = shutil.which('gcloud')
            if not gcloud_path:
                # Try common install locations
                possible_paths = [
                    os.path.expanduser('~/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'),
                    os.path.expanduser('~/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud'),
                    'C:\\Program Files (x86)\\Google\\Cloud SDK\\google-cloud-sdk\\bin\\gcloud.cmd',
                    '/usr/local/bin/gcloud',
                    '/usr/bin/gcloud'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        gcloud_path = path
                        break

            if gcloud_path:
                result = subprocess.run(
                    [gcloud_path, 'auth', 'print-identity-token'],
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=True if sys.platform == 'win32' else False
                )
                token = result.stdout.strip()
                if token:
                    headers['Authorization'] = f'Bearer {token}'
                    print("Using authenticated request with gcloud identity token")
                else:
                    print("Warning: Could not get identity token from gcloud")
                    print("Attempting unauthenticated request...")
            else:
                print("Warning: gcloud CLI not found in PATH or common locations")
                print("Attempting unauthenticated request...")
        except subprocess.CalledProcessError as e:
            print(f"Warning: gcloud command failed: {e}")
            print("Attempting unauthenticated request...")
        except Exception as auth_error:
            print(f"Warning: Could not get ID token: {auth_error}")
            print("Attempting unauthenticated request...")

        # Send POST request
        response = requests.post(
            f"{cloud_url}/run",
            json=payload,
            headers=headers,
            timeout=600  # 10 minute timeout for large requests
        )

        if response.status_code == 200:
            # Save CSV to file
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'cloud_analysis_results_{timestamp}.csv')
            else:
                # Ensure directory exists for custom path
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Count rows in CSV
            df = pd.read_csv(output_path)
            row_count = len(df)

            print(f"\n[SUCCESS] Cloud analysis complete!")
            print(f"  Results saved to: {output_path}")
            print(f"  Total records: {row_count}")

            return output_path
        else:
            print(f"[ERROR] Cloud request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out after 10 minutes")
        print(f"  Try reducing the limit or using a smaller date range")
        return None
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to {cloud_url}")
        print(f"  Check that the URL is correct and the service is running")
        return None
    except Exception as e:
        print(f"[ERROR] Error during cloud request: {e}")
        import traceback
        traceback.print_exc()
        return None

# =================================================================================================
# LOCAL ANALYSIS
# =================================================================================================

def run_analysis(companies, start_date=None, end_date=None, limit=None, write_to_bq=False, include_content=True, output_path=None):
    """
    Run the analysis pipeline with specified parameters.

    Args:
        companies: List of company symbols to analyze
        start_date: Start date (YYYY-MM-DD) or None for no start filter
        end_date: End date (YYYY-MM-DD) or None for no end filter
        limit: Maximum number of records to process, or None for all
        write_to_bq: Whether to write results to BigQuery
        include_content: Whether to include transcript content in output
    """
    print(f"\n{'='*80}")
    print(f"STARTING ANALYSIS")
    print(f"{'='*80}")
    print(f"Companies: {len(companies)} ({', '.join(companies[:5])}{'...' if len(companies) > 5 else ''})")
    print(f"Date Range: {start_date or 'All'} to {end_date or 'All'}")
    print(f"Limit: {limit or 'No limit'}")
    print(f"Write to BigQuery: {write_to_bq}")
    print(f"{'='*80}\n")

    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Build query
    where_clauses = []

    # Company filter
    companies_str = "', '".join(companies)
    where_clauses.append(f"m.symbol IN ('{companies_str}')")

    # Date filters
    if start_date:
        where_clauses.append(f"m.report_date >= '{start_date}'")
    if end_date:
        where_clauses.append(f"m.report_date <= '{end_date}'")

    where_clause = " AND ".join(where_clauses)

    query = f"""
        SELECT
            t.transcript_id,
            t.paragraph_number,
            t.speaker,
            t.content,
            m.* EXCEPT(transcript_id)
        FROM `{BQ_SOURCE_TABLE}` t
        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
        WHERE {where_clause}
        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
        {f'LIMIT {limit}' if limit else ''}
    """

    print(f"Executing query...")
    print(f"Query: {query[:200]}...")

    df = client.query(query).to_dataframe()

    if df.empty:
        print("No data found matching criteria.")
        return

    print(f"Found {len(df)} transcript segments")

    df = rejoin_fragments(df)
    print(f"Processing {len(df)} rows after rejoining fragments...")

    texts = df['content'].astype(str).tolist()
    truncated_texts = [t[:512] for t in texts]

    # 1. Batch Classification
    print(f"   Running batch interaction classification for {len(df)} segments...")
    int_results = interaction_classifier(truncated_texts, batch_size=4)

    print(f"   Running batch role classification for {len(df)} segments...")
    role_results = role_classifier(truncated_texts, batch_size=4)

    # 2. Batch Topic & Sentiment Analysis
    print(f"   Running batch topic/sentiment analysis...")
    enrichment_results = analyze_batch(texts)

    # 3. Assemble Results
    all_results = []
    current_session_id = 0
    current_analyst = "None"
    last_transcript_id = None

    SESSION_START_PATTERNS = [
        r"next question (?:comes|is coming)",
        r"next we (?:have|will go to)",
        r"question (?:comes|is coming) from",
        r"your first question (?:comes|is coming)",
        r"move to the line of",
        r"go to the line of",
        r"from the line of",
        r"comes? from the line of"
    ]
    session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)

    intro_regex = re.compile(
        r"(?:line of|comes from|is from|from|at)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from|at|is coming)",
        re.IGNORECASE
    )

    for i, (_, row) in enumerate(df.iterrows()):
        if last_transcript_id is not None and row['transcript_id'] != last_transcript_id:
            current_session_id = 0
            current_analyst = "None"
        last_transcript_id = row['transcript_id']

        int_res = int_results[i]['label']
        role_res = role_results[i]['label']
        interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)
        role_label = ROLE_ID_MAP.get(role_res, role_res)
        text = str(row['content'])

        lower_text = text.lower()
        is_operator = role_label == "Operator"
        has_session_start_keyword = session_start_regex.search(lower_text)
        is_transition_text = any(k in lower_text for k in ["question", "line of", "analyst"])

        if (is_operator and is_transition_text) or has_session_start_keyword:
            current_session_id += 1
            match = intro_regex.search(text)
            if match:
                current_analyst = match.group(1).strip()
            elif is_operator and "question" in lower_text:
                current_analyst = "Unknown Analyst"

        detected = enrichment_results[i]
        if not detected:
            detected = [{
                "topic": None,
                "sentiment": None,
                "sentiment_score": None,
                "all_scores": None,
                "similarity_score": None,
                "matched_anchor": None
            }]

        for d in detected:
            res_row = {
                "transcript_id": row['transcript_id'],
                "paragraph_number": row['paragraph_number'],
                "speaker": row['speaker'],
                "qa_session_id": current_session_id,
                "qa_session_label": current_analyst,
                "interaction_type": interaction_type,
                "role": role_label,
                "topic": d.get('topic'),
                "issue_area": ISSUE_AREA_MAP.get(d.get('topic'), "Unknown"),
                "issue_subtopic": d.get('topic'),
                "sentiment_label": d.get('sentiment'),
                "sentiment_score": d.get('sentiment_score'),
                "all_scores": d.get('all_scores'),
                "similarity_score": d.get('similarity_score'),
                "matched_anchor": d.get('matched_anchor'),
            }

            # Add all metadata columns from BigQuery
            # This includes report_date, symbol, and any other metadata fields
            for col in row.index:
                if col not in ['transcript_id', 'paragraph_number', 'speaker', 'content']:
                    res_row[col] = row[col]

            # Add content if requested
            if include_content:
                res_row["content"] = text

            all_results.append(res_row)

    # 4. Save Results
    if all_results:
        results_df = pd.DataFrame(all_results)

        if not output_path:
            output_dir = os.path.join(current_dir, 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f'cli_analysis_results_{timestamp}.csv')
        else:
            # Ensure directory exists for custom path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Analysis complete!")
        print(f"  Results saved to: {output_path}")
        print(f"  Total records: {len(results_df)}")

        if write_to_bq:
            print(f"\n  Writing results to BigQuery: {BQ_DEST_TABLE}")
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )
            job = client.load_table_from_dataframe(results_df, BQ_DEST_TABLE, job_config=job_config)
            job.result()
            print(f"  [SUCCESS] Wrote {len(results_df)} rows to BigQuery")
    else:
        print("\nNo results found.")

# =================================================================================================
# CLI INTERFACE
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Earnings Call Transcript Analysis CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode with Fortune 100 (50 records)
  python cli_analysis.py --test

  # Specific companies, last 90 days
  python cli_analysis.py --companies AAPL,MSFT,GOOGL

  # Full analysis for all Fortune 100, specific date range
  python cli_analysis.py --mode full --start-date 2024-01-01 --end-date 2024-12-31

  # Custom company file
  python cli_analysis.py --company-file my_companies.csv --mode full
        """
    )

    # Company selection
    parser.add_argument('--companies', type=str,
                       help='Comma-separated list of company symbols (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--company-file', type=str,
                       help='Path to CSV file with company symbols (must have "symbol" column)')

    # Date range
    parser.add_argument('--start-date', type=str,
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=90,
                       help='Number of days back from today (default: 90, ignored if start-date specified)')

    # Mode
    parser.add_argument('--mode', type=str, choices=['test', 'full'], default='test',
                       help='Analysis mode: test (50 records) or full (all records)')
    parser.add_argument('--test', action='store_true',
                       help='Shorthand for --mode test')
    parser.add_argument('--limit', type=int,
                       help='Custom limit for number of records to process')

    # Output
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (default: outputs/cli_analysis_results_TIMESTAMP.csv)')
    parser.add_argument('--write-to-bq', action='store_true',
                       help='Write results to BigQuery in addition to local CSV')
    parser.add_argument('--no-content', action='store_true',
                       help='Exclude transcript content from output (smaller file size)')

    # Cloud execution
    parser.add_argument('--cloud', action='store_true',
                       help='Execute analysis on Cloud Run instead of locally')
    parser.add_argument('--cloud-url', type=str, default=DEFAULT_CLOUD_URL,
                       help=f'Cloud Run service URL (default: {DEFAULT_CLOUD_URL})')

    args = parser.parse_args()

    # Parse companies
    company_symbols = None
    if args.companies:
        company_symbols = [s.strip() for s in args.companies.split(',')]

    companies = load_companies(
        company_file=args.company_file,
        company_symbols=company_symbols
    )

    # Parse dates
    start_date = args.start_date
    end_date = args.end_date

    if not start_date and not end_date:
        # Default to last N days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days_back)).strftime('%Y-%m-%d')

    # Parse mode/limit
    if args.test:
        mode = 'test'
    else:
        mode = args.mode

    if args.limit:
        limit = args.limit
    elif mode == 'test':
        limit = 50
    else:
        limit = None

    # Run analysis (cloud or local)
    if args.cloud:
        # Cloud execution
        cloud_url = args.cloud_url or DEFAULT_CLOUD_URL

        run_cloud_analysis(
            cloud_url=cloud_url,
            companies=companies,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            mode=mode,
            output_path=args.output
        )
    else:
        # Local execution
        run_analysis(
            companies=companies,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            write_to_bq=args.write_to_bq,
            include_content=not args.no_content,
            output_path=args.output
        )

if __name__ == "__main__":
    main()
