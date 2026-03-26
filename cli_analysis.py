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

# MEMORY OPTIMIZATION: Force CPU-only mode to reduce memory usage
# Comment out these lines if you have a GPU and want to use it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import time
import re
import requests
import warnings
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
import torch

# MEMORY OPTIMIZATION: Limit CPU threads for lower memory footprint
torch.set_num_threads(2)

# Suppress tokenizer regex pattern warnings (known issue with DeBERTa tokenizers)
warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*', category=FutureWarning)

from generate_topics import generate_all
from analyzer import IssueAnalyzer
from parallel_analyzer import ParallelAnalyzer, get_optimal_config
from tqdm import tqdm

# =================================================================================================
# CONFIGURATION
# =================================================================================================

# Regenerate topics.json from topic_definitions.csv
# generate_all(from_raw=False)

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
# SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")  # COMMENTED OUT - Using VADER instead

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
BQ_CORP_REF_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.corporation_reference"

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
        elif model_type == "sentiment":
            # Load tokenizer - suppress regex pattern warning (known DeBERTa issue, doesn't affect functionality)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*incorrect regex pattern.*')
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                return pipeline("text-classification", model=model_path, tokenizer=tokenizer)
        else:
            return pipeline("text-classification", model=model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_type} model: {e}")
        sys.exit(1)

# sentiment_analyzer = load_model_safely(SENTIMENT_MODEL_PATH, "sentiment")  # COMMENTED OUT - Using VADER instead
interaction_classifier = load_model_safely(INTERACTION_MODEL_PATH, "interaction")
role_classifier = load_model_safely(ROLE_MODEL_PATH, "role")

# Initialize VADER sentiment analyzer (fast and simple)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
print("Using VADER for sentiment analysis (fast mode)")

print("Models loaded successfully.")

anchor_embeddings = issue_analyzer.anchor_embeddings
anchor_metadata = issue_analyzer.anchor_metadata
matcher = issue_analyzer.matcher
EXCLUSIONS_MAP = issue_analyzer.exclusions_map
ISSUE_AREA_MAP = issue_analyzer.issue_area_map

# =================================================================================================
# PARALLEL CONFIGURATION
# =================================================================================================

# Get optimal configuration based on system resources
print("\nDetecting optimal configuration...")
OPTIMAL_CONFIG = get_optimal_config()

# Initialize parallel analyzer after models are loaded
print("\nInitializing parallel processing...")
parallel_analyzer = ParallelAnalyzer(
    sentiment_analyzer=sentiment_analyzer,
    interaction_classifier=interaction_classifier,
    role_classifier=role_classifier,
    embedder=embedder,
    nlp=nlp,
    matcher=matcher,
    anchor_embeddings=anchor_embeddings,
    anchor_metadata=anchor_metadata,
    exclusions_map=EXCLUSIONS_MAP,
    similarity_threshold=SIMILARITY_THRESHOLD,
    num_workers=OPTIMAL_CONFIG['num_workers'],
    sentiment_batch_size=OPTIMAL_CONFIG['sentiment_batch_size'],
    classification_batch_size=OPTIMAL_CONFIG['classification_batch_size']
)

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

def format_time(seconds):
    """Format elapsed time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

def ensure_complete_transcripts(df):
    """
    Ensure only complete transcripts are returned.
    If the dataframe cuts off mid-transcript, remove the incomplete one.

    Args:
        df: DataFrame with transcript data

    Returns:
        DataFrame with only complete transcripts
    """
    if df.empty:
        return df

    # Get unique transcript IDs in order of appearance
    transcript_ids = df['transcript_id'].unique()

    # Check if the last transcript is complete by counting paragraphs
    # A transcript is considered complete if we have all sequential paragraphs
    complete_transcript_ids = []

    for tid in transcript_ids:
        transcript_df = df[df['transcript_id'] == tid]
        paragraphs = sorted(transcript_df['paragraph_number'].unique())

        # Check if paragraphs are sequential from 1 (or 0)
        expected_paragraphs = list(range(paragraphs[0], paragraphs[0] + len(paragraphs)))

        if paragraphs == expected_paragraphs:
            complete_transcript_ids.append(tid)
        else:
            # This transcript is incomplete, don't include it
            print(f"   Excluding incomplete transcript: {tid} (has {len(paragraphs)} non-sequential paragraphs)")

    # Return only complete transcripts
    return df[df['transcript_id'].isin(complete_transcript_ids)]

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

def _sentiment_only_batch(texts):
    """Run standalone VADER sentiment on each text (no topic detection)."""
    results = []
    for text in texts:
        scores = sentiment_analyzer.polarity_scores(str(text))
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        results.append([{
            "topic": None,
            "sentiment": label,
            "sentiment_score": abs(compound),
            "all_scores": f"pos: {scores['pos']:.2f}, neu: {scores['neu']:.2f}, neg: {scores['neg']:.2f}, compound: {compound:.2f}",
            "similarity_score": None,
            "matched_anchor": None
        }])
    return results

def analyze_batch(texts, skip_sentiment=False):
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

    # 3. VADER Sentiment Analysis (Fast and Simple)
    if sentiment_queue and not skip_sentiment:
        print(f"      Running VADER sentiment analysis for {len(sentiment_queue)} topic pairs...")

        # VADER is super fast - just analyze each text directly
        for item in sentiment_queue:
            text = str(item["text"])
            # Get VADER scores
            scores = sentiment_analyzer.polarity_scores(text)

            # Determine sentiment label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            # Update the result
            meta = item
            target = results_by_text[meta["text_idx"]][meta["topic_idx"]]
            target["sentiment"] = label
            target["sentiment_score"] = abs(compound)  # Use absolute compound score
            target["all_scores"] = f"pos: {scores['pos']:.2f}, neu: {scores['neu']:.2f}, neg: {scores['neg']:.2f}, compound: {compound:.2f}"

    return results_by_text

# =================================================================================================
# CLOUD EXECUTION
# =================================================================================================

def run_cloud_analysis(cloud_url, companies, start_date=None, end_date=None, limit=None, latest=None,
                      earliest=None, mode='test', output_path=None):
    """
    Send analysis request to Cloud Run and download CSV results.

    Args:
        cloud_url: URL of the Cloud Run service
        companies: List of company symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum number of records (complete transcripts only)
        latest: Number of latest transcripts to pull
        earliest: Number of earliest transcripts to pull
        mode: 'test' or 'full'

    Returns:
        Path to downloaded CSV file
    """
    start_time = time.time()

    mode_desc = f"Latest {latest} transcripts" if latest else f"Earliest {earliest} transcripts" if earliest else f"Limit: {limit or 'Default'}"

    print(f"\n{'='*80}")
    print(f"CLOUD ANALYSIS REQUEST")
    print(f"{'='*80}")
    print(f"Cloud URL: {cloud_url}")
    print(f"Companies: {len(companies)} ({', '.join(companies[:5])}{'...' if len(companies) > 5 else ''})")
    print(f"Date Range: {start_date or 'All'} to {end_date or 'All'}")
    print(f"Mode: {mode_desc}")
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
    if latest:
        payload['latest'] = latest
    if earliest:
        payload['earliest'] = earliest

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
            timeout=3600  # 60 minute timeout to match Cloud Run max timeout
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

            elapsed_time = time.time() - start_time
            time_str = format_time(elapsed_time)

            print(f"\n[SUCCESS] Cloud analysis complete!")
            print(f"  Results saved to: {output_path}")
            print(f"  Total records: {row_count}")
            print(f"  Time taken: {time_str}")

            return output_path
        else:
            print(f"[ERROR] Cloud request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out after 60 minutes")
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

def run_analysis(companies, start_date=None, end_date=None, limit=None, latest=None, earliest=None,
                 write_to_bq=False, include_content=True, output_path=None, use_parallel=True,
                 num_workers=None, sentiment_batch_size=None, classification_batch_size=None,
                 enable_interaction_type=True, enable_role=True, enable_topics=True,
                 enable_sentiment=True, enable_sessions=True):
    """
    Run the analysis pipeline with specified parameters.

    Args:
        companies: List of company symbols to analyze
        start_date: Start date (YYYY-MM-DD) or None for no start filter
        end_date: End date (YYYY-MM-DD) or None for no end filter
        limit: Maximum number of records to process, or None for all (complete transcripts only)
        latest: Number of latest transcripts to pull (overrides limit)
        earliest: Number of earliest transcripts to pull (overrides limit)
        write_to_bq: Whether to write results to BigQuery
        include_content: Whether to include transcript content in output
        use_parallel: Whether to use parallel processing (default: True)
        num_workers: Number of worker processes (None = auto-detect)
        sentiment_batch_size: Batch size for sentiment analysis (None = auto-detect)
        classification_batch_size: Batch size for classification (None = auto-detect)
    """
    start_time = time.time()

    # Override parallel analyzer settings if specified
    global parallel_analyzer
    if use_parallel and (num_workers or sentiment_batch_size or classification_batch_size):
        print("\nReconfiguring parallel analyzer with custom settings...")
        parallel_analyzer = ParallelAnalyzer(
            sentiment_analyzer=sentiment_analyzer,
            interaction_classifier=interaction_classifier,
            role_classifier=role_classifier,
            embedder=embedder,
            nlp=nlp,
            matcher=matcher,
            anchor_embeddings=anchor_embeddings,
            anchor_metadata=anchor_metadata,
            exclusions_map=EXCLUSIONS_MAP,
            similarity_threshold=SIMILARITY_THRESHOLD,
            num_workers=num_workers or OPTIMAL_CONFIG['num_workers'],
            sentiment_batch_size=sentiment_batch_size or OPTIMAL_CONFIG['sentiment_batch_size'],
            classification_batch_size=classification_batch_size or OPTIMAL_CONFIG['classification_batch_size']
        )

    mode_desc = f"Latest {latest} transcripts" if latest else f"Earliest {earliest} transcripts" if earliest else f"Limit: {limit or 'No limit'}"

    print(f"\n{'='*80}")
    print(f"STARTING ANALYSIS")
    print(f"{'='*80}")
    print(f"Companies: {len(companies)} ({', '.join(companies[:5])}{'...' if len(companies) > 5 else ''})")
    print(f"Date Range: {start_date or 'All'} to {end_date or 'All'}")
    print(f"Mode: {mode_desc}")
    print(f"Write to BigQuery: {write_to_bq}")
    enabled_steps = [name for name, on in [
        ("interaction_type", enable_interaction_type), ("role", enable_role),
        ("topics", enable_topics), ("sentiment", enable_sentiment), ("sessions", enable_sessions)
    ] if on]
    skipped_steps = [name for name, on in [
        ("interaction_type", enable_interaction_type), ("role", enable_role),
        ("topics", enable_topics), ("sentiment", enable_sentiment), ("sessions", enable_sessions)
    ] if not on]
    print(f"Enrichment: {', '.join(enabled_steps) if enabled_steps else 'NONE'}")
    if skipped_steps:
        print(f"Skipped: {', '.join(skipped_steps)}")
    print(f"{'='*80}\n")

    # Use ADC with Drive scope (required for Drive-backed BQ tables)
    credentials, _ = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/drive',
        ]
    )
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)

    # Build query based on mode
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

    # Build query based on latest/earliest/limit mode
    if latest:
        # Get the latest N transcripts, then pull all content for them
        query = f"""
            WITH selected_transcripts AS (
                SELECT m.transcript_id
                FROM `{BQ_METADATA_TABLE}` m
                WHERE {where_clause}
                ORDER BY m.report_date DESC, m.symbol
                LIMIT {latest}
            )
            SELECT
                t.transcript_id,
                t.paragraph_number,
                t.speaker,
                t.content,
                m.* EXCEPT(transcript_id, corporation, sector),
                cr.corporation,
                cr.sector
            FROM `{BQ_SOURCE_TABLE}` t
            JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
            LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
            WHERE t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
            ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
        """
    elif earliest:
        # Get the earliest N transcripts, then pull all content for them
        query = f"""
            WITH selected_transcripts AS (
                SELECT m.transcript_id
                FROM `{BQ_METADATA_TABLE}` m
                WHERE {where_clause}
                ORDER BY m.report_date ASC, m.symbol
                LIMIT {earliest}
            )
            SELECT
                t.transcript_id,
                t.paragraph_number,
                t.speaker,
                t.content,
                m.* EXCEPT(transcript_id),
                cr.corporation,
                cr.sector
            FROM `{BQ_SOURCE_TABLE}` t
            JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
            LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
            WHERE t.transcript_id IN (SELECT transcript_id FROM selected_transcripts)
            ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
        """
    else:
        # Standard query with optional limit (will be post-processed for complete transcripts)
        query = f"""
            SELECT
                t.transcript_id,
                t.paragraph_number,
                t.speaker,
                t.content,
                m.* EXCEPT(transcript_id),
                cr.corporation,
                cr.sector
            FROM `{BQ_SOURCE_TABLE}` t
            JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
            LEFT JOIN `{BQ_CORP_REF_TABLE}` cr ON m.symbol = cr.Ticker
            WHERE {where_clause}
            ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
            {f'LIMIT {limit * 2}' if limit else ''}
        """

    print(f"\n[BIGQUERY STAGE]")
    print(f"=" * 80)
    print(f"Executing query...")
    print(f"Query preview: {query[:200]}...")

    # Execute query with progress tracking
    query_start = time.time()
    query_job = client.query(query)

    # Show query progress
    print("   Waiting for query to complete...", end="", flush=True)
    df = query_job.result().to_dataframe()
    query_time = time.time() - query_start
    print(f" Done! ({query_time:.2f}s)")

    if df.empty:
        print("\n[WARNING] No data found matching criteria.")
        return

    print(f"   Found {len(df)} transcript segments from {df['transcript_id'].nunique()} transcripts")
    print(f"   Data size: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"=" * 80)

    # Ensure only complete transcripts (important when using limit)
    if not latest and not earliest:
        df = ensure_complete_transcripts(df)
        print(f"Ensured complete transcripts: {len(df)} segments from {df['transcript_id'].nunique()} complete transcripts")

    df = rejoin_fragments(df)
    print(f"Processing {len(df)} rows after rejoining fragments...")

    texts = df['content'].astype(str).tolist()

    # PREPROCESSING: Clean operator intro text BEFORE classification
    # This prevents classifier from seeing operator language and misclassifying analyst questions
    print(f"Preprocessing texts (stripping operator intros)...")
    operator_intro_pattern = r"^(?:We'll|We will|Let's|Certainly\.?)?\s*(?:go ahead and\s+)?(?:take|move to|now go to)\s+(?:our\s+)?(?:the\s+)?(?:first|next)?\s*question\s+(?:from|is from)\s+[^.!?]+[.!?]\s+"
    cleaned_texts = []
    cleaning_count = 0
    for text in texts:
        cleaned = re.sub(operator_intro_pattern, '', text, flags=re.IGNORECASE).strip()
        if len(cleaned) < len(text):
            cleaning_count += 1
        cleaned_texts.append(cleaned if cleaned else text)  # Keep original if cleaning removed everything

    if cleaning_count > 0:
        print(f"   Cleaned operator intro from {cleaning_count} segments")

    # Use cleaned texts for everything
    texts = cleaned_texts

    # Choose processing method
    n_segments = len(df)
    int_results = None
    role_results = None
    enrichment_results = None

    if use_parallel:
        print(f"\n[PARALLEL PROCESSING MODE]")
        print(f"=" * 80)

        # 1. Parallel Classification
        if enable_interaction_type or enable_role:
            print(f"\n1. Classification Stage ({n_segments} segments)")
            print("-" * 80)

        if enable_interaction_type:
            print(f"   Running batch interaction classification...")
            int_results = parallel_analyzer.classify_batch_parallel(texts, 'interaction')

        if enable_role:
            print(f"   Running batch role classification...")
            role_results = parallel_analyzer.classify_batch_parallel(texts, 'role')

        # 2. Parallel Topic & Sentiment Analysis
        if enable_topics or enable_sentiment:
            print(f"\n2. Topic & Sentiment Analysis Stage")
            print("-" * 80)
            if enable_topics:
                enrichment_results = parallel_analyzer.analyze_batch_parallel(
                    texts, show_progress=True, skip_sentiment=not enable_sentiment
                )
            elif enable_sentiment:
                # Sentiment only (no topics) — run VADER directly on each text
                print(f"   Running standalone VADER sentiment analysis...")
                enrichment_results = _sentiment_only_batch(texts)

        print(f"\n" + "=" * 80)
    else:
        print(f"\n[SEQUENTIAL PROCESSING MODE]")
        print(f"=" * 80)

        truncated_texts = [t[:512] for t in texts]

        # 1. Batch Classification
        # MEMORY OPTIMIZATION: Using batch_size=2 instead of 4 to reduce memory usage
        if enable_interaction_type:
            print(f"   Running batch interaction classification for {n_segments} segments...")
            int_results = interaction_classifier(truncated_texts, batch_size=2)

        if enable_role:
            print(f"   Running batch role classification for {n_segments} segments...")
            role_results = role_classifier(truncated_texts, batch_size=2)

        # 2. Batch Topic & Sentiment Analysis
        if enable_topics:
            print(f"   Running batch topic/sentiment analysis...")
            enrichment_results = analyze_batch(texts, skip_sentiment=not enable_sentiment)
        elif enable_sentiment:
            print(f"   Running standalone VADER sentiment analysis...")
            enrichment_results = _sentiment_only_batch(texts)

        print(f"\n" + "=" * 80)

    # 3. Assemble Results
    print(f"\n3. Results Assembly Stage")
    print("-" * 80)
    all_results = []
    current_session_id = 0
    last_transcript_id = None

    # Session tracking regexes (only needed if sessions enabled)
    session_start_regex = None
    if enable_sessions:
        SESSION_START_PATTERNS = [
            r"next question",
            r"first question",
            r"question (?:is |will be )?(?:coming )?from",
            r"(?:we'll |we will )?(?:now )?(?:go to|take)",
            r"move to the line of",
            r"go to the line of",
            r"from the line of"
        ]
        session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)

    # Track previous interaction type for Answer validation
    previous_interaction = None
    previous_role = None

    print(f"   Assembling {len(df)} segments into enriched results...")
    for i, (_, row) in enumerate(tqdm(list(df.iterrows()), desc="Assembling Results", leave=False)):
        if last_transcript_id is not None and row['transcript_id'] != last_transcript_id:
            current_session_id = 0
            previous_interaction = None
            previous_role = None
        last_transcript_id = row['transcript_id']

        # Use cleaned text from preprocessing step
        text = texts[i]

        # Classification results (None if step was skipped)
        interaction_type = None
        role_label = None

        if enable_interaction_type and int_results:
            int_res = int_results[i]['label']
            interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)

        if enable_role and role_results:
            role_res = role_results[i]['label']
            role_label = ROLE_ID_MAP.get(role_res, role_res)

        # POST-PROCESSING: Validate "Answer" labels (requires both interaction + role)
        if enable_interaction_type and enable_role and interaction_type and role_label:
            if interaction_type == "Answer":
                if previous_interaction != "Question" or previous_role != "Analyst":
                    interaction_type = "Admin"

            # Boost "Question" detection for Analyst segments
            if role_label == "Analyst" and interaction_type != "Question":
                question_indicators = [
                    "have two", "have a question", "wondering", "curious", "want to ask",
                    "can you", "could you", "would you", "will you",
                    "how do", "how does", "how should", "how would",
                    "what", "why", "when", "where", "which",
                    "talk about", "comment on", "thoughts on", "perspective on"
                ]
                lower_text = text.lower()
                if any(indicator in lower_text for indicator in question_indicators):
                    interaction_type = "Question"

        # Session tracking (requires role to detect Operator transitions)
        if enable_sessions:
            lower_text = text.lower()
            is_operator = role_label == "Operator" if role_label else False
            has_session_start_keyword = session_start_regex.search(lower_text)
            is_transition_text = any(k in lower_text for k in ["question", "line of", "analyst"])

            if (is_operator and is_transition_text) or has_session_start_keyword:
                current_session_id += 1

        # Build enrichment data for this segment
        if enrichment_results:
            detected = enrichment_results[i]
        else:
            detected = None

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
            }

            if enable_sessions:
                res_row["qa_session_id"] = current_session_id
            if enable_interaction_type:
                res_row["interaction_type"] = interaction_type
            if enable_role:
                res_row["role"] = role_label
            if enable_topics:
                res_row["issue_area"] = ISSUE_AREA_MAP.get(d.get('topic'), "Unknown")
                res_row["issue_subtopic"] = d.get('topic')
                res_row["similarity_score"] = d.get('similarity_score')
                res_row["matched_anchor"] = d.get('matched_anchor')
            if enable_sentiment:
                res_row["sentiment_label"] = d.get('sentiment')
                res_row["sentiment_score"] = d.get('sentiment_score')
                res_row["all_scores"] = d.get('all_scores')

            # Add all metadata columns from BigQuery
            for col in row.index:
                if col not in ['transcript_id', 'paragraph_number', 'speaker', 'content']:
                    res_row[col] = row[col]

            # Add content if requested
            if include_content:
                res_row["content"] = text

            all_results.append(res_row)

        # Update previous interaction tracking for next iteration
        previous_interaction = interaction_type
        previous_role = role_label

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

        elapsed_time = time.time() - start_time
        time_str = format_time(elapsed_time)

        # Calculate statistics
        unique_transcripts = results_df['transcript_id'].nunique()
        unique_companies = results_df.get('symbol', pd.Series()).nunique()
        avg_processing_rate = len(df) / elapsed_time

        print(f"\n{'='*80}")
        print(f"[SUCCESS] Analysis Complete!")
        print(f"{'='*80}")
        print(f"\nOutput:")
        print(f"  File: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        print(f"\nStatistics:")
        print(f"  Total records: {len(results_df):,}")
        print(f"  Transcripts: {unique_transcripts}")
        if unique_companies > 0:
            print(f"  Companies: {unique_companies}")
        if 'issue_subtopic' in results_df.columns:
            print(f"  Topics detected: {results_df['issue_subtopic'].notna().sum():,}")
        print(f"\nPerformance:")
        print(f"  Time taken: {time_str}")
        print(f"  Processing rate: {avg_processing_rate:.1f} segments/sec")
        print(f"{'='*80}\n")

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

    # Issue config generation
    parser.add_argument('-p', '--parse-raw', action='store_true',
                       help="If present, generates issue configs from the raw Zignal formatted input CSV; otherwise, uses the intermediate tidy CSV.")
    parser.add_argument('--keyword-file', type=str,
                       help='Path to a keyword inputs CSV file in the intermediate format (issue_area,topic,term,type). Overrides default keyword config.')

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
                       help='Custom limit for number of records to process (will only return complete transcripts)')

    # Transcript selection
    parser.add_argument('--latest', type=int,
                       help='Pull the latest N complete transcripts (overrides --limit and --mode)')
    parser.add_argument('--earliest', type=int,
                       help='Pull the earliest N complete transcripts (overrides --limit and --mode)')

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

    # Enrichment toggles
    parser.add_argument('--no-interaction-type', action='store_true',
                       help='Skip interaction type classification (Question/Answer/Admin)')
    parser.add_argument('--no-role', action='store_true',
                       help='Skip speaker role classification (Analyst/Executive/Operator/Admin)')
    parser.add_argument('--no-topics', action='store_true',
                       help='Skip topic detection (pattern matching + semantic similarity)')
    parser.add_argument('--no-sentiment', action='store_true',
                       help='Skip sentiment analysis (VADER)')
    parser.add_argument('--no-sessions', action='store_true',
                       help='Skip Q&A session grouping')

    # Parallelization options (local execution only)
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use sequential mode)')
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes for parallel processing (default: auto-detect)')
    parser.add_argument('--sentiment-batch-size', type=int,
                       help='Batch size for sentiment analysis (default: auto-detect based on system)')
    parser.add_argument('--classification-batch-size', type=int,
                       help='Batch size for classification (default: auto-detect based on system)')

    args = parser.parse_args()

    # Parse parsing options
    generate_all(from_raw=args.parse_raw, keyword_file=args.keyword_file)

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

    # Parse mode/limit/latest/earliest
    latest = args.latest
    earliest = args.earliest

    if args.test:
        mode = 'test'
    else:
        mode = args.mode

    # Latest and earliest override limit and mode
    if latest or earliest:
        limit = None
    elif args.limit:
        limit = args.limit
    elif mode == 'test':
        limit = 50
    else:
        limit = None

    # Validate that only one of latest/earliest is specified
    if latest and earliest:
        print("ERROR: Cannot specify both --latest and --earliest")
        sys.exit(1)

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
            latest=latest,
            earliest=earliest,
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
            latest=latest,
            earliest=earliest,
            write_to_bq=args.write_to_bq,
            include_content=not args.no_content,
            output_path=args.output,
            use_parallel=not args.no_parallel,
            num_workers=args.workers,
            sentiment_batch_size=args.sentiment_batch_size,
            classification_batch_size=args.classification_batch_size,
            enable_interaction_type=not args.no_interaction_type,
            enable_role=not args.no_role,
            enable_topics=not args.no_topics,
            enable_sentiment=not args.no_sentiment,
            enable_sessions=not args.no_sessions
        )

if __name__ == "__main__":
    main()
