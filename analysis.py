import json
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd
from google.cloud import bigquery
import os
import sys
import time
import re
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from generate_topics import generate_topics_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class AnalysisServerHandler(BaseHTTPRequestHandler):
    """Server to handle health checks and trigger the analysis pipeline."""
    def do_GET(self):
        if self.path == '/':
            # Health check endpoint
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == '/run':
            # Trigger analysis endpoint
            logger.info("Manual trigger received via /run")
            try:
                # Run pipeline in a separate thread to avoid blocking the response 
                # (Cloud Run has a timeout, but for small batches this is fine)
                threading.Thread(target=process_pipeline).start()
                
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Pipeline triggered. Check logs for progress.")
            except Exception as e:
                logger.error(f"Error triggering pipeline: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Redirect http.server logs to our logger
        logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format%args))

def run_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), AnalysisServerHandler)
    logger.info(f"Analysis server listening on port {port}...")
    server.serve_forever()

"""
analysis.py

Production pipeline for Cloud Run. Processes earnings call transcripts from BigQuery,
enriches them with classification and sentiment data, and writes back to BigQuery.

Features:
- Fragment Rejoining
- Q&A Session Clustering
- Role and Interaction Type Classification
- Aspect-Based Sentiment Analysis
"""

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Always regenerate topics.json in production to ensure sync with CSV
generate_topics_json()

current_dir = os.getcwd() 
TOPICS_FILE = os.path.join(current_dir, 'topics.json')

# Threshold for vector similarity (0 to 1). 
SIMILARITY_THRESHOLD = 0.7

# Local Model Paths
INTERACTION_MODEL_PATH = os.path.join(current_dir, "models", "eng_type_class_v1")
ROLE_MODEL_PATH = os.path.join(current_dir, "models", "role_class_v1")
EMBEDDING_MODEL_PATH = os.path.join(current_dir, "models", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")

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
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 500))
# Default to production-like behavior unless explicitly told otherwise in env
PRODUCTION_TESTING = os.environ.get("PRODUCTION_MODE", "false").lower() != "true"

BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched"

if PRODUCTION_TESTING:
    logger.info("RUNNING IN PRODUCTION TESTING MODE (Limited batches)")
else:
    logger.info("RUNNING IN FULL PRODUCTION MODE")

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")
nlp = spacy.load("en_core_web_sm")

def load_model_safely(model_path, model_type="embedding"):
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model path not found: {model_path}")
        print("Ensure models are baked into the Docker image during build.")
        sys.exit(1)
        
    print(f"Loading {model_type} model from {model_path}")
    try:
        if model_type == "embedding":
            return SentenceTransformer(model_path, local_files_only=True)
        else:
            return pipeline("text-classification", model=model_path, local_files_only=True)
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_type} model: {e}")
        sys.exit(1)

# Load all models strictly from local paths
embedder = load_model_safely(EMBEDDING_MODEL_PATH, "embedding")
sentiment_analyzer = load_model_safely(SENTIMENT_MODEL_PATH, "sentiment")
interaction_classifier = load_model_safely(INTERACTION_MODEL_PATH, "interaction")
role_classifier = load_model_safely(ROLE_MODEL_PATH, "role")

print("Models loaded successfully.")

# =================================================================================================
# DATA LOADING & UTILITIES
# =================================================================================================

def load_topics(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Topics file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('topics', [])

topics_data = load_topics(TOPICS_FILE)

# Prepare spaCy Matcher
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
for topic in topics_data:
    label = topic['label']
    patterns = topic.get('patterns', [])
    if patterns:
        matcher.add(label, patterns)

# Pre-compute embeddings for all anchor terms
all_anchors_text = []
anchor_metadata = [] 
for topic in topics_data:
    for anchor in topic.get('anchors', []):
        all_anchors_text.append(anchor)
        anchor_metadata.append((topic['label'], anchor))

if all_anchors_text:
    anchor_embeddings = embedder.encode(all_anchors_text, convert_to_tensor=True)
else:
    anchor_embeddings = None

def rejoin_fragments(df):
    """Rejoins segments split by line breaks."""
    if df.empty:
        return df
    rejoined_rows = []
    current_row = df.iloc[0].to_dict()
    for i in range(1, len(df)):
        next_row = df.iloc[i].to_dict()
        msg = str(current_row['content']).strip()
        is_fragment = not any(msg.endswith(p) for p in ['.', '?', '!', '"', '“', '”'])
        if next_row['transcript_id'] == current_row['transcript_id'] and is_fragment:
            current_row['content'] = msg + " " + str(next_row['content']).strip()
        else:
            rejoined_rows.append(current_row)
            current_row = next_row
    rejoined_rows.append(current_row)
    return pd.DataFrame(rejoined_rows)

# =================================================================================================
# CORE ANALYSIS
# =================================================================================================

def analyze_text(text):
    """Analyzes text for topics and sentiment."""
    doc = nlp(text)
    matches = matcher(doc)
    found_topics = set()
    if matches:
        for match_id, start, end in matches:
            found_topics.add(nlp.vocab.strings[match_id])
        results = []
        for topic in found_topics:
            sentiment = sentiment_analyzer(text, text_pair=topic, top_k=None)
            sentiment.sort(key=lambda x: x['score'], reverse=True)
            results.append({"topic": topic, "sentiment": sentiment[0]['label'], "score": sentiment[0]['score']})
        return results

    if anchor_embeddings is None:
        return []

    query_embedding = embedder.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, anchor_embeddings)[0]
    results = []
    for idx, score in enumerate(cos_scores):
        if score.item() >= SIMILARITY_THRESHOLD:
            results.append({"topic": anchor_metadata[idx][0], "score": score.item()})
    results.sort(key=lambda x: x['score'], reverse=True)
    
    unique = {}
    for r in results:
        if r['topic'] not in unique: unique[r['topic']] = r
    top_3 = list(unique.values())[:3]
    
    final = []
    for r in top_3:
        sentiment = sentiment_analyzer(text, text_pair=r['topic'], top_k=None)
        sentiment.sort(key=lambda x: x['score'], reverse=True)
        final.append({"topic": r['topic'], "sentiment": sentiment[0]['label'], "score": sentiment[0]['score']})
    return final

# =================================================================================================
# MAIN PIPELINE
# =================================================================================================

    logger.info("Models verified on disk. Starting pipeline logic...")
    client = bigquery.Client(project=BQ_PROJECT_ID)
    
    total_processed = 0
    
    try:
        while True:
            logger.info(f"Checking for new data to process in {BQ_SOURCE_TABLE}...")
            # Determine query limit based on testing mode
            current_limit = 20 if PRODUCTION_TESTING else BATCH_SIZE
            
            # Idempotent Query: Process only rows not already in destination
            query = f"""
                SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content
                FROM `{BQ_SOURCE_TABLE}` t
                LEFT JOIN (SELECT DISTINCT transcript_id, paragraph_number FROM `{BQ_DEST_TABLE}`) e
                ON t.transcript_id = e.transcript_id AND t.paragraph_number = e.paragraph_number
                WHERE e.transcript_id IS NULL
                ORDER BY t.transcript_id, t.paragraph_number
                LIMIT {current_limit}
            """
            
            logger.info(f"Executing query: {query}")
            df = client.query(query).to_dataframe()
            
            if df.empty:
                if total_processed == 0:
                    logger.info("No new data to process. All records in source seem to exist in destination.")
                else:
                    logger.info(f"Finished processing all new data. Total segments: {total_processed}")
                break
            
            if PRODUCTION_TESTING:
                logger.info(f"--- PRODUCTION TESTING MODE: Processing {len(df)} segments ---")
            else:
                logger.info(f"--- Processing batch of {len(df)} segments (Total so far: {total_processed}) ---")
        
        df = rejoin_fragments(df)
        print(f"   (After rejoining: {len(df)} segments)")

        all_results = []
        
        # For each batch, we need to know the 'current' session state.
        # Simplest is to check the last row in BQ for this transcript, 
        # but to keep it fast/low-memory, we'll initialize per run and accept 
        # slight session ID resets if batches split in the very middle of a Q&A.
        current_session_id = 0
        current_analyst = "None"
        intro_regex = re.compile(r"(?:question comes from|from the line of|from)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from)", re.IGNORECASE)

        for _, row in df.iterrows():
            text = str(row['content'])
            # Classification
            int_res = interaction_classifier(text[:512])[0]['label']
            role_res = role_classifier(text[:512])[0]['label']
            interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)
            role_label = ROLE_ID_MAP.get(role_res, role_res)

            # Session tracking
            if role_label == "Operator" and ("next question" in text.lower() or "question comes" in text.lower()):
                current_session_id += 1
                match = intro_regex.search(text)
                current_analyst = match.group(1).strip() if match else "Unknown Analyst"

            detected = analyze_text(text)
            if not detected: detected = [{"topic": None, "sentiment": None, "score": None}]

            for d in detected:
                all_results.append({
                    "transcript_id": row['transcript_id'],
                    "paragraph_number": row['paragraph_number'],
                    "speaker": row['speaker'],
                    "qa_session_id": current_session_id,
                    "qa_session_label": current_analyst,
                    "interaction_type": interaction_type,
                    "role": role_label,
                    "topic": d['topic'],
                    "sentiment_label": d['sentiment'],
                    "sentiment_score": d['score']
                })

        if all_results:
            res_df = pd.DataFrame(all_results)
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            client.load_table_from_dataframe(res_df, BQ_DEST_TABLE, job_config=job_config).result()
            logger.info(f"   Completed enrichment. Uploaded {len(res_df)} rows to {BQ_DEST_TABLE}")
            total_processed += len(df)
        else:
            logger.info("   No topics detected in this batch.")
            total_processed += len(df)
            
        # Exit if in testing mode after first batch
        if PRODUCTION_TESTING:
            logger.info("Production testing complete. Set 'PRODUCTION_MODE=true' env var for full run.")
            break
    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    run_server()
