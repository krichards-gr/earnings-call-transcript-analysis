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
from generate_topics import generate_topics_json

"""
local_analysis.py

This script performs an enriched analysis of earnings call transcripts, combining:
1. Topic Matching (Exact Match & Vector Similarity)
2. Aspect-Based Sentiment Analysis (ABSA)
3. Interaction Type Classification (Admin, Answer, Question)
4. Role Classification (Admin, Analyst, Executive, Operator)
5. Q&A Session Clustering (Tracing questions to answers)
6. Transcript Fragmentation Rejoining (Fixing split segments)

The script can output results to a local CSV and optionally to Google BigQuery.
"""

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Automatically regenerate topics.json from topic_definitions.csv
generate_topics_json()

current_dir = os.getcwd() 
TOPICS_FILE = os.path.join(current_dir, 'topics.json')

# Threshold for vector similarity (0 to 1). 
# Higher values mean stricter matching.
SIMILARITY_THRESHOLD = 0.7

# New Local Classification Models
# These paths point to the local directories containing the pretrained models.
INTERACTION_MODEL_PATH = os.path.join(current_dir, "models", "eng_type_class_v1")
ROLE_MODEL_PATH = os.path.join(current_dir, "models", "role_class_v1")
EMBEDDING_MODEL_PATH = os.path.join(current_dir, "models", "all-MiniLM-L6-v2")
SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")

# Human-Readable Label Mappings
# Maps internal model output labels (e.g., LABEL_1) to descriptive names.
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
WRITE_TO_BQ = False # Set to True to enable writing results back to BigQuery
INCLUDE_CONTENT = True # Set to True to include the original 'content' in the output results
BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched_local"

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load embedding model (prefer local)
if os.path.exists(EMBEDDING_MODEL_PATH):
    print(f"Loading embedding model from {EMBEDDING_MODEL_PATH}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
else:
    print("Loading embedding model from Hugging Face")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load sentiment model (prefer local)
if os.path.exists(SENTIMENT_MODEL_PATH):
    print(f"Loading sentiment model from {SENTIMENT_MODEL_PATH}")
    sentiment_analyzer = pipeline("text-classification", model=SENTIMENT_MODEL_PATH)
else:
    print("Loading sentiment model from Hugging Face")
    sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")

# Load Interaction Type model
interaction_classifier = pipeline("text-classification", model=INTERACTION_MODEL_PATH)

# Load Role model
role_classifier = pipeline("text-classification", model=ROLE_MODEL_PATH)

print("Models loaded successfully.")

# =================================================================================================
# DATA LOADING
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
anchor_embeddings_map = []
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

# =================================================================================================
# ANALYSIS LOGIC
# =================================================================================================

def analyze_text(text):
    """
    Analyzes the provided text to identify topics.
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Exact Match (Key Terms)
    # -------------------------------------------------------------------------
    doc = nlp(text)
    matches = matcher(doc)
    
    found_topics = set()
    
    if matches:
        print(f"  [DEBUG] Found {len(matches)} exact matches.")
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id] 
            span = doc[start:end]
            print(f"    - Match: '{span.text}' -> Topic: {string_id}")
            found_topics.add(string_id)
        
        results = []
        for topic in found_topics:
            # DIRECT CALL - NO ERROR HANDLING
            sentiment = sentiment_analyzer(text, text_pair=topic, top_k=None)
            sentiment.sort(key=lambda x: x['score'], reverse=True)
            
            top_label = sentiment[0]['label']
            top_score = sentiment[0]['score']
            scores_str = ", ".join([f"{s['label'][:3]}: {s['score']:.2f}" for s in sentiment])

            results.append({
                "topic": topic,
                "sentiment": top_label,
                "score": top_score,
                "all_scores": scores_str
            })
        
        return results

    # -------------------------------------------------------------------------
    # STEP 2: Vector Similarity (Anchor Terms)
    # -------------------------------------------------------------------------
    if anchor_embeddings is None:
        return []

    # print("  [DEBUG] No exact matches found. Checking vector similarity...")
    
    query_embedding = embedder.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, anchor_embeddings)[0]
    
    results = []
    for idx, score in enumerate(cos_scores):
        score_val = score.item()
        if score_val >= SIMILARITY_THRESHOLD:
            topic_label, anchor_text = anchor_metadata[idx]
            results.append({
                "topic": topic_label,
                "score": score_val,
                "matched_anchor": anchor_text
            })
            
    results.sort(key=lambda x: x['score'], reverse=True)
    
    unique_results = {}
    for r in results:
        t = r['topic']
        if t not in unique_results:
            unique_results[t] = r
    
    final_results = list(unique_results.values())
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    top_3 = final_results[:3]
    
    if top_3:
        print(f"  [DEBUG] Vector matches found: {len(top_3)}")
        for r in top_3:
            print(f"    - Topic: {r['topic']} (Score: {r['score']:.4f}, Anchor: '{r['matched_anchor']}')")
        
        final_output = []
        for r in top_3:
            topic = r['topic']
            
            # DIRECT CALL - NO ERROR HANDLING
            sentiment = sentiment_analyzer(text, text_pair=topic, top_k=None)
            sentiment.sort(key=lambda x: x['score'], reverse=True)
            
            top_label = sentiment[0]['label']
            top_score = sentiment[0]['score']
            scores_str = ", ".join([f"{s['label'][:3]}: {s['score']:.2f}" for s in sentiment])
            
            final_output.append({
                "topic": topic,
                "sentiment": top_label,
                "score": top_score,
                "all_scores": scores_str,
                "similarity_score": r['score'], 
                "matched_anchor": r['matched_anchor']
            })

        return final_output
            
    return []

def rejoin_fragments(df):
    """
    Cleans fragmented transcript segments.
    
    Earnings call data may be split incorrectly during ingestion (e.g., at every \n).
    This function rejoins consecutive rows for the same transcript if the previous
    row does not end with sentence-ending punctuation.
    """
    if df.empty:
        return df
    
    print(f"Checking for fragments in {len(df)} rows...")
    
    rejoined_rows = []
    current_row = df.iloc[0].to_dict()
    
    for i in range(1, len(df)):
        next_row = df.iloc[i].to_dict()
        
        # Check if same transcript
        same_transcript = next_row['transcript_id'] == current_row['transcript_id']
        
        # Heuristic for fragment: doesn't end with . ? ! " ”
        content = str(current_row['content']).strip()
        is_fragment = not any(content.endswith(p) for p in ['.', '?', '!', '"', '“', '”'])
        
        if same_transcript and is_fragment:
            # Merge content
            current_row['content'] = content + " " + str(next_row['content']).strip()
        else:
            # Save current and move to next
            rejoined_rows.append(current_row)
            current_row = next_row
            
    rejoined_rows.append(current_row)
    rejoined_df = pd.DataFrame(rejoined_rows)
    
    diff = len(df) - len(rejoined_df)
    if diff > 0:
        print(f"Rejoined {diff} fragmented segments.")
    
    return rejoined_df

# =================================================================================================
# BIGQUERY PROCESSING
# =================================================================================================

def run_local_analysis():
    print(f"Connecting to BigQuery...")
    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Simple read - LIMIT to 50 for quick iterative testing
    print(f"Reading SAMPLE (Limit 50) from {BQ_SOURCE_TABLE} joined with {BQ_METADATA_TABLE}...")
    
    query = f"""
        SELECT 
            t.transcript_id, 
            t.paragraph_number, 
            t.speaker,
            t.content,
            m.report_date,
            m.symbol
        FROM `{BQ_SOURCE_TABLE}` t
        JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
        ORDER BY m.report_date DESC, m.symbol, t.paragraph_number
        LIMIT 50
    """
    
    df = client.query(query).to_dataframe()

    if df.empty:
        print("No data found in source table.")
        return
    
    # Merge fragments before analysis
    df = rejoin_fragments(df)
        
    print(f"Processing {len(df)} rows after rejoining. Starting analysis...")
    
    all_results = []
    
    # Q&A Session Tracking
    current_session_id = 0
    current_analyst = "None"
    
    # Regex for Operator intro: "question comes from the line of [Name] with [Bank]"
    # or "Your next question comes from [Name] from [Bank]"
    intro_regex = re.compile(r"(?:question comes from|from the line of|from)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from)", re.IGNORECASE)
    
    for index, row in df.iterrows():
        text = row['content']
        transcript_id = row['transcript_id']
        paragraph = row['paragraph_number']
        speaker = row.get('speaker')
        
        if not isinstance(text, str) or not text.strip():
            continue
            
        print(f"\n--- [Row {index}] Transcript: {transcript_id} | Para: {paragraph} ---")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        # Run new models
        start_time = time.time()
        interaction_res = interaction_classifier(text[:512]) # Truncate to 512 for safely
        interaction_time = time.time() - start_time
        
        start_time = time.time()
        role_res = role_classifier(text[:512])
        role_time = time.time() - start_time
        
        raw_interaction = interaction_res[0]['label']
        raw_role = role_res[0]['label']
        
        interaction_type = INTERACTION_ID_MAP.get(raw_interaction, raw_interaction)
        role_label = ROLE_ID_MAP.get(raw_role, raw_role)
        
        # Session Tracking Logic
        # If Operator is introducing someone, start new session
        if role_label == "Operator" and ("next question" in text.lower() or "question comes" in text.lower()):
            current_session_id += 1
            match = intro_regex.search(text)
            if match:
                current_analyst = match.group(1).strip()
            else:
                current_analyst = "Unknown Analyst"
            print(f"   [SESSION] New Session {current_session_id}: {current_analyst}")
        
        print(f"   [TIME] Interaction: {interaction_time:.3f}s | Role: {role_time:.3f}s")

        detected = analyze_text(text)
        
        if detected:
            for d in detected:
                print(f"   >>> FOUND: Topic={d['topic']}, Sent={d['sentiment']} ({d['score']:.2f}) | Interaction={interaction_type} | Role={role_label}")
                
                # Combine metadata with results
                result_row = {
                    "transcript_id": transcript_id,
                    "paragraph_number": paragraph,
                    "speaker": speaker,
                    "qa_session_id": current_session_id,
                    "qa_session_label": current_analyst,
                    "interaction_type": interaction_type,
                    "role": role_label,
                    "topic": d['topic'],
                    "sentiment": d['sentiment'],
                    "score": d['score'],
                    "all_scores": d['all_scores'],
                    "similarity_score": d.get('similarity_score'),
                    "matched_anchor": d.get('matched_anchor')
                }
                if INCLUDE_CONTENT:
                    result_row["content"] = text
                    
                all_results.append(result_row)
        else:
            print(f"   (No topics detected) | Interaction={interaction_type} | Role={role_label}")
            # Add row with empty topic/sentiment info
            result_row = {
                "transcript_id": transcript_id,
                "paragraph_number": paragraph,
                "speaker": speaker,
                "qa_session_id": current_session_id,
                "qa_session_label": current_analyst,
                "interaction_type": interaction_type,
                "role": role_label,
                "topic": None,
                "sentiment": None,
                "score": None,
                "all_scores": None,
                "similarity_score": None,
                "matched_anchor": None
            }
            if INCLUDE_CONTENT:
                result_row["content"] = text
                
            all_results.append(result_row)

    # Save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Ensure output directory exists
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'local_analysis_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nAnalysis complete. Results saved locally to: {output_path}")
        
        # Write back to BigQuery
        if WRITE_TO_BQ:
            print(f"Writing results to BigQuery table: {BQ_DEST_TABLE}...")
            
            # Mapping internal columns to BQ schema
            # Current result columns: transcript_id, paragraph_number, topic, sentiment, score, all_scores, similarity_score, matched_anchor
            # Map sentiment -> sentiment_label, score -> sentiment_score
            bq_df = results_df.rename(columns={
                "sentiment": "sentiment_label",
                "score": "sentiment_score"
            })
            
            schema = [
                bigquery.SchemaField("transcript_id", "STRING"),
                bigquery.SchemaField("paragraph_number", "INTEGER"),
                bigquery.SchemaField("speaker", "STRING"),
                bigquery.SchemaField("qa_session_id", "INTEGER"),
                bigquery.SchemaField("qa_session_label", "STRING"),
                bigquery.SchemaField("interaction_type", "STRING"),
                bigquery.SchemaField("role", "STRING"),
                bigquery.SchemaField("topic", "STRING"),
                bigquery.SchemaField("sentiment_label", "STRING"),
                bigquery.SchemaField("sentiment_score", "FLOAT"),
                bigquery.SchemaField("all_scores", "STRING"),
                bigquery.SchemaField("similarity_score", "FLOAT"),
                bigquery.SchemaField("matched_anchor", "STRING"),
            ]
            
            if INCLUDE_CONTENT:
                schema.append(bigquery.SchemaField("content", "STRING"))
            
            job_config = bigquery.LoadJobConfig(
                # NOTE: We use WRITE_TRUNCATE here to replace the table for the first run
                # as requested. For subsequent runs where you want to append data, 
                # change this to "WRITE_APPEND".
                write_disposition="WRITE_TRUNCATE", 
                schema=schema
            )
            
            try:
                job = client.load_table_from_dataframe(bq_df, BQ_DEST_TABLE, job_config=job_config)
                job.result()
                print(f"Successfully replaced/updated {BQ_DEST_TABLE} with {len(bq_df)} enriched rows.")
            except Exception as e:
                print(f"Error writing to BigQuery: {e}")
        else:
            print("\nBigQuery write is DISABLED (WRITE_TO_BQ = False). Skipping cloud write.")
            
        print(f"Total rows analyzed: {len(df)}")
        print(f"Total topics detected: {len(results_df)}")
    else:
        print("\nAnalysis complete. No topics detected in the sample.")

if __name__ == "__main__":
    run_local_analysis()
