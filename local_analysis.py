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
import torch
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

The script outputs results to a local CSV and optionally to Google BigQuery.
It is synchronized with the production logic (analysis.py) but configured for local execution.
"""

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Automatically regenerate topics.json from topic_definitions.csv
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
WRITE_TO_BQ = False 
INCLUDE_CONTENT = True 
BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched_local"

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")
nlp = spacy.load("en_core_web_sm")

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
# Create a map for quick exclusion lookups
EXCLUSIONS_MAP = {t['label']: t.get('exclusions', []) for t in topics_data}

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
                # Check for exclusionary terms
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
                # Check for exclusionary terms
                exclusions = EXCLUSIONS_MAP.get(topic, [])
                if any(ext.lower() in text.lower() for ext in exclusions):
                    # We don't print here to avoid flooding logs for vector matches
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
                    
                    # Convert labels to all_scores string for local detail
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
# MAIN EXECUTION
# =================================================================================================

def run_local_analysis():
    print(f"Connecting to BigQuery...")
    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Simplified read for local processing
    print(f"Reading SAMPLE (Limit 50) from {BQ_SOURCE_TABLE}...")
    
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
    
    df = rejoin_fragments(df)
    print(f"Processing {len(df)} rows after rejoining. Starting analysis...")
    
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
    
    # Session tracking regexes (Sync with analysis.py)
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
        # Reset session tracking if we've moved to a new transcript
        if last_transcript_id is not None and row['transcript_id'] != last_transcript_id:
            current_session_id = 0
            current_analyst = "None"
        last_transcript_id = row['transcript_id']

        int_res = int_results[i]['label']
        role_res = role_results[i]['label']
        interaction_type = INTERACTION_ID_MAP.get(int_res, int_res)
        role_label = ROLE_ID_MAP.get(role_res, role_res)
        text = str(row['content'])

        # Session tracking
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
            print(f"      [SESSION] New Session {current_session_id}: {current_analyst}")

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
                "sentiment_label": d.get('sentiment'),
                "sentiment_score": d.get('sentiment_score'),
                "all_scores": d.get('all_scores'),
                "similarity_score": d.get('similarity_score'),
                "matched_anchor": d.get('matched_anchor'),
                "report_date": row['report_date'],
                "symbol": row['symbol']
            }
            if INCLUDE_CONTENT:
                res_row["content"] = text
            all_results.append(res_row)

    # 4. Save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'local_analysis_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nAnalysis complete. Results saved locally to: {output_path}")
        
        if WRITE_TO_BQ:
            print(f"Writing results to BigQuery: {BQ_DEST_TABLE}")
            # ... BigQuery loading logic ...
            pass
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    run_local_analysis()
