import json
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pandas as pd
from google.cloud import bigquery
import os
import sys

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

current_dir = os.getcwd() 
TOPICS_FILE = os.path.join(current_dir, 'topics.json')

# Threshold for vector similarity (0 to 1). 
SIMILARITY_THRESHOLD = 0.7

# BigQuery Configuration
BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models...")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load ABSA model
sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")

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

# =================================================================================================
# BIGQUERY PROCESSING
# =================================================================================================

def run_local_analysis():
    print(f"Connecting to BigQuery...")
    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Simple read - LIMIT to 50 for quick iterative testing
    print(f"Reading SAMPLE (Limit 50) from {BQ_SOURCE_TABLE}...")
    
    query = f"""
        SELECT transcript_id, paragraph_number, content 
        FROM `{BQ_SOURCE_TABLE}`
        LIMIT 50
    """
    
    df = client.query(query).to_dataframe()

    if df.empty:
        print("No data found in source table.")
        return
        
    print(f"Loaded {len(df)} rows. Starting analysis...")
    
    all_results = []
    
    for index, row in df.iterrows():
        text = row['content']
        transcript_id = row['transcript_id']
        paragraph = row['paragraph_number']
        
        if not isinstance(text, str) or not text.strip():
            continue
            
        print(f"\n--- [Row {index}] Transcript: {transcript_id} | Para: {paragraph} ---")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        detected = analyze_text(text)
        
        if detected:
            for d in detected:
                print(f"   >>> FOUND: Topic={d['topic']}, Sent={d['sentiment']} ({d['score']:.2f})")
                
                # Combine metadata with results
                result_row = {
                    "transcript_id": transcript_id,
                    "paragraph_number": paragraph,
                    "content": text,
                    "topic": d['topic'],
                    "sentiment": d['sentiment'],
                    "score": d['score'],
                    "all_scores": d['all_scores'],
                    "similarity_score": d.get('similarity_score'),
                    "matched_anchor": d.get('matched_anchor')
                }
                all_results.append(result_row)
        else:
            print("   (No topics detected)")
             # Add row with empty topic/sentiment info
            result_row = {
                "transcript_id": transcript_id,
                "paragraph_number": paragraph,
                "content": text,
                "topic": None,
                "sentiment": None,
                "score": None,
                "all_scores": None,
                "similarity_score": None,
                "matched_anchor": None
            }
            all_results.append(result_row)

    # Save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Ensure output directory exists
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'local_analysis_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nAnalysis complete. Results saved to: {output_path}")
        print(f"Total rows analyzed: {len(df)}")
        print(f"Total topics detected: {len(results_df)}")
    else:
        print("\nAnalysis complete. No topics detected in the sample.")

if __name__ == "__main__":
    run_local_analysis()
