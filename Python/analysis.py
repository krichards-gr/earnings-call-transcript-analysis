import json
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os
import sys

# =================================================================================================
# CONFIGURATION & SETUP
# =================================================================================================

# Define paths relative to the script location to ensure portability
current_dir = os.path.dirname(os.path.abspath(__file__))
TOPICS_FILE = os.path.join(current_dir, 'topics.json')

# Threshold for vector similarity (0 to 1). 
# Text with similarity score below this will not be considered a match for that anchor.
SIMILARITY_THRESHOLD = 0.3

# =================================================================================================
# MODEL LOADING
# =================================================================================================

print("Loading models... (This may take a moment on first run)")

# Load spaCy model for EXACT match patterns.
# We use 'en_core_web_sm' because we only need efficient tokenization and simple matching,
# not deep syntactic dependency parsing.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Load SentenceTransformer model for VECTOR SIMILARITY.
# 'all-MiniLM-L6-v2' is a high-speed, low-memory model ideal for near-real-time usage.
# It maps sentences & paragraphs to a 384 dimensional dense vector space.
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    sys.exit(1)

# Load Aspect-Based Sentiment Analysis (ABSA) model
# We use 'yangheng/deberta-v3-base-absa-v1.1' as it is a general-purpose model
# that supports determining sentiment relative to a specific aspect/topic.
try:
    sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
except Exception as e:
    print(f"Error loading ABSA model: {e}")
    sys.exit(1)

print("Models loaded successfully.")

# =================================================================================================
# DATA LOADING
# =================================================================================================

def load_topics(filepath):
    """
    Reads the topics JSON file and returns the list of topics.
    
    Expected JSON structure:
    {
      "topics": [
        {
          "label": "TOPIC_NAME",
          "patterns": [[{"LOWER": "term"}]],  # spaCy match patterns
          "anchors": ["phrase one", "phrase two"] # Sentences for vector comparison
        },
        ...
      ]
    }
    """
    if not os.path.exists(filepath):
        print(f"Error: Topics file not found at {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('topics', [])

topics_data = load_topics(TOPICS_FILE)

# Prepare spaCy Matcher
# We initialize the Matcher with the vocab from our loaded spacy model.
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Register all patterns from our topics file into the spaCy matcher.
# This allows us to scan the doc once and find all exact matches efficiently.
for topic in topics_data:
    label = topic['label']
    patterns = topic.get('patterns', [])
    if patterns:
        matcher.add(label, patterns)

# Pre-compute embeddings for all anchor terms to save time during inference.
# We flatten the list to pairs of (topic_label, anchor_text) and then encode the texts.
anchor_embeddings_map = []
all_anchors_text = []
anchor_metadata = [] # Stores (topic_label, anchor_text) corresponding to each embedding

for topic in topics_data:
    for anchor in topic.get('anchors', []):
        all_anchors_text.append(anchor)
        anchor_metadata.append((topic['label'], anchor))

if all_anchors_text:
    # Encode all anchors in one batch for efficiency
    anchor_embeddings = embedder.encode(all_anchors_text, convert_to_tensor=True)
else:
    anchor_embeddings = None

# =================================================================================================
# ANALYSIS LOGIC
# =================================================================================================

def analyze_text(text):
    """
    Analyzes the provided text to identify topics.
    
    Strategy:
    1. EXACT MATCH: Check against "key terms" (patterns) using spaCy.
       If matches are found, return those topics immediately (high confidence).
       
    2. SIMILARITY MATCH: If no exact matches, compare text vector against anchor term vectors.
       Return top 3 matches that exceed SIMILARITY_THRESHOLD.
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Exact Match (Key Terms)
    # -------------------------------------------------------------------------
    doc = nlp(text)
    matches = matcher(doc)
    
    found_topics = set()
    
    # process matches
    if matches:
        print(f"  [DEBUG] Found {len(matches)} exact matches.")
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation (the topic label)
            span = doc[start:end]  # The matched span
            print(f"    - Match: '{span.text}' -> Topic: {string_id}")
            found_topics.add(string_id)
        
        # Logic: If key terms are present, we are 100% sure of the topic.
        # Check sentiment for each found topic
        results = []
        for topic in found_topics:
            try:
                # The model expects text (context) and text_pair (aspect)
                # Pass text as positional argument, text_pair as kwarg
                # top_k=None returns all scores
                sentiment = sentiment_analyzer(text, text_pair=topic, top_k=None)
                
                # sentiment is a list of dicts: [{'label': 'Positive', 'score': 0.9}, {'label': 'Neutral', ...}]
                # Sort by score desc to be safe, though usually returned sorted
                # But we want to capture the simplified top label AND the full breakdown
                sentiment.sort(key=lambda x: x['score'], reverse=True)
                
                top_label = sentiment[0]['label']
                top_score = sentiment[0]['score']
                
                # Create a compact string representation of all scores for display
                # e.g. "Pos: 0.60, Neu: 0.30, Neg: 0.10"
                scores_str = ", ".join([f"{s['label'][:3]}: {s['score']:.2f}" for s in sentiment])

                results.append({
                    "topic": topic,
                    "sentiment": top_label,
                    "score": top_score,
                    "all_scores": scores_str
                })
            except Exception as e:
                print(f"    [WARN] Sentiment analysis failed for topic '{topic}': {e}")
                results.append({"topic": topic, "sentiment": "Unknown", "score": 0.0})
        
        return results

    # -------------------------------------------------------------------------
    # STEP 2: Vector Similarity (Anchor Terms)
    # -------------------------------------------------------------------------
    # Only proceed if we have anchors defined
    if anchor_embeddings is None:
        return []

    print("  [DEBUG] No exact matches found. Checking vector similarity...")
    
    # Encode the input text
    # convert_to_tensor=True ensures we get a format suitable for cosine_similarity
    query_embedding = embedder.encode(text, convert_to_tensor=True)
    
    # Calculate cosine similarity between input text and ALL anchor embeddings
    # util.cos_sim returns a matrix, we want the first row [0] since we only have one query
    cos_scores = util.cos_sim(query_embedding, anchor_embeddings)[0]
    
    # Combine scores with metadata
    # results will be a list of (score, topic_label, anchor_text)
    results = []
    for idx, score in enumerate(cos_scores):
        score_val = score.item() # convert tensor to float
        if score_val >= SIMILARITY_THRESHOLD:
            topic_label, anchor_text = anchor_metadata[idx]
            results.append({
                "topic": topic_label,
                "score": score_val,
                "matched_anchor": anchor_text
            })
            
    # Sort by score descending (highest similarity first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate by topic (keep highest score per topic)
    unique_results = {}
    for r in results:
        t = r['topic']
        if t not in unique_results:
            unique_results[t] = r
    
    # Convert back to list and sort again
    final_results = list(unique_results.values())
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top 3
    top_3 = final_results[:3]
    
    if top_3:
        print(f"  [DEBUG] Vector matches found: {len(top_3)}")
        for r in top_3:
            print(f"    - Topic: {r['topic']} (Score: {r['score']:.4f}, Anchor: '{r['matched_anchor']}')")
        # For vector matches, also calculate sentiment
        final_output = []
        for r in top_3:
            topic = r['topic']
            try:
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
                    "similarity_score": r['score'], # Keep the similarity score for reference
                    "matched_anchor": r['matched_anchor']
                })
            except Exception as e:
                 print(f"    [WARN] Sentiment analysis failed for topic '{topic}': {e}")
                 final_output.append({
                    "topic": topic, 
                    "sentiment": "Unknown", 
                    "score": 0.0,
                    "similarity_score": r['score'],
                    "matched_anchor": r['matched_anchor']
                })

        return final_output
            
    return []

# =================================================================================================
# MAIN EXECUTION
# =================================================================================================

if __name__ == "__main__":
    # If a file argument is provided, read it. Otherwise use the default sample.
    input_file = os.path.join(current_dir, 'inputs', 'sample_transcript.txt')
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    print(f"Reading input from: {input_file}")
    
    if not os.path.exists(input_file):
        print("Error: Input file not found.")
        sys.exit(1)
        
    with open(input_file, 'r', encoding='utf-8') as f:
        # We read paragraph by paragraph to simulate analyzing chunks of a transcript
        content = f.read()
        
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    print(f"\nAnalyzing {len(paragraphs)} paragraphs...\n")
    print("-" * 60)
    
    for i, paragraph in enumerate(paragraphs):
        print(f"Paragraph {i+1}: \"{paragraph[:100]}...\"")
        detected_topics = analyze_text(paragraph)
        print(f"Result: {detected_topics}")
        if detected_topics:
            for dt in detected_topics:
                print(f"    -> Topic: {dt['topic']}")
                print(f"       Sentiment: {dt['sentiment']} ({dt['score']:.2f})")
                print(f"       Breakdown: [{dt.get('all_scores', '')}]")
        print("-" * 60)
