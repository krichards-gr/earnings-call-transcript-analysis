from transformers import pipeline
import os
import torch

current_dir = os.getcwd()
SENTIMENT_MODEL_PATH = os.path.join(current_dir, "models", "deberta-v3-base-absa-v1.1")

print(f"Loading model from {SENTIMENT_MODEL_PATH}...")
try:
    # Test loading with local_files_only
    sentiment_analyzer = pipeline("text-classification", model=SENTIMENT_MODEL_PATH)
    print("Model loaded successfully.")
    
    tokenizer = sentiment_analyzer.tokenizer
    print(f"Tokenizer type: {type(tokenizer)}")
    
    # Test batch encoding
    batch_texts = ["Hello world", "Test sentence"]
    batch_pairs = ["Topic A", "Topic B"]
    
    print("Testing tokenizer callback...")
    inputs = tokenizer(
        batch_texts,
        text_pair=batch_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    print("Tokenizer call successful!")
    print(f"Inputs keys: {inputs.keys()}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
