import os
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def download_with_retry(model_name, download_func, max_retries=3):
    """Downloads a model with a simple retry mechanism for 429 errors."""
    for i in range(max_retries):
        try:
            return download_func()
        except Exception as e:
            if "429" in str(e) and i < max_retries - 1:
                wait = (i + 1) * 30
                print(f"Rate limited (429) downloading {model_name}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed to download {model_name}: {e}")
                raise e

def download_models():
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download Sentence Transformer model
    print("Attempting to download all-MiniLM-L6-v2...")
    embed_model_path = os.path.join(models_dir, "all-MiniLM-L6-v2")
    
    def dl_embed():
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embedder.save(embed_model_path)
    
    download_with_retry("all-MiniLM-L6-v2", dl_embed)
    print(f"Saved to {embed_model_path}")

    # 2. Download Sentiment model
    print("Attempting to download yangheng/deberta-v3-base-absa-v1.1...")
    sentiment_model_path = os.path.join(models_dir, "deberta-v3-base-absa-v1.1")
    
    def dl_sentiment():
        sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
        sentiment_analyzer.save_pretrained(sentiment_model_path)
    
    download_with_retry("yangheng/deberta-v3-base-absa-v1.1", dl_sentiment)
    print(f"Saved to {sentiment_model_path}")

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
