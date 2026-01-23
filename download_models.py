import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def download_models():
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Download Sentence Transformer model
    print("Downloading all-MiniLM-L6-v2...")
    embed_model_path = os.path.join(models_dir, "all-MiniLM-L6-v2")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embedder.save(embed_model_path)
    print(f"Saved to {embed_model_path}")

    # 2. Download Sentiment model
    print("Downloading yangheng/deberta-v3-base-absa-v1.1...")
    sentiment_model_path = os.path.join(models_dir, "deberta-v3-base-absa-v1.1")
    # We use pipeline to download, then save the underlying model and tokenizer
    sentiment_analyzer = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
    sentiment_analyzer.save_pretrained(sentiment_model_path)
    print(f"Saved to {sentiment_model_path}")

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
