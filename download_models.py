import os
import time
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from google.cloud import storage

# --- CONFIGURATION ---
GCS_BUCKET_NAME = "earnings-analysis-models" 

# Maps local folder names to their HF source (only used if GCS pull fails)
MODELS_TO_SYNC = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "deberta-v3-base-absa-v1.1": "yangheng/deberta-v3-base-absa-v1.1",
    "eng_type_class_v1": None, # Custom model (GCS only)
    "role_class_v1": None      # Custom model (GCS only)
}
# ---------------------

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

def download_from_gcs(bucket_name, model_name, destination_base_dir):
    """Downloads a model folder from GCS."""
    print(f"Checking GCS for model: {model_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        prefix = f"models/{model_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            return False

        dest_dir = os.path.join(destination_base_dir, model_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        for blob in blobs:
            rel_path = blob.name[len(prefix):]
            if not rel_path: continue 
            dest_file = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            blob.download_to_filename(dest_file)
        return True
    except Exception as e:
        print(f"  GCS pull failed for {model_name}: {e}")
        return False

def download_models():
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    for local_name, hf_source in MODELS_TO_SYNC.items():
        # 1. Try GCS first (Faster, no rate limits)
        if download_from_gcs(GCS_BUCKET_NAME, local_name, models_dir):
            print(f"Successfully pulled {local_name} from GCS.")
            continue
        
        # 2. Fallback to HF (only if source is defined)
        if hf_source:
            print(f"Model {local_name} not in GCS. Falling back to Hugging Face...")
            dest_path = os.path.join(models_dir, local_name)
            
            def dl_func():
                if "MiniLM" in local_name:
                    m = SentenceTransformer(hf_source)
                    m.save(dest_path)
                else:
                    m = pipeline("text-classification", model=hf_source)
                    m.save_pretrained(dest_path)
            
            download_with_retry(local_name, dl_func)
            print(f"Saved {local_name} from Hugging Face.")
        else:
            print(f"CRITICAL: Custom model {local_name} missing from GCS and has no HF fallback.")

    print("\nModel synchronization complete.")

if __name__ == "__main__":
    download_models()
