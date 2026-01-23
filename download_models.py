import os
import time
import sys
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
                print(f"FAILED to download {model_name} from Hugging Face: {e}")
                raise e

def download_from_gcs(bucket_name, model_name, destination_base_dir):
    """Downloads a model folder from GCS."""
    print(f"Checking GCS for model: {model_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        prefix = f"models/{model_name}/"
        
        # Test if bucket is accessible
        if not bucket.exists():
            print(f"  ERROR: Bucket '{bucket_name}' not found or inaccessible.")
            return False

        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            print(f"  Warning: No blobs found with prefix '{prefix}' in bucket '{bucket_name}'.")
            return False

        dest_dir = os.path.join(destination_base_dir, model_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"  Downloading {len(blobs)} files...")
        for blob in blobs:
            rel_path = blob.name[len(prefix):]
            if not rel_path: continue 
            dest_file = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            blob.download_to_filename(dest_file)
        return True
    except Exception as e:
        print(f"  GCS pull FAILED for {model_name}: {e}")
        return False

def download_models():
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    failed_models = []

    for local_name, hf_source in MODELS_TO_SYNC.items():
        # 1. Try GCS first (Faster, no rate limits)
        if download_from_gcs(GCS_BUCKET_NAME, local_name, models_dir):
            print(f"Successfully pulled {local_name} from GCS.")
            continue
        
        # 2. Fallback to HF (only if source is defined)
        if hf_source:
            print(f"Model {local_name} NOT in GCS. Falling back to Hugging Face...")
            dest_path = os.path.join(models_dir, local_name)
            
            try:
                def dl_func():
                    if "MiniLM" in local_name:
                        m = SentenceTransformer(hf_source)
                        m.save(dest_path)
                    else:
                        m = pipeline("text-classification", model=hf_source)
                        m.save_pretrained(dest_path)
                
                download_with_retry(local_name, dl_func)
                print(f"Successfully saved {local_name} from Hugging Face.")
            except Exception as e:
                print(f"CRITICAL FAILURE: Could not get model '{local_name}': {e}")
                failed_models.append(local_name)
        else:
            print(f"CRITICAL FAILURE: Custom model '{local_name}' missing from GCS and has NO Hugging Face fallback.")
            failed_models.append(local_name)

    if failed_models:
        print(f"\nERROR: The following models failed to download: {', '.join(failed_models)}")
        print("Build cannot proceed without all required models.")
        sys.exit(1)

    print("\nModel synchronization complete. All models ready.")

if __name__ == "__main__":
    download_models()
