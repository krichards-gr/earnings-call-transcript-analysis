import requests
import sys

def test_cloud_pipeline(base_url="http://localhost:8080"):
    """
    Test the cloud analysis pipeline (AnalysisServer).
    Verifies health check and trigger endpoints.
    """
    print(f"=== Testing Cloud Analysis Pipeline ({base_url}) ===")
    
    # 1. Health Check
    print("Checking health endpoint (/) ...")
    try:
        resp = requests.get(f"{base_url}/", timeout=5)
        if resp.status_code == 200 and resp.text == "OK":
            print("Success: Health check OK.")
        else:
            print(f"Failure: Health check returned {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Error connecting to health endpoint: {e}")
        print("Tip: Ensure the server is running locally on port 8080 if testing 'localhost'.")
        return

    # 2. Trigger Analysis
    print("Triggering pipeline (/run) ...")
    try:
        # Note: This might take a while depending on the batch size
        print("Sent trigger, waiting for response (this may take 10-60s)...")
        resp = requests.get(f"{base_url}/run", timeout=120)
        if resp.status_code == 200:
            print(f"Success: Pipeline trigger OK. Response: {resp.text}")
        elif resp.status_code == 500:
            print(f"Server Error (500): {resp.text}")
        else:
            print(f"Failure: Pipeline trigger returned {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Error connecting to trigger endpoint: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    test_cloud_pipeline(url)
