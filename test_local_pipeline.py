import pandas as pd
import os
import subprocess
import sys

def test_local_pipeline():
    """
    Test the local analysis pipeline.
    This script triggers local_analysis.py and verifies that results are generated.
    """
    print("=== Testing Local Analysis Pipeline ===")
    
    # 1. Ensure outputs directory exists
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. Run local_analysis.py
    # We use subprocess to run it as a separate process
    print("Running local_analysis.py (limited to 50 rows)...")
    try:
        result = subprocess.run(
            [sys.executable, "local_analysis.py"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("Execution successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return

    # 3. Verify output file exists
    output_file = os.path.join(output_dir, "local_analysis_results.csv")
    if os.path.exists(output_file):
        print(f"Success: Output file found at {output_file}")
        df = pd.read_csv(output_file)
        print(f"Total results generated: {len(df)}")
        
        # Check for key columns
        required_cols = ['transcript_id', 'topic', 'sentiment_label', 'qa_session_id']
        missing = [c for c in required_cols if c not in df.columns]
        if not missing:
            print("All required columns present.")
        else:
            print(f"Missing columns: {missing}")
            
        # Verify if sessions were grouped
        sessions = df['qa_session_id'].nunique()
        print(f"Number of Q&A sessions detected: {sessions}")
        
    else:
        print(f"Failure: Output file NOT found at {output_file}")

if __name__ == "__main__":
    test_local_pipeline()
