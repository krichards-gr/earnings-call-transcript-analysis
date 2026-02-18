#!/usr/bin/env python3
"""
Diagnostic script to examine transcript content and classification results.
Helps identify issues with QA session detection and interaction classification.
"""

import pandas as pd
from google.cloud import bigquery
import re

BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_SOURCE_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_content"
BQ_METADATA_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_metadata"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched_local"

def examine_raw_transcript(transcript_id=None, limit=50):
    """Look at raw transcript content to understand the format."""
    client = bigquery.Client(project=BQ_PROJECT_ID)

    if transcript_id:
        query = f"""
            SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content, m.symbol
            FROM `{BQ_SOURCE_TABLE}` t
            JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
            WHERE t.transcript_id = '{transcript_id}'
            ORDER BY t.paragraph_number
            LIMIT {limit}
        """
    else:
        query = f"""
            SELECT t.transcript_id, t.paragraph_number, t.speaker, t.content, m.symbol
            FROM `{BQ_SOURCE_TABLE}` t
            JOIN `{BQ_METADATA_TABLE}` m ON t.transcript_id = m.transcript_id
            WHERE m.symbol = 'AAPL'
            ORDER BY m.report_date DESC, t.paragraph_number
            LIMIT {limit}
        """

    print(f"\n{'='*100}")
    print("RAW TRANSCRIPT CONTENT")
    print(f"{'='*100}\n")

    df = client.query(query).result().to_dataframe()

    if df.empty:
        print("No data found!")
        return

    print(f"Transcript ID: {df.iloc[0]['transcript_id']}")
    print(f"Company: {df.iloc[0]['symbol']}\n")

    # Look for operator segments
    operator_segments = []
    question_segments = []

    for idx, row in df.iterrows():
        speaker = str(row['speaker']).lower()
        content = str(row['content'])

        print(f"\n--- Paragraph {row['paragraph_number']} ---")
        print(f"Speaker: {row['speaker']}")
        print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")

        # Check if this looks like operator introducing analyst
        if 'operator' in speaker:
            operator_segments.append({
                'paragraph': row['paragraph_number'],
                'speaker': row['speaker'],
                'content': content
            })

        # Check if content mentions "question"
        if 'question' in content.lower():
            question_segments.append({
                'paragraph': row['paragraph_number'],
                'speaker': row['speaker'],
                'content': content
            })

    # Analyze operator segments
    if operator_segments:
        print(f"\n{'='*100}")
        print(f"FOUND {len(operator_segments)} OPERATOR SEGMENTS")
        print(f"{'='*100}\n")

        for seg in operator_segments[:10]:  # Show first 10
            print(f"\nParagraph {seg['paragraph']}:")
            print(f"Speaker: {seg['speaker']}")
            print(f"Content: {seg['content']}\n")

            # Test regex patterns
            print("Testing regex patterns:")

            # Current intro regex
            intro_regex = re.compile(
                r"(?:line of|comes from|is from|from|at)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from|at|is coming)",
                re.IGNORECASE
            )
            match = intro_regex.search(seg['content'])
            if match:
                print(f"  [OK] intro_regex matched: '{match.group(1)}'")
            else:
                print(f"  [NO] intro_regex did not match")

            # Alternative patterns to try
            alt_patterns = [
                (r"from\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", "from NAME pattern"),
                (r"analyst[,:]?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", "analyst NAME pattern"),
                (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+from", "NAME from pattern"),
                (r"([A-Z][a-z]+\s+[A-Z][a-z]+),\s+(?:your|you)", "NAME, your pattern"),
            ]

            for pattern, desc in alt_patterns:
                match = re.search(pattern, seg['content'])
                if match:
                    print(f"  [OK] {desc} matched: '{match.group(1)}'")

            print("-" * 100)

    # Analyze question segments
    if question_segments:
        print(f"\n{'='*100}")
        print(f"FOUND {len(question_segments)} SEGMENTS WITH 'QUESTION'")
        print(f"{'='*100}\n")

        for seg in question_segments[:5]:  # Show first 5
            print(f"Paragraph {seg['paragraph']}: {seg['speaker']}")
            print(f"  {seg['content'][:150]}...")
            print()

def examine_enriched_results(limit=100):
    """Look at enriched results to see classification outcomes."""
    client = bigquery.Client(project=BQ_PROJECT_ID)

    query = f"""
        SELECT
            transcript_id,
            paragraph_number,
            speaker,
            qa_session_id,
            qa_session_label,
            interaction_type,
            role,
            issue_subtopic,
            sentiment_label,
            LEFT(content, 100) as content_preview
        FROM `{BQ_DEST_TABLE}`
        ORDER BY transcript_id, paragraph_number
        LIMIT {limit}
    """

    print(f"\n{'='*100}")
    print("ENRICHED RESULTS ANALYSIS")
    print(f"{'='*100}\n")

    df = client.query(query).result().to_dataframe()

    if df.empty:
        print("No enriched results found! Run the analysis first.")
        return

    print(f"Total records: {len(df)}\n")

    # Analyze QA session labels
    print("QA Session Labels:")
    print(df['qa_session_label'].value_counts())
    print()

    # Analyze interaction types
    print("Interaction Types:")
    print(df['interaction_type'].value_counts())
    print()

    # Analyze roles
    print("Roles:")
    print(df['role'].value_counts())
    print()

    # Show sample records with different classifications
    print(f"\n{'='*100}")
    print("SAMPLE RECORDS BY INTERACTION TYPE")
    print(f"{'='*100}\n")

    for int_type in df['interaction_type'].unique():
        print(f"\n--- {int_type} ---")
        samples = df[df['interaction_type'] == int_type].head(3)
        for _, row in samples.iterrows():
            print(f"  Speaker: {row['speaker']}")
            print(f"  Role: {row['role']}")
            print(f"  Content: {row['content_preview']}...")
            print()

    # Check for "Unknown Analyst" issue
    unknown_count = (df['qa_session_label'] == 'Unknown Analyst').sum()
    none_count = (df['qa_session_label'] == 'None').sum()

    print(f"\n{'='*100}")
    print("QA SESSION DETECTION SUMMARY")
    print(f"{'='*100}\n")
    print(f"Total segments: {len(df)}")
    print(f"'Unknown Analyst': {unknown_count} ({unknown_count/len(df)*100:.1f}%)")
    print(f"'None': {none_count} ({none_count/len(df)*100:.1f}%)")
    print(f"Named analysts: {len(df) - unknown_count - none_count} ({(len(df)-unknown_count-none_count)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    import sys

    print("\n" + "="*100)
    print("TRANSCRIPT DIAGNOSTIC TOOL")
    print("="*100)

    # Check if transcript ID provided
    transcript_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Step 1: Examine raw transcript
    print("\n[1/2] Examining raw transcript content...")
    examine_raw_transcript(transcript_id=transcript_id, limit=30)

    # Step 2: Examine enriched results
    print("\n[2/2] Examining enriched results...")
    try:
        examine_enriched_results(limit=200)
    except Exception as e:
        print(f"Could not examine enriched results: {e}")
        print("This is normal if you haven't run the analysis yet.")

    print("\n" + "="*100)
    print("DIAGNOSTIC COMPLETE")
    print("="*100)
    print("\nUse this information to:")
    print("1. Identify the actual format of analyst introductions")
    print("2. Verify interaction type classification is working")
    print("3. Update regex patterns if needed")
    print("="*100 + "\n")
