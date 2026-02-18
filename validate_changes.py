#!/usr/bin/env python3
"""
Validation script to verify all changes are working correctly.
Tests:
1. No qa_session_label column in output
2. Answer validation logic (Answers only follow Questions from Analysts)
3. Session detection increments correctly
4. Interaction types are properly distributed
"""

import pandas as pd
import sys
from google.cloud import bigquery

BQ_PROJECT_ID = "sri-benchmarking-databases"
BQ_DATASET = "pressure_monitoring"
BQ_DEST_TABLE = f"{BQ_PROJECT_ID}.{BQ_DATASET}.earnings_call_transcript_enriched_local"

def validate_output():
    """Validate the enriched output for correctness."""

    print("\n" + "="*100)
    print("VALIDATION TEST - Checking Recent Changes")
    print("="*100)

    client = bigquery.Client(project=BQ_PROJECT_ID)

    # Get latest 500 records
    query = f"""
        SELECT
            transcript_id,
            paragraph_number,
            speaker,
            qa_session_id,
            interaction_type,
            role,
            issue_subtopic,
            sentiment_label
        FROM `{BQ_DEST_TABLE}`
        ORDER BY transcript_id DESC, paragraph_number ASC
        LIMIT 500
    """

    try:
        df = client.query(query).result().to_dataframe()
    except Exception as e:
        print(f"\n❌ ERROR: Could not query results table: {e}")
        print("   Make sure you've run the analysis at least once.")
        return False

    if df.empty:
        print("\n❌ ERROR: No data found in enriched table.")
        print("   Run the analysis first: python cli_analysis.py --test")
        return False

    print(f"\n✓ Found {len(df)} records to validate\n")

    all_passed = True

    # Test 1: Check for qa_session_label column (should NOT exist)
    print("="*100)
    print("TEST 1: qa_session_label Column Removed")
    print("="*100)

    if 'qa_session_label' in df.columns:
        print("❌ FAIL: qa_session_label column still exists!")
        print("   This column should have been removed.")
        all_passed = False
    else:
        print("✓ PASS: qa_session_label column successfully removed")

    # Test 2: Validate Answer logic
    print("\n" + "="*100)
    print("TEST 2: Answer Validation (Answers should follow Questions from Analysts)")
    print("="*100)

    invalid_answers = 0
    valid_answers = 0

    for transcript_id in df['transcript_id'].unique():
        transcript_df = df[df['transcript_id'] == transcript_id].sort_values('paragraph_number')

        prev_interaction = None
        prev_role = None

        for idx, row in transcript_df.iterrows():
            if row['interaction_type'] == 'Answer':
                if prev_interaction == 'Question' and prev_role == 'Analyst':
                    valid_answers += 1
                else:
                    invalid_answers += 1
                    if invalid_answers <= 3:  # Show first 3 examples
                        print(f"\n   Example of invalid Answer:")
                        print(f"   Transcript: {row['transcript_id']}")
                        print(f"   Paragraph: {row['paragraph_number']}")
                        print(f"   Speaker: {row['speaker']} (Role: {row['role']})")
                        print(f"   Previous was: {prev_interaction} from {prev_role}")

            prev_interaction = row['interaction_type']
            prev_role = row['role']

    total_answers = valid_answers + invalid_answers
    if total_answers > 0:
        accuracy = (valid_answers / total_answers) * 100
        print(f"\n   Valid Answers: {valid_answers}/{total_answers} ({accuracy:.1f}%)")

        if accuracy >= 95:  # Allow small margin for edge cases
            print("✓ PASS: Answer validation working correctly")
        else:
            print(f"❌ FAIL: Too many invalid answers ({100-accuracy:.1f}% error rate)")
            all_passed = False
    else:
        print("   No Answer segments found - unable to validate")

    # Test 3: Session ID increments
    print("\n" + "="*100)
    print("TEST 3: QA Session ID Incrementation")
    print("="*100)

    session_counts = df['qa_session_id'].value_counts().sort_index()
    unique_sessions = len(session_counts)

    print(f"   Found {unique_sessions} unique session IDs")
    print(f"   Session ID range: {df['qa_session_id'].min()} to {df['qa_session_id'].max()}")

    if unique_sessions > 0:
        print("✓ PASS: Session IDs are being assigned")
    else:
        print("❌ FAIL: No session IDs found")
        all_passed = False

    # Test 4: Interaction type distribution
    print("\n" + "="*100)
    print("TEST 4: Interaction Type Distribution")
    print("="*100)

    interaction_counts = df['interaction_type'].value_counts()
    print("\n   Distribution:")
    for itype, count in interaction_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {itype}: {count} ({pct:.1f}%)")

    # Check we have all three types
    expected_types = {'Admin', 'Question', 'Answer'}
    found_types = set(interaction_counts.index)

    if expected_types == found_types:
        print("\n✓ PASS: All three interaction types present")
    else:
        missing = expected_types - found_types
        print(f"\n❌ FAIL: Missing interaction types: {missing}")
        all_passed = False

    # Test 5: Role distribution
    print("\n" + "="*100)
    print("TEST 5: Role Distribution")
    print("="*100)

    role_counts = df['role'].value_counts()
    print("\n   Distribution:")
    for role, count in role_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {role}: {count} ({pct:.1f}%)")

    if len(role_counts) >= 3:  # Expect at least Analyst, Executive, Operator
        print("\n✓ PASS: Multiple roles detected")
    else:
        print(f"\n⚠ WARNING: Only {len(role_counts)} unique roles found")

    # Final summary
    print("\n" + "="*100)
    print("VALIDATION SUMMARY")
    print("="*100)

    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYour changes are working correctly!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease review the failures above and verify the code changes.")
        return False

if __name__ == "__main__":
    try:
        success = validate_output()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
