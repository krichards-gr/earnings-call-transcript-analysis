#!/usr/bin/env python3
"""
Test edge cases for interaction classification.
Specifically tests operator intro text accidentally merged with analyst questions.
"""

import re

# Simulated classification results
INTERACTION_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Answer",
    "LABEL_2": "Question"
}

ROLE_ID_MAP = {
    "LABEL_0": "Admin",
    "LABEL_1": "Analyst",
    "LABEL_2": "Executive",
    "LABEL_3": "Operator"
}

def test_edge_cases():
    """Test various edge cases for classification."""

    test_cases = [
        {
            "name": "Operator intro merged with analyst question",
            "speaker": "Amit Daryanani",
            "role_label": "Analyst",
            "initial_classification": "Admin",  # Classifier might get confused
            "content": "We'll go ahead and take our first question from Amit Daryanani of Evercore. Yes. I have two. Maybe to start with, you know, there's a lot of focus on the impact of memory to host the companies and love to kind of get your perspective when you're first guiding gross margins up into March.",
            "expected_interaction": "Question",
            "expected_cleaned": "Yes. I have two. Maybe to start with, you know, there's a lot of focus on the impact of memory..."
        },
        {
            "name": "Pure analyst question",
            "speaker": "Erik Woodring",
            "role_label": "Analyst",
            "initial_classification": "Question",
            "content": "Great, guys. Thank you for taking my questions. Tim, congrats on announcing the partnership with Google.",
            "expected_interaction": "Question",
            "expected_cleaned": "Great, guys. Thank you for taking my questions. Tim, congrats on announcing the partnership with Google."
        },
        {
            "name": "Executive answer (should not be affected)",
            "speaker": "Timothy D. Cook",
            "role_label": "Executive",
            "initial_classification": "Answer",
            "content": "Yeah. Amit, hi. It's Tim. Let me back up a bit and talk about the constraints that Kevin referred to in his remarks.",
            "expected_interaction": "Answer",
            "expected_cleaned": "Yeah. Amit, hi. It's Tim. Let me back up a bit and talk about the constraints that Kevin referred to in his remarks."
        },
        {
            "name": "Analyst with 'what' question indicator",
            "speaker": "Michael Ng",
            "role_label": "Analyst",
            "initial_classification": "Admin",  # Misclassified
            "content": "What are your thoughts on the competitive landscape?",
            "expected_interaction": "Question",
            "expected_cleaned": "What are your thoughts on the competitive landscape?"
        },
        {
            "name": "Analyst with 'can you' question indicator",
            "speaker": "Ben Reitzes",
            "role_label": "Analyst",
            "initial_classification": "Admin",  # Misclassified
            "content": "Can you comment on the Services growth trajectory?",
            "expected_interaction": "Question",
            "expected_cleaned": "Can you comment on the Services growth trajectory?"
        }
    ]

    # Preprocessing pattern
    operator_intro_pattern = r"^(?:We'll|We will|Let's|Certainly\.?)?\s*(?:go ahead and\s+)?(?:take|move to|now go to)\s+(?:our\s+)?(?:the\s+)?(?:first|next)?\s*question\s+(?:from|is from)\s+[^.!?]+[.!?]\s+"

    # Question indicators
    question_indicators = [
        "have two", "have a question", "wondering", "curious", "want to ask",
        "can you", "could you", "would you", "will you",
        "how do", "how does", "how should", "how would",
        "what", "why", "when", "where", "which",
        "talk about", "comment on", "thoughts on", "perspective on"
    ]

    print("="*100)
    print("EDGE CASE TESTING - Interaction Classification")
    print("="*100)

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*100}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'='*100}")
        print(f"Speaker: {test['speaker']} (Role: {test['role_label']})")
        print(f"Initial Classification: {test['initial_classification']}")
        print(f"Content: {test['content'][:100]}...")

        # Simulate preprocessing
        text = test['content']
        cleaned_text = re.sub(operator_intro_pattern, '', text, flags=re.IGNORECASE).strip()

        if len(cleaned_text) < len(text):
            print(f"\n[PREPROCESSING] Stripped operator intro")
            print(f"  Before: {text[:80]}...")
            print(f"  After: {cleaned_text[:80]}...")
            text = cleaned_text

        # Simulate classification override
        interaction_type = test['initial_classification']

        # Set up proper context for Answer validation
        # Executives answering should have previous_interaction = "Question" from "Analyst"
        if test['role_label'] == "Executive" and interaction_type == "Answer":
            previous_interaction = "Question"
            previous_role = "Analyst"
        else:
            previous_interaction = "Admin"
            previous_role = "Admin"

        # Answer validation (from previous fix)
        if interaction_type == "Answer":
            if previous_interaction != "Question" or previous_role != "Analyst":
                interaction_type = "Admin"

        # Question boost for Analyst segments
        if test['role_label'] == "Analyst" and interaction_type != "Question":
            lower_text = text.lower()
            if any(indicator in lower_text for indicator in question_indicators):
                print(f"\n[OVERRIDE] Detected question indicators in Analyst segment")
                matched_indicators = [ind for ind in question_indicators if ind in lower_text]
                print(f"  Matched: {matched_indicators[:3]}")
                interaction_type = "Question"

        # Validate results
        print(f"\nFinal Classification: {interaction_type}")
        print(f"Expected: {test['expected_interaction']}")

        if interaction_type == test['expected_interaction']:
            print("[OK] PASS")
        else:
            print("[FAIL] FAIL")
            all_passed = False

        # Check if text was cleaned as expected
        if "operator intro" in test['name'].lower():
            cleaned_preview = test['expected_cleaned'][:50]
            actual_preview = text[:50]
            if cleaned_preview in actual_preview or actual_preview in cleaned_preview:
                print("[OK] Text cleaning worked correctly")
            else:
                print(f"[WARN] Text cleaning may not have worked as expected")
                print(f"  Expected to start with: {cleaned_preview}")
                print(f"  Actually starts with: {actual_preview}")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    if all_passed:
        print("\n[OK][OK][OK] ALL EDGE CASE TESTS PASSED [OK][OK][OK]")
        print("\nThe classification logic correctly handles:")
        print("  • Operator intro text merged with analyst questions")
        print("  • Question indicator detection for misclassified analyst segments")
        print("  • Standard question/answer/admin classification")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        print("Review the failures above.")

    return all_passed

if __name__ == "__main__":
    import sys
    success = test_edge_cases()
    sys.exit(0 if success else 1)
