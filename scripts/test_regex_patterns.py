#!/usr/bin/env python3
"""
Test the updated regex patterns against actual transcript content.
"""

import re

# Test cases from actual transcript
test_cases = [
    {
        "text": "Our next question is from Erik Woodring of Morgan Stanley. Please go ahead.",
        "expected_analyst": "Erik Woodring",
        "should_trigger_session": True
    },
    {
        "text": "We'll now go to Michael Ng of Goldman Sachs. Please go ahead.",
        "expected_analyst": "Michael Ng",
        "should_trigger_session": True
    },
    {
        "text": "The next question will be coming from Ben Reitzes of Melius. Please go ahead.",
        "expected_analyst": "Ben Reitzes",
        "should_trigger_session": True
    },
    {
        "text": "We'll go ahead and take our first question from Amit Daryanani of Evercore.",
        "expected_analyst": "Amit Daryanani",
        "should_trigger_session": True
    },
    {
        "text": "Certainly.",
        "expected_analyst": None,
        "should_trigger_session": False
    },
    {
        "text": "Hi, Ben.",
        "expected_analyst": None,
        "should_trigger_session": False
    }
]

# Updated regex patterns
intro_regex = re.compile(
    r"(?:from|to)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:\s+[A-Z][a-z]+)*)\s+(?:of|with|at)",
    re.IGNORECASE
)

SESSION_START_PATTERNS = [
    r"next question",
    r"first question",
    r"question (?:is |will be )?(?:coming )?from",
    r"(?:we'll |we will )?(?:now )?(?:go to|take)",
    r"move to the line of",
    r"go to the line of",
    r"from the line of"
]
session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)

print("="*100)
print("REGEX PATTERN TESTING")
print("="*100)

all_passed = True

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Text: {test['text']}")

    # Test analyst extraction
    match = intro_regex.search(test['text'])
    extracted_analyst = match.group(1) if match else None

    analyst_passed = extracted_analyst == test['expected_analyst']
    print(f"Expected analyst: {test['expected_analyst']}")
    print(f"Extracted analyst: {extracted_analyst}")
    print(f"Analyst extraction: {'PASS' if analyst_passed else 'FAIL'}")

    # Test session trigger
    has_session_pattern = bool(session_start_regex.search(test['text']))
    session_passed = has_session_pattern == test['should_trigger_session']

    print(f"Expected session trigger: {test['should_trigger_session']}")
    print(f"Detected session trigger: {has_session_pattern}")
    print(f"Session detection: {'PASS' if session_passed else 'FAIL'}")

    if not (analyst_passed and session_passed):
        all_passed = False
        print("*** TEST FAILED ***")

print("\n" + "="*100)
if all_passed:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED - Review patterns above")
print("="*100)
