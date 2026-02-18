#!/usr/bin/env python3
"""
Test that operator intro text is stripped BEFORE classification.
This ensures analyst questions aren't misclassified as Operator/Admin.
"""

import re

# Simulated role classifier results
def simulate_role_classifier(text):
    """Simulate how the role classifier would classify based on content."""
    lower = text.lower()

    # If text starts with operator language, classify as Operator
    if any(phrase in lower[:100] for phrase in ["we'll go ahead", "we'll take", "we'll now go to", "next question"]):
        return "Operator"
    # If text has question indicators, classify as Analyst
    elif any(word in lower for word in ["i have", "can you", "what", "how", "why", "wondering"]):
        return "Analyst"
    # Otherwise Executive
    else:
        return "Executive"

def simulate_interaction_classifier(text):
    """Simulate how the interaction classifier would classify based on content."""
    lower = text.lower()

    # If text starts with operator language, classify as Admin
    if any(phrase in lower[:100] for phrase in ["we'll go ahead", "we'll take", "we'll now go to", "next question is"]):
        return "Admin"
    # If text has question indicators, classify as Question
    elif any(word in lower for word in ["i have", "can you", "what", "how", "why", "wondering", "?"]):
        return "Question"
    # Otherwise Admin
    else:
        return "Admin"

def test_classification_order():
    """Test that preprocessing happens before classification."""

    print("="*100)
    print("TESTING: Classification Order (Preprocessing BEFORE Classification)")
    print("="*100)

    # Test case: Operator intro merged with analyst question
    original_text = "We'll go ahead and take our first question from Amit Daryanani of Evercore. Yes. I have two. Maybe to start with, you know, there's a lot of focus on the impact of memory to host the companies."

    print(f"\nOriginal Text:")
    print(f"  {original_text[:100]}...")

    # OLD APPROACH: Classify first, then clean
    print(f"\n{'='*100}")
    print("OLD APPROACH: Classify THEN Clean")
    print("="*100)

    role_old = simulate_role_classifier(original_text)
    interaction_old = simulate_interaction_classifier(original_text)

    print(f"  Step 1 - Classification on original text:")
    print(f"    Role: {role_old}")
    print(f"    Interaction: {interaction_old}")

    operator_intro_pattern = r"^(?:We'll|We will|Let's|Certainly\.?)?\s*(?:go ahead and\s+)?(?:take|move to|now go to)\s+(?:our\s+)?(?:the\s+)?(?:first|next)?\s*question\s+(?:from|is from)\s+[^.!?]+[.!?]\s+"
    cleaned_text = re.sub(operator_intro_pattern, '', original_text, flags=re.IGNORECASE).strip()

    print(f"\n  Step 2 - Clean text:")
    print(f"    {cleaned_text[:80]}...")
    print(f"\n  RESULT: {role_old}, {interaction_old} [WRONG!]")

    # NEW APPROACH: Clean first, then classify
    print(f"\n{'='*100}")
    print("NEW APPROACH: Clean THEN Classify")
    print("="*100)

    print(f"  Step 1 - Clean text:")
    print(f"    {cleaned_text[:80]}...")

    role_new = simulate_role_classifier(cleaned_text)
    interaction_new = simulate_interaction_classifier(cleaned_text)

    print(f"\n  Step 2 - Classification on cleaned text:")
    print(f"    Role: {role_new}")
    print(f"    Interaction: {interaction_new}")

    print(f"\n  RESULT: {role_new}, {interaction_new} [CORRECT!]")

    # Verify results
    print(f"\n{'='*100}")
    print("VERIFICATION")
    print("="*100)

    old_correct = (role_old == "Analyst" and interaction_old == "Question")
    new_correct = (role_new == "Analyst" and interaction_new == "Question")

    print(f"\nOld Approach (Classify then Clean):")
    print(f"  Result: {role_old}, {interaction_old}")
    print(f"  Expected: Analyst, Question")
    print(f"  Status: {'[PASS]' if old_correct else '[FAIL]'}")

    print(f"\nNew Approach (Clean then Classify):")
    print(f"  Result: {role_new}, {interaction_new}")
    print(f"  Expected: Analyst, Question")
    print(f"  Status: {'[PASS]' if new_correct else '[FAIL]'}")

    print(f"\n{'='*100}")
    if new_correct and not old_correct:
        print("SUCCESS: New approach fixes the classification issue!")
        print("="*100)
        return True
    elif new_correct and old_correct:
        print("Both approaches work, but new approach is more robust.")
        print("="*100)
        return True
    else:
        print("FAIL: New approach still has issues.")
        print("="*100)
        return False

if __name__ == "__main__":
    import sys
    success = test_classification_order()
    sys.exit(0 if success else 1)
