import json

# Sample of the logic implemented in local_analysis.py
EXCLUSIONS_MAP = {
    "Climate Change": ["financial results", "plastic surgery"]
}

def check_exclusion(topic, text):
    exclusions = EXCLUSIONS_MAP.get(topic, [])
    for ext in exclusions:
        if ext.lower() in text.lower():
            return True, ext
    return False, None

test_cases = [
    ("Climate Change", "We are discussing carbon credits and net zero targets."),
    ("Climate Change", "Our financial results show a decrease in emissions."),
    ("Climate Change", "The company is investing in plastic recycling."),
    ("Climate Change", "The doctor performed plastic surgery on the patient.")
]

print("=== Testing Exclusion Logic ===")
for topic, text in test_cases:
    is_excluded, term = check_exclusion(topic, text)
    status = f"EXCLUDED (matched: '{term}')" if is_excluded else "ALLOWED"
    print(f"Topic: {topic:15} | Status: {status:25} | Text: {text}")
