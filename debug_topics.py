import json
import os

def check_topics(filepath):
    print(f"Checking {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    topics = data.get('topics', [])
    print(f"Found {len(topics)} topics.")
    
    errors = 0
    for i, topic in enumerate(topics):
        label = topic.get('label')
        if not isinstance(label, str):
            print(f"Error at index {i}: label is not a string! Value: {label} (Type: {type(label)})")
            errors += 1
            
        anchors = topic.get('anchors', [])
        if not isinstance(anchors, list):
            print(f"Error at index {i}: anchors is not a list!")
            errors += 1
            continue
            
        for j, anchor in enumerate(anchors):
            if not isinstance(anchor, str):
                print(f"Error at index {i}, anchor {j}: anchor is not a string! Value: {anchor} (Type: {type(anchor)})")
                errors += 1
                
    if errors == 0:
        print("SUCCESS: All labels and anchors are strings.")
    else:
        print(f"FAILURE: Found {errors} errors.")

if __name__ == "__main__":
    check_topics("topics.json")
