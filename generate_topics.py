import pandas as pd
import json
import os

def generate_topics_json(csv_path='updated_issue_config_inputs.csv', json_path='topics.json'):
    print(f"Generating {json_path} from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    topics_dict = {}
    
    for _, row in df.iterrows():
        topic_label = row['topic']
        term = str(row['term']).strip()
        term_type = row['type']
        
        if topic_label not in topics_dict:
            topics_dict[topic_label] = {
                "label": topic_label,
                "patterns": [],
                "anchors": []
            }
            
        if term_type == 'anchor term':
            topics_dict[topic_label]["anchors"].append(term)
        elif term_type == 'pattern':
            tokens = term.split()
            pattern = [{"LOWER": t.lower()} for t in tokens]
            topics_dict[topic_label]["patterns"].append(pattern)
        elif term_type == 'exclusionary term':
            if "exclusions" not in topics_dict[topic_label]:
                topics_dict[topic_label]["exclusions"] = []
            topics_dict[topic_label]["exclusions"].append(term.lower())
            
    output_data = {
        "topics": list(topics_dict.values())
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Successfully generated {json_path} with {len(output_data['topics'])} topics.")

if __name__ == "__main__":
    generate_topics_json()
