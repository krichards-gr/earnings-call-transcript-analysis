import pandas as pd
import re

SESSION_START_PATTERNS = [
    r"next question (?:comes|is coming)",
    r"next we (?:have|will go to)",
    r"question (?:comes|is coming) from",
    r"your first question (?:comes|is coming)",
    r"move to the line of",
    r"go to the line of",
    r"from the line of",
    r"comes? from the line of"
]
session_start_regex = re.compile("|".join(SESSION_START_PATTERNS), re.IGNORECASE)

# Refined analyst extraction regex
# Handles "lines of", "from", "at", "with", and "is coming from"
intro_regex = re.compile(
    r"(?:line of|comes from|is from|from|at)\s+(?:the line of\s+)?([^,.]+?)\s+(?:with|from|at|is coming)", 
    re.IGNORECASE
)

def test_logic():
    try:
        df = pd.read_excel('q_a.xlsx')
        df = df.sort_values('paragraph_number')
        
        current_session_id = 0
        current_analyst = "None"
        
        results = []
        
        for i, row in df.iterrows():
            text = str(row['content'])
            role_label = str(row['role'])
            
            lower_text = text.lower()
            is_operator = role_label == "Operator"
            has_session_start_keyword = session_start_regex.search(lower_text)
            is_transition_text = any(k in lower_text for k in ["question", "line of", "analyst"])

            if (is_operator and is_transition_text) or has_session_start_keyword:
                current_session_id += 1
                match = intro_regex.search(text)
                if match:
                    current_analyst = match.group(1).strip()
                elif is_operator and "question" in lower_text:
                    current_analyst = "Unknown Analyst"
                
                results.append({
                    "para": row['paragraph_number'],
                    "session_id": current_session_id,
                    "analyst": current_analyst,
                    "text": text
                })
        
        print(f"\nDetected {len(results)} Transitions:")
        for res in results:
            print(f"Para {res['para']:>3} | ID: {res['session_id']:>2} | Analyst: {res['analyst']:<20} | Text: {res['text'][:80]}...")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_logic()
