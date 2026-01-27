import pandas as pd

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 100)

try:
    df = pd.read_excel('q_a.xlsx')
    
    # Sort by paragraph number to ensure order
    df = df.sort_values('paragraph_number')
    
    print("Conversation Flow (checking for missed transitions):")
    
    # Iterate and print relevant columns
    for i, row in df.iterrows():
        content = str(row['content'])
        short_content = (content[:75] + '..') if len(content) > 75 else content
        
        # Highlight potential operator lines
        is_operator_like = "question" in content.lower() or "operator" in str(row['role']).lower()
        
        marker = ">>" if is_operator_like else "  "
        print(f"{marker} ID: {row['qa_session_id']} | Role: {row['role']:<10} | {short_content}")

except Exception as e:
    print(f"Error: {e}")
