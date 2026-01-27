import pandas as pd

try:
    df = pd.read_excel('q_a.xlsx')
    print("Columns List:")
    for col in df.columns:
        print(f"- {col}")
        
    cols_to_check = ['qa_session_id', 'qa_session_label', 'interaction_type', 'role']
    for col in cols_to_check:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts().head())
        else:
            print(f"\nColumn '{col}' NOT found.")

except Exception as e:
    print(f"Error: {e}")
