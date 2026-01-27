import pandas as pd

try:
    df = pd.read_excel('q_a.xlsx')
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    if 'qa_grouping' in df.columns:
        print("\nUnique values in qa_grouping:")
        print(df['qa_grouping'].unique())
        print("\nNull count in qa_grouping:", df['qa_grouping'].isnull().sum())
    else:
        print("\n'qa_grouping' column not found.")
        # Look for similar columns
        print("Columns containing 'group':", [c for c in df.columns if 'group' in c.lower()])

except Exception as e:
    print(f"Error reading excel file: {e}")
