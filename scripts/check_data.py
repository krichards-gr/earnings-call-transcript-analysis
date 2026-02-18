from google.cloud import bigquery

client = bigquery.Client(project='sri-benchmarking-databases')

# Check what transcripts exist in the date range
query = """
SELECT m.symbol, m.report_date, m.transcript_id
FROM `sri-benchmarking-databases.pressure_monitoring.earnings_call_transcript_metadata` m
WHERE m.report_date >= '2025-12-31' AND m.report_date <= '2026-02-17'
ORDER BY m.report_date DESC
"""

df = client.query(query).to_dataframe()

print(f"Total transcripts in date range (2025-12-31 to 2026-02-17): {len(df)}")
print(f"\nBy company:")
print(df['symbol'].value_counts())
print(f"\nDate distribution:")
print(df.groupby('report_date').size().sort_index())

# Show all transcripts
print(f"\nAll transcripts:")
print(df[['symbol', 'report_date', 'transcript_id']])
