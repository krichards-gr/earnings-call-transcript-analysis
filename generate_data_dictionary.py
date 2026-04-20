#!/usr/bin/env python3
"""
generate_data_dictionary.py

Auto-generates a data dictionary for the earnings call transcript analysis output.
Reads the latest output CSV to extract live schema info and topic configuration,
then calls an LLM to generate human-friendly field descriptions.

Usage:
    python generate_data_dictionary.py                    # uses latest output CSV
    python generate_data_dictionary.py --csv path/to.csv  # uses specific CSV
    python generate_data_dictionary.py --no-llm           # skip LLM, use hardcoded descriptions only
"""

import argparse
import csv
import json
import os
import re
import glob
from datetime import datetime

# ---------------------------------------------------------------------------
# Hardcoded field metadata (ground truth from the pipeline code)
# ---------------------------------------------------------------------------

FIELD_METADATA = {
    "transcript_id": {
        "type": "STRING",
        "source": "BigQuery (source)",
        "description": "MD5 hash uniquely identifying the earnings call transcript.",
        "example": "e99503abf58764d91ef1bbdd3adba26d",
    },
    "paragraph_number": {
        "type": "INTEGER",
        "source": "BigQuery (source)",
        "description": "Sequential position of this speech segment within the transcript, starting at 1.",
        "example": "3",
    },
    "speaker": {
        "type": "STRING",
        "source": "BigQuery (source)",
        "description": "Name of the person speaking, or 'Operator' / 'Unknown' for unidentified speakers.",
        "example": "Sanjay Mehrotra",
    },
    "qa_session_id": {
        "type": "INTEGER",
        "source": "Pipeline (session detection)",
        "description": "Incrementing identifier for Q&A sessions within a transcript. Increments when an operator transition or session-boundary keyword is detected.",
        "example": "1",
    },
    "interaction_type": {
        "type": "STRING",
        "source": "Pipeline (ML classifier: eng_type_class_v1)",
        "description": "Classification of the speech segment's function in the call.",
        "values": "Question, Answer, Admin",
    },
    "role": {
        "type": "STRING",
        "source": "Pipeline (ML classifier: role_class_v1)",
        "description": "Classification of the speaker's role on the call.",
        "values": "Analyst, Executive, Operator, Admin",
    },
    "issue_area": {
        "type": "STRING",
        "source": "Pipeline (topic detection)",
        "description": "Broad category for the topic detected in this segment. 'Unknown' if no topic matched.",
        "example": "AI",
    },
    "issue_subtopic": {
        "type": "STRING",
        "source": "Pipeline (topic detection)",
        "description": "Specific topic label matched in the text. NULL if no topic was detected.",
        "example": "Layoffs",
    },
    "key_terms_found": {
        "type": "STRING",
        "source": "Pipeline (topic detection)",
        "description": "Comma-separated list of keyword pattern matches that triggered the topic detection for this segment. NULL if detected via semantic similarity only or if no topic matched.",
        "example": "workforce reduction, headcount",
    },
    "similarity_score": {
        "type": "FLOAT",
        "source": "Pipeline (topic detection, semantic path)",
        "description": "Cosine similarity score (0.0-1.0) between the segment and the matched topic anchor embedding. Populated only when the topic was detected via semantic similarity (threshold >= 0.7). NULL for pattern-matched topics or when no topic was detected.",
        "example": "0.73",
    },
    "matched_anchor": {
        "type": "STRING",
        "source": "Pipeline (topic detection, semantic path)",
        "description": "The anchor term whose embedding was closest to the segment text. Populated only for semantically matched topics. NULL otherwise.",
        "example": "automation replacing jobs",
    },
    "sentiment_label": {
        "type": "STRING",
        "source": "Pipeline (VADER sentiment)",
        "description": "Sentiment polarity of the segment text based on VADER compound score: 'positive' (>= 0.05), 'negative' (<= -0.05), or 'neutral'.",
        "values": "positive, negative, neutral",
    },
    "sentiment_score": {
        "type": "FLOAT",
        "source": "Pipeline (VADER sentiment)",
        "description": "Absolute value of the VADER compound sentiment score, ranging from 0.0 (neutral) to 1.0 (strongly polar).",
        "example": "0.85",
    },
    "all_scores": {
        "type": "STRING",
        "source": "Pipeline (VADER sentiment)",
        "description": "Full VADER polarity breakdown showing positive, neutral, negative, and compound scores.",
        "example": "pos: 0.14, neu: 0.84, neg: 0.02, compound: 1.00",
    },
    "symbol": {
        "type": "STRING",
        "source": "BigQuery (metadata)",
        "description": "Company ticker symbol (NYSE/NASDAQ).",
        "example": "AAPL",
    },
    "report_date": {
        "type": "DATE",
        "source": "BigQuery (metadata)",
        "description": "Date of the earnings call (YYYY-MM-DD).",
        "example": "2026-03-18",
    },
    "fiscal_year": {
        "type": "INTEGER",
        "source": "BigQuery (metadata)",
        "description": "Fiscal year of the earnings report.",
        "example": "2026",
    },
    "fiscal_quarter": {
        "type": "INTEGER",
        "source": "BigQuery (metadata)",
        "description": "Fiscal quarter of the earnings report (1-4).",
        "example": "2",
    },
    "corporation": {
        "type": "STRING",
        "source": "BigQuery (corporation_reference)",
        "description": "Full legal name of the company.",
        "example": "Micron Technology",
    },
    "sector": {
        "type": "STRING",
        "source": "BigQuery (corporation_reference)",
        "description": "Industry sector classification for the company.",
        "example": "Information Technology",
    },
    "content": {
        "type": "TEXT",
        "source": "BigQuery (source)",
        "description": "Full transcript text of the speech segment. Omitted when the pipeline is run with --no-content.",
        "example": "(variable-length transcript text)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_output_csv(outputs_dir="outputs"):
    """Find the most recently created cli_analysis_results CSV."""
    pattern = os.path.join(outputs_dir, "cli_analysis_results_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_csv_schema(csv_path):
    """Read column names and sample values from a CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=50)
    schema = []
    for col in df.columns:
        non_null = df[col].dropna()
        sample = str(non_null.iloc[0]) if len(non_null) > 0 else ""
        nunique = non_null.nunique()
        dtype = str(df[col].dtype)
        schema.append({
            "column": col,
            "pandas_dtype": dtype,
            "non_null_count": len(non_null),
            "unique_count": nunique,
            "sample_value": sample[:120],
        })
    return schema, len(df)


def load_topic_config(topics_file="topics.json"):
    """Load topics.json and summarize configured topics."""
    if not os.path.exists(topics_file):
        return []
    with open(topics_file) as f:
        data = json.load(f)
    summaries = []
    for topic in data.get("topics", []):
        n_patterns = len(topic.get("patterns", []))
        n_anchors = len(topic.get("anchors", []))
        n_exclusions = len(topic.get("exclusions", []))
        summaries.append({
            "label": topic["label"],
            "issue_area": topic.get("issue_area", ""),
            "pattern_count": n_patterns,
            "anchor_count": n_anchors,
            "exclusion_count": n_exclusions,
        })
    return summaries


def derive_dataset_name(csv_path):
    """Extract the dataset identifier from the CSV filename."""
    basename = os.path.basename(csv_path)
    # cli_analysis_results_20260407_150207.csv -> cli_analysis_results_20260407_150207
    name = os.path.splitext(basename)[0]
    return name


def try_llm_descriptions(schema_info, topic_summaries):
    """
    Call Claude API to generate user-friendly field descriptions.
    Returns a dict of {column_name: description} or None if unavailable.
    """
    try:
        import anthropic
    except ImportError:
        print("  [INFO] anthropic package not installed -- using hardcoded descriptions only.")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [INFO] ANTHROPIC_API_KEY not set -- using hardcoded descriptions only.")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    schema_text = "\n".join(
        f"  - {s['column']} (dtype: {s['pandas_dtype']}, sample: {s['sample_value']})"
        for s in schema_info
    )
    topic_text = "\n".join(
        f"  - {t['label']} (area: {t['issue_area']}, {t['pattern_count']} patterns, {t['anchor_count']} anchors)"
        for t in topic_summaries
    ) or "  (none configured)"

    prompt = f"""You are writing a data dictionary for business analysts who consume an earnings call transcript analysis dataset. Below is the schema and topic configuration.

COLUMNS:
{schema_text}

CONFIGURED TOPICS:
{topic_text}

For each column, write a 1-2 sentence plain-English description suitable for a non-technical user. Include the data type, possible values where applicable, and what "NULL" means for that column. Return ONLY a JSON object mapping column name to description string. No markdown fences."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  [WARN] LLM call failed ({e}) -- falling back to hardcoded descriptions.")
        return None


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def generate_markdown(csv_path, schema_info, row_count, topic_summaries, llm_descriptions=None):
    """Build the data dictionary markdown string."""
    dataset_name = derive_dataset_name(csv_path)
    gen_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# Data Dictionary: `{dataset_name}`")
    lines.append("")
    lines.append(f"> Auto-generated on {gen_timestamp} by `generate_data_dictionary.py`")
    lines.append(f"> Source file: `{os.path.basename(csv_path)}`")
    lines.append(f"> Sample rows inspected: {row_count}")
    lines.append("")

    # --- Overview ---
    lines.append("## Overview")
    lines.append("")
    lines.append("This dataset contains enriched earnings call transcript segments produced by the **Earnings Call Transcript Analysis Pipeline**. Each row represents one speech segment from a corporate earnings call, enriched with ML-based classifications (speaker role, interaction type), topic detection, sentiment analysis, and Q&A session tracking.")
    lines.append("")
    lines.append("**Grain:** One row per detected topic per transcript segment. A segment with no topic match produces one row with NULL topic fields. A segment matching multiple topics produces multiple rows (up to 3 from semantic similarity).")
    lines.append("")

    # --- BigQuery destination ---
    lines.append("**BigQuery table:** `sri-benchmarking-databases.pressure_monitoring.earnings_call_transcript_enriched_local`")
    lines.append("")

    # --- Topic configuration ---
    lines.append("## Configured Topics")
    lines.append("")
    if topic_summaries:
        lines.append("| Topic Label | Issue Area | Keyword Patterns | Semantic Anchors | Exclusion Terms |")
        lines.append("|---|---|---|---|---|")
        for t in topic_summaries:
            lines.append(f"| {t['label']} | {t['issue_area']} | {t['pattern_count']} | {t['anchor_count']} | {t['exclusion_count']} |")
    else:
        lines.append("No topics configured in `topics.json`.")
    lines.append("")
    lines.append("Topics are detected via two methods: (1) **exact keyword pattern matching** using spaCy, and (2) **semantic similarity** using sentence-transformer embeddings with a cosine similarity threshold of 0.7.")
    lines.append("")

    # --- Field reference ---
    lines.append("## Field Reference")
    lines.append("")
    lines.append("| # | Field | Type | Source | Description |")
    lines.append("|---|---|---|---|---|")

    for i, s in enumerate(schema_info, 1):
        col = s["column"]
        meta = FIELD_METADATA.get(col, {})
        dtype = meta.get("type", s["pandas_dtype"])
        source = meta.get("source", "")

        # Prefer LLM description if available, fall back to hardcoded
        if llm_descriptions and col in llm_descriptions:
            desc = llm_descriptions[col]
        else:
            desc = meta.get("description", "")

        # Add possible values inline if defined
        values = meta.get("values")
        if values and values not in desc:
            desc += f" Possible values: {values}."

        # Escape pipes in descriptions
        desc = desc.replace("|", "\\|")

        lines.append(f"| {i} | `{col}` | {dtype} | {source} | {desc} |")

    lines.append("")

    # --- Key notes ---
    lines.append("## Key Notes")
    lines.append("")
    lines.append("- **Sentiment model:** VADER (rule-based). `sentiment_score` is the absolute value of the compound score; `all_scores` provides the full polarity breakdown.")
    lines.append("- **Classification models:** Two custom HuggingFace models (`eng_type_class_v1` for interaction type, `role_class_v1` for speaker role) run on CPU.")
    lines.append("- **Post-processing rules:** Segments classified as 'Question' are forced to role = 'Analyst'. Analyst segments containing question-indicator phrases are promoted to interaction_type = 'Question'.")
    lines.append("- **Session boundaries:** `qa_session_id` increments when an Operator transition or session-start keyword is detected.")
    lines.append("- **Fragment rejoining:** Adjacent incomplete sentences in the source data are concatenated before analysis.")
    lines.append("- **Content column:** Present by default; omitted when the pipeline runs with `--no-content`.")
    lines.append("- **Topic configuration** is driven by `topics.json`, which can be regenerated from keyword CSVs via `python cli_analysis.py --parse-raw` or `--keyword-file`.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a data dictionary for the analysis output.")
    parser.add_argument("--csv", type=str, help="Path to a specific output CSV. Defaults to the latest in outputs/.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment; use hardcoded descriptions only.")
    parser.add_argument("--output", type=str, help="Output path for the data dictionary. Defaults to outputs/<dataset_name>_data_dictionary.md.")
    args = parser.parse_args()

    # 1. Locate the CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_output_csv()

    if not csv_path or not os.path.exists(csv_path):
        print("[ERROR] No output CSV found. Run the analysis pipeline first, or pass --csv.")
        return

    print(f"[1/4] Reading schema from: {csv_path}")
    schema_info, row_count = read_csv_schema(csv_path)
    print(f"       Found {len(schema_info)} columns, {row_count} sample rows")

    # 2. Load topic config
    print("[2/4] Loading topic configuration from topics.json")
    topic_summaries = load_topic_config()
    print(f"       Found {len(topic_summaries)} configured topics")

    # 3. Optional LLM enrichment
    llm_descriptions = None
    if not args.no_llm:
        print("[3/4] Requesting LLM-generated field descriptions...")
        llm_descriptions = try_llm_descriptions(schema_info, topic_summaries)
        if llm_descriptions:
            print(f"       Received descriptions for {len(llm_descriptions)} fields")
    else:
        print("[3/4] Skipping LLM enrichment (--no-llm)")

    # 4. Generate markdown
    print("[4/4] Generating data dictionary...")
    markdown = generate_markdown(csv_path, schema_info, row_count, topic_summaries, llm_descriptions)

    # Determine output path tied to the dataset name
    dataset_name = derive_dataset_name(csv_path)
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.dirname(csv_path) or "outputs"
        out_path = os.path.join(out_dir, f"{dataset_name}_data_dictionary.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n[DONE] Data dictionary written to: {out_path}")
    print(f"       Dataset: {dataset_name}")


if __name__ == "__main__":
    main()
