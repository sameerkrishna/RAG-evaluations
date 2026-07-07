"""
Match synthetic test-set questions (from generate_testset.py) against your
app's logged Supabase answers, then run the FULL RAGAS metric suite —
including the reference-based metrics that need ground truth
(context_recall, answer_correctness) which run_ragas_eval.py couldn't do.

-------------------------------------------------------------------------------
WORKFLOW (manual step in between, by design)
-------------------------------------------------------------------------------
1. Run generate_testset.py -> produces ragas_synthetic_testset.csv
   (columns include: user_input, reference, reference_contexts)

2. YOU manually type each `user_input` question from that CSV into your app.
   Your app logs each conversation (query, chunks, answer, latency) into
   Supabase's Conversation_History table automatically, as it already does.

3. Run THIS script. It:
     a. Loads ragas_synthetic_testset.csv
     b. Pulls all rows from Conversation_History in Supabase
     c. Matches each synthetic question to its logged row by EXACT query text
     d. Builds a combined dataset: question, answer, contexts, ground_truth
     e. Runs RAGAS metrics (reference-free AND reference-based) and saves results

-------------------------------------------------------------------------------
SETUP
-------------------------------------------------------------------------------
pip install supabase ragas datasets langchain-openai pandas --break-system-packages

Environment variables required:
  SUPABASE_URL, SUPABASE_KEY, OPENROUTER_API_KEY

-------------------------------------------------------------------------------
BEFORE RUNNING: edit CONFIG below, and make sure you've completed steps 1-2 above.
-------------------------------------------------------------------------------
"""

import os
import sys
import json

import pandas as pd
from supabase import create_client, Client
from datasets import Dataset

from ragas import evaluate
try:
    from ragas.metrics.collections import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness,
    )
except ImportError:
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision,
        context_recall, answer_correctness,
    )

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


# ==============================================================================
# FREE-TIER RATE LIMIT NOTE (15 requests/min)
# ==============================================================================
# This script runs 5 metrics per row (3 reference-free + 2 reference-based).
# Rough cost per row: faithfulness (~2) + answer_relevancy (~1 + embeddings)
# + context_precision (~1 PER RETRIEVED CHUNK, the biggest driver — 5
# chunks/row ≈ 5 calls) + context_recall (~1) + answer_correctness (~2)
# ≈ 11 calls/row. For 20 rows that's ~220 calls before any retries.
#
# evaluate() defaults to max_workers=16 concurrent calls, which blows past a
# 15/min limit immediately. max_workers=1 forces sequential calls — slower,
# but the actual throttle that works (RunConfig.rate_limits doesn't reliably
# enforce a hard RPM cap in current versions).
RATE_LIMIT_SAFE_CONFIG = RunConfig(
    max_workers=1,
    timeout=300,
    max_retries=10,
    max_wait=90,
)


# ==============================================================================
# CONFIG
# ==============================================================================

SYNTHETIC_TESTSET_CSV = "ragas_synthetic_testset.csv"  # output of generate_testset.py

# Supabase table (same schema as run_ragas_eval.py)
TABLE_NAME = "Conversation_History"
JSON_COLUMN = "conversation"
JSON_KEYS = {
    "query": "query",
    "chunks": "chunks",
    "answer": "llm_response",
    "latency": "latency_TTFT",
}
ROW_LIMIT = None  # pull everything, since we need to match against all logged rows

# Judge model — same reasoning as before: stronger than your flash-lite generator
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

OUTPUT_CSV = "ragas_ground_truth_results.csv"
OUTPUT_SUMMARY_JSON = "ragas_ground_truth_summary.json"
UNMATCHED_CSV = "unmatched_questions.csv"  # synthetic questions with no logged match


def normalize_query(text: str) -> str:
    """Normalize whitespace/case for exact-text matching, to tolerate trivial
    differences (extra spaces, trailing punctuation typed by hand) without
    doing fuzzy matching that could mismatch unrelated questions."""
    return " ".join(str(text).strip().lower().split())


def parse_chunks(raw_value):
    """Same defensive chunk parser as run_ragas_eval.py — handles dict, list of
    strings, or list of {"text": ...} objects."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except json.JSONDecodeError:
            return [raw_value]
    if isinstance(raw_value, dict):
        return [str(v) for v in raw_value.values() if v]
    if isinstance(raw_value, list):
        parsed = []
        for item in raw_value:
            if isinstance(item, str):
                parsed.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("chunk") or item.get("content")
                if text:
                    parsed.append(text)
        return parsed
    return []


def fetch_supabase_logs(client: Client) -> pd.DataFrame:
    query = client.table(TABLE_NAME).select(f'id, "{JSON_COLUMN}", "Latency", feedback')
    if ROW_LIMIT:
        query = query.limit(ROW_LIMIT)
    response = query.execute()
    rows = response.data
    if not rows:
        print(f"No rows found in '{TABLE_NAME}'. Check credentials/table name.")
        sys.exit(1)
    print(f"Fetched {len(rows)} logged conversation(s) from Supabase.")
    return pd.DataFrame(rows)


def build_supabase_lookup(df: pd.DataFrame) -> dict:
    """Build a dict: normalized_query -> {answer, contexts, latency_ms, latency_ttft}
    for exact-text matching against the synthetic test set."""
    lookup = {}
    for _, row in df.iterrows():
        convo = row[JSON_COLUMN]
        if isinstance(convo, str):
            try:
                convo = json.loads(convo)
            except json.JSONDecodeError:
                convo = {}
        if not isinstance(convo, dict):
            convo = {}

        query_text = convo.get(JSON_KEYS["query"], "")
        if not query_text:
            continue

        key = normalize_query(query_text)
        # If the same question was asked more than once, keep the most recent
        # (later row in the fetched order) — adjust here if you'd rather keep
        # the first occurrence instead.
        lookup[key] = {
            "answer": str(convo.get(JSON_KEYS["answer"], "") or ""),
            "contexts": parse_chunks(convo.get(JSON_KEYS["chunks"])),
            "latency_ms": row.get("Latency"),
            "latency_ttft": convo.get(JSON_KEYS["latency"]),
        }
    return lookup


def build_combined_dataset(testset_df: pd.DataFrame, lookup: dict):
    """Match each synthetic question to its logged Supabase answer by exact
    (normalized) query text, and assemble the final RAGAS-ready dataset."""

    # generate_testset.py's output column names can vary slightly by ragas
    # version — handle the common ones defensively.
    question_col = next(
        (c for c in ["user_input", "question"] if c in testset_df.columns), None
    )
    reference_col = next(
        (c for c in ["reference", "ground_truth"] if c in testset_df.columns), None
    )
    if question_col is None or reference_col is None:
        print(
            f"ERROR: Couldn't find question/reference columns in "
            f"{SYNTHETIC_TESTSET_CSV}. Found columns: {list(testset_df.columns)}"
        )
        sys.exit(1)

    questions, answers, contexts, ground_truths, latencies_ms, latencies_ttft = (
        [], [], [], [], [], []
    )
    unmatched = []

    for _, row in testset_df.iterrows():
        question_text = row[question_col]
        key = normalize_query(question_text)
        match = lookup.get(key)

        if match is None:
            unmatched.append(question_text)
            continue

        questions.append(question_text)
        answers.append(match["answer"])
        contexts.append(match["contexts"])
        ground_truths.append(row[reference_col])
        latencies_ms.append(match["latency_ms"])
        latencies_ttft.append(match["latency_ttft"])

    print(f"Matched {len(questions)} of {len(testset_df)} synthetic questions "
          f"to logged Supabase answers.")

    if unmatched:
        print(f"{len(unmatched)} question(s) had no exact match in Supabase — "
              f"saving to {UNMATCHED_CSV}. Likely not yet asked, or typed "
              f"with different wording/punctuation than the synthetic version.")
        pd.DataFrame({question_col: unmatched}).to_csv(UNMATCHED_CSV, index=False)

    if not questions:
        print("No matches found at all. Check that you've actually run the "
              "synthetic questions through your app, and that TABLE_NAME/"
              "JSON_KEYS in CONFIG match your schema.")
        sys.exit(1)

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    return Dataset.from_dict(eval_data), latencies_ms, latencies_ttft


def main():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    if not os.path.exists(SYNTHETIC_TESTSET_CSV):
        print(f"ERROR: {SYNTHETIC_TESTSET_CSV} not found. Run generate_testset.py first.")
        sys.exit(1)

    print(f"Loading synthetic test set from {SYNTHETIC_TESTSET_CSV}...")
    testset_df = pd.read_csv(SYNTHETIC_TESTSET_CSV)
    print(f"Loaded {len(testset_df)} synthetic question(s).")

    client = create_client(supabase_url, supabase_key)
    print("Fetching logged conversations from Supabase...")
    supabase_df = fetch_supabase_logs(client)
    lookup = build_supabase_lookup(supabase_df)

    print("Matching synthetic questions to logged answers...")
    dataset, latencies_ms, latencies_ttft = build_combined_dataset(testset_df, lookup)

    print(f"Running full RAGAS evaluation on {len(dataset)} matched rows "
          f"(reference-free + reference-based metrics, judge: {JUDGE_MODEL})...")

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    ))

    result = evaluate(
        dataset,
        metrics=[
            faithfulness, answer_relevancy, context_precision,   # reference-free
            context_recall, answer_correctness,                  # reference-based
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RATE_LIMIT_SAFE_CONFIG,
    )

    print("\n=== RAGAS Summary (averaged across matched rows) ===")
    summary = {k: float(v) for k, v in result.items()}
    for metric, score in summary.items():
        print(f"  {metric:20s}: {score:.4f}")

    per_row_df = result.to_pandas()
    per_row_df["latency_ms"] = latencies_ms
    per_row_df["latency_ttft"] = latencies_ttft
    per_row_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-row results saved to: {OUTPUT_CSV}")

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(
            {
                "rows_matched": len(dataset),
                "rows_in_synthetic_testset": len(testset_df),
                "metrics": summary,
            },
            f,
            indent=2,
        )
    print(f"Summary saved to: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
