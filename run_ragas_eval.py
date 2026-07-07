"""
RAGAS evaluation script for RAG pipeline logs stored in Supabase.

Computes three reference-free metrics (no ground-truth answers required):
  - faithfulness       : is the generated answer grounded in the retrieved chunks?
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : are the relevant chunks ranked near the top of what was retrieved?

Reference-based metrics (context_recall, answer_correctness, answer_similarity)
are NOT included here since they require a human-written ground-truth answer,
which this dataset doesn't have.

-------------------------------------------------------------------------------
SETUP
-------------------------------------------------------------------------------
pip install supabase ragas datasets langchain-openai pandas --break-system-packages

Environment variables required:
  SUPABASE_URL       - your Supabase project URL
  SUPABASE_KEY       - service role key (needed to read the table server-side)
  OPENROUTER_API_KEY - used for the judge model (nvidia/nemotron-3-super-120b-a12b:free)
                       and embeddings model (nvidia/llama-nemotron-embed-vl-1b-v2:free),
                       both free-tier, routed through OpenRouter's OpenAI-compatible
                       API. Deliberately a different/stronger model family than your
                       flash-lite production generator, to avoid self-evaluation bias
                       (a model tends to rate its own reasoning/phrasing favorably).

-------------------------------------------------------------------------------
BEFORE RUNNING: edit the CONFIG section below to match your actual table/columns.
-------------------------------------------------------------------------------
"""

import os
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from supabase import create_client, Client
from datasets import Dataset

from ragas import evaluate
try:
    # Newer, non-deprecated import path
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
except ImportError:
    # Fallback for older ragas versions that don't have ragas.metrics.collections yet
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


# ==============================================================================
# FREE-TIER RATE LIMIT NOTE (15 requests/min)
# ==============================================================================
# Each row costs roughly: faithfulness (~2 calls) + answer_relevancy (~1 call
# + embeddings) + context_precision (~1 call PER RETRIEVED CHUNK — this is the
# one that scales fastest; 5 chunks/row ≈ 5 calls alone). For ~5 chunks/row
# that's roughly 8 calls/row across the 3 reference-free metrics here.
#
# evaluate() runs these concurrently by default (max_workers=16), which blows
# past a 15/min limit immediately. max_workers is the lever that actually
# throttles this (RunConfig.rate_limits does not reliably enforce a hard RPM
# cap in current versions — open upstream issue). Setting max_workers=1 forces
# sequential calls; slower, but keeps you under the limit in practice.
RATE_LIMIT_SAFE_CONFIG = RunConfig(
    max_workers=1,
    timeout=300,
    max_retries=10,
    max_wait=90,
)


# ==============================================================================
# JUDGE MODEL CONFIG
# ==============================================================================
# Using a stronger, different model family as judge (Nemotron via OpenRouter)
# than your production flash-lite generator. Judging a model's output with
# the same model (or same family/tier) tends to inflate scores — a model
# rates its own reasoning/phrasing style favorably. A different, capable
# judge gives more trustworthy faithfulness/precision scores.
#
# Routed through OpenRouter using free-tier Nemotron models to avoid rate
# limits on a single provider's free tier. Free-tier OpenRouter models have
# their own (usually lower) rate/daily caps too — the max_workers=1 throttle
# below still applies regardless of provider.
#
# Needs OPENROUTER_API_KEY set in your environment (from openrouter.ai/keys).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


# ==============================================================================
# CONFIG - matches the "Conversation_History" table
# ==============================================================================
#
# Actual schema:
#   conversation  jsonb  -> {
#                              "session_id": "...",
#                              "query": "...",
#                              "chunks": [...],       <- see CHUNKS FORMAT note
#                              "llm_response": "...",
#                              "latency_TTFT": "1194ms"
#                            }
#   id            bigint (PK)
#   feedback      text
#   answer_key    uuid   (unique)
#   "Latency"     bigint  <- top-level column, separate from the JSON's latency_TTFT
#
# Everything we need (query, chunks, answer) lives INSIDE the `conversation`
# jsonb blob, not as top-level columns, so we pull that whole column and parse
# it in Python rather than using flat COLUMNS like a simple table would need.

TABLE_NAME = "Conversation_History"   # Postgres is case-sensitive here since the
                                       # table was created with quotes in the DDL

JSON_COLUMN = "conversation"          # the jsonb column holding query/chunks/answer

# Keys inside the `conversation` JSON blob
JSON_KEYS = {
    "query": "query",
    "chunks": "chunks",
    "answer": "llm_response",
    "latency": "latency_TTFT",   # e.g. "1194ms" (string) — informational only
}

# How many rows to pull. Set to None to pull everything.
ROW_LIMIT = 200

# Where to save results
OUTPUT_CSV = "ragas_results.csv"
OUTPUT_SUMMARY_JSON = "ragas_summary.json"

# ==============================================================================
# CHUNKS FORMAT
# ==============================================================================
# RAGAS expects `contexts` as a list[str] per row. The sample you shared for
# "chunks" was written as:
#   "chunks": ["chunk1": " ", "chunk2": " ", ...]
# which isn't valid JSON as literally written (that's dict syntax inside a
# list). parse_chunks() below is defensive and handles all of the realistic
# possibilities so you don't have to fix the pipeline's serialization first:
#   (a) a JSON object:              {"chunk1": "text...", "chunk2": "text..."}
#   (b) a JSON array of strings:    ["text...", "text..."]
#   (c) a JSON array of objects:    [{"text": "...", ...}, ...]
# If your real stored data turns out to be something else, adjust this
# function — run print(df[JSON_COLUMN].iloc[0]) first to check the actual
# shape before assuming.


def parse_chunks(raw_value):
    """Normalize a stored `chunks` value (any of the shapes above) into list[str]."""
    if raw_value is None:
        return []

    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except json.JSONDecodeError:
            return [raw_value]

    if isinstance(raw_value, dict):
        # e.g. {"chunk1": "text", "chunk2": "text"}
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


def fetch_logs(client: Client) -> pd.DataFrame:
    """Pull rows from Supabase and return as a DataFrame."""
    query = client.table(TABLE_NAME).select(f'id, "{JSON_COLUMN}", "Latency", feedback')
    if ROW_LIMIT:
        query = query.limit(ROW_LIMIT)

    response = query.execute()
    rows = response.data

    if not rows:
        print(f"No rows found in table '{TABLE_NAME}'. Check TABLE_NAME/credentials.")
        sys.exit(1)

    print(f"Fetched {len(rows)} rows from '{TABLE_NAME}'.")
    return pd.DataFrame(rows)


def build_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Extract question/contexts/answer from the nested `conversation` jsonb column."""
    questions, answers, contexts = [], [], []

    for raw in df[JSON_COLUMN]:
        # Supabase-py usually returns jsonb columns already decoded as dicts,
        # but handle the string case defensively too.
        convo = raw
        if isinstance(convo, str):
            try:
                convo = json.loads(convo)
            except json.JSONDecodeError:
                convo = {}
        if not isinstance(convo, dict):
            convo = {}

        questions.append(str(convo.get(JSON_KEYS["query"], "") or ""))
        answers.append(str(convo.get(JSON_KEYS["answer"], "") or ""))
        contexts.append(parse_chunks(convo.get(JSON_KEYS["chunks"])))

    valid_rows = [
        i for i in range(len(questions))
        if len(contexts[i]) > 0 and questions[i].strip() and answers[i].strip()
    ]
    dropped = len(questions) - len(valid_rows)
    if dropped > 0:
        print(f"Skipping {dropped} row(s) with missing/empty query, chunks, or llm_response.")

    eval_data = {
        "question": [questions[i] for i in valid_rows],
        "answer": [answers[i] for i in valid_rows],
        "contexts": [contexts[i] for i in valid_rows],
    }

    return Dataset.from_dict(eval_data), valid_rows


def main():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        sys.exit(1)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable (used for the judge/embedding models).")
        sys.exit(1)

    client = create_client(supabase_url, supabase_key)

    print("Fetching logs from Supabase...")
    df = fetch_logs(client)

    print("Building RAGAS-format dataset...")
    dataset, valid_rows = build_ragas_dataset(df)

    if len(dataset) == 0:
        print("No valid rows to evaluate after filtering. Check JSON_KEYS config and your actual data shape.")
        sys.exit(1)

    print(f"Running RAGAS evaluation on {len(dataset)} rows using judge model "
          f"'{JUDGE_MODEL}' (this calls the LLM per row per metric — may take a while)...")

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
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RATE_LIMIT_SAFE_CONFIG,
    )

    print("\n=== RAGAS Summary (averaged across all rows) ===")
    summary = {k: float(v) for k, v in result.items()}
    for metric, score in summary.items():
        print(f"  {metric:20s}: {score:.4f}")

    # Per-row results
    per_row_df = result.to_pandas()

    # Reattach the top-level "Latency" column (ms) for context, aligned to filtered rows
    if "Latency" in df.columns:
        per_row_df["latency_ms"] = df.iloc[valid_rows]["Latency"].values

    # Also reattach the informational latency_TTFT string from inside the JSON blob, if present
    ttft_values = []
    for raw in df.iloc[valid_rows][JSON_COLUMN]:
        convo = raw if isinstance(raw, dict) else {}
        ttft_values.append(convo.get(JSON_KEYS["latency"]))
    per_row_df["latency_ttft"] = ttft_values

    per_row_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-row results saved to: {OUTPUT_CSV}")

    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rows_evaluated": len(dataset),
                "rows_skipped": len(df) - len(dataset),
                "metrics": summary,
            },
            f,
            indent=2,
        )
    print(f"Summary saved to: {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
