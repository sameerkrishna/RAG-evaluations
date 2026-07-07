"""
Generate a synthetic ground-truth test set from PDFs using RAGAS's TestsetGenerator.

Unlike run_ragas_eval.py (which evaluates your existing production logs with
reference-free metrics), this script reads your source PDFs directly and uses
an LLM to synthesize:
  - realistic questions a user might ask about the content
  - the "reference" (ground-truth) answer
  - the "reference_contexts" (the source chunks the answer was derived from)

IMPORTANT: PDFs are pre-chunked using chunker_port.py — a Python port of your
app's actual server/utils/chunker.js (same 600 target / 750 max token sizes,
same paragraph/heading/sliding-window logic) — instead of raw PDF pages or
RAGAS's own default chunking. This keeps the generated reference_contexts at
the same granularity as what your live retriever actually returns, so
context_recall comparisons are apples-to-apples rather than comparing
differently-sized chunks.

That gives you a labeled dataset you didn't have before, which lets you run
RAGAS's reference-based metrics (context_recall, answer_correctness) in
addition to the reference-free ones (faithfulness, answer_relevancy,
context_precision) already covered by run_ragas_eval.py.

-------------------------------------------------------------------------------
SETUP
-------------------------------------------------------------------------------
pip install ragas langchain-community langchain-openai pypdf pandas --break-system-packages

Environment variables required:
  OPENROUTER_API_KEY - used for both the generator LLM and embeddings, via
                       OpenRouter's free-tier Nemotron models

-------------------------------------------------------------------------------
BEFORE RUNNING: edit the CONFIG section below.
-------------------------------------------------------------------------------
"""

import os
import sys
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig

from chunker_port import chunk_text, clean_text


# ==============================================================================
# CONFIG
# ==============================================================================

PDF_DIR = "./seed_documents"        # folder containing your PDFs
TESTSET_SIZE = 5                    # keep SMALL on free tier — see note below

# Same "stronger judge than generator" logic as run_ragas_eval.py — using a
# solid model here matters a lot, since a weak generator LLM produces vague
# or low-quality synthetic questions/ground-truths, which then makes every
# downstream evaluation using this test set less trustworthy.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

OUTPUT_CSV = "ragas_synthetic_testset.csv"

# ==============================================================================
# FREE-TIER RATE LIMIT NOTE (15 requests/min)
# ==============================================================================
# Testset generation makes MANY more LLM calls than TESTSET_SIZE implies:
#   Phase 1 (knowledge graph construction, once per chunk): ~2-4 calls/chunk
#            for summary/keyphrase/entity extraction
#   Phase 2 (question synthesis, once per requested question): ~1-3 calls/question
#
# For 40 chunks + testset_size=20, that's roughly 100-200+ calls total.
# RAGAS runs these CONCURRENTLY by default (max_workers=16), which will blow
# past a 15/min limit immediately and trigger repeated RateLimitErrors.
#
# RunConfig.rate_limits does NOT reliably enforce a hard requests-per-minute
# cap in current versions (open upstream issue) — max_workers (concurrency)
# is the lever that actually works. Setting it to 1 forces fully sequential
# calls, which combined with generous retry/backoff settings keeps you under
# 15 RPM in practice, at the cost of the run taking a while.
#
# Recommended for a first run on free tier:
#   - Start with 1-2 small PDFs and TESTSET_SIZE=5 to confirm everything
#     works end-to-end before scaling up.
#   - Keep max_workers=1 (below) rather than raising it, even though it's slow.
RATE_LIMIT_SAFE_CONFIG = RunConfig(
    max_workers=1,       # fully sequential — the actual throttle that matters
    timeout=300,         # allow slow sequential calls to complete
    max_retries=10,       # retry on transient 429s instead of failing the run
    max_wait=90,         # backoff ceiling between retries
)


def load_pdfs(pdf_dir: str):
    """Load all PDFs, then chunk them with the SAME algorithm as production
    (chunker_port.py, mirroring server/utils/chunker.js), instead of raw PDF
    pages or RAGAS's own default chunking.

    Why this matters: context_recall compares your app's retrieved chunks
    against RAGAS's reference_contexts. If those two are chunked differently
    (different sizes/boundaries), the comparison is measuring "different
    granularity" rather than "did retrieval actually find the right content" —
    which makes the metric misleading. Chunking identically here keeps the
    comparison apples-to-apples.
    """
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in '{pdf_dir}'. Check PDF_DIR.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s) in '{pdf_dir}':")
    for p in pdf_paths:
        print(f"  - {os.path.basename(p)}")

    docs = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        loader = PyPDFLoader(path)
        pages = loader.load()

        # Join all pages into one text blob per PDF, same as parsePDF() in
        # buildseed.js/documents.js does (cleanedPages.join('\n')), then run
        # it through the identical chunker.
        full_text = "\n".join(clean_text(p.page_content) for p in pages)
        chunks = chunk_text(full_text)

        for chunk in chunks:
            docs.append(Document(
                page_content=chunk.text,
                metadata={
                    "source_file": filename,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                },
            ))

        print(f"  -> {filename}: {len(pages)} page(s) -> {len(chunks)} chunk(s) "
              f"(production chunking: ~600 target / 750 max tokens)")

    print(f"Total: {len(docs)} chunk(s) across all PDFs, matching production granularity.")
    return docs


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    docs = load_pdfs(PDF_DIR)

    print(f"Initializing generator (LLM: {GENERATOR_MODEL}, embeddings: {EMBEDDING_MODEL})...")
    generator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=GENERATOR_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    ))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    ))

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    print(f"Generating {TESTSET_SIZE} synthetic question/ground-truth pairs "
          f"(this builds a knowledge graph from your docs first, then synthesizes "
          f"questions — running max_workers=1 to respect your 15 RPM free-tier "
          f"limit, so this will be slow. Be patient.)...")

    testset = generator.generate_with_langchain_docs(
        docs, testset_size=TESTSET_SIZE, run_config=RATE_LIMIT_SAFE_CONFIG
    )

    df = testset.to_pandas()
    print(f"\nGenerated {len(df)} rows. Columns: {list(df.columns)}")
    print(df.head())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved synthetic test set to: {OUTPUT_CSV}")

    print(
        "\nNext step: run each generated question through your actual RAG "
        "pipeline (embed -> retrieve -> LLM) to collect your pipeline's real "
        "`answer` and `contexts` for each row, then combine with this file's "
        "`reference` (ground truth) column to run reference-based RAGAS "
        "metrics like context_recall and answer_correctness."
    )


if __name__ == "__main__":
    main()
