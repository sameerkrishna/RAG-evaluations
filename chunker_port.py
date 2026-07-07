"""
Python port of server/utils/chunker.js — used ONLY so that
generate_testset.py can chunk PDFs identically to how the production app
chunks them, ensuring RAGAS's reference_contexts are directly comparable to
what your live retriever actually returns (same granularity, same boundaries).

This mirrors chunker.js's logic exactly:
  - same constants (600 target / 750 max / 100 overlap tokens, 4 chars/token)
  - same heading detection regex
  - same paragraph-accumulation + soft-target-flush behavior
  - same oversized-paragraph sliding-window fallback with sentence/word
    boundary snapping (the fix from the original chunker.js discussion)

If you ever change chunker.js's constants or logic, mirror the change here
too, or the two will silently drift apart again.
"""

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 750
OVERLAP_TOKENS = 100
MIN_CHUNK_CHARS = 100

# Same heading pattern as chunker.js's HEADING_RE
HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s]{2,60}$|#{1,4}\s.+|(?:\d+\.)+\s.+)", re.MULTILINE
)

SENTENCE_BOUNDARIES = [". ", ".\n", "? ", "! ", "\n"]


@dataclass
class Chunk:
    text: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)  # ceil division, matches Math.ceil


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\f", "\n")
    text = re.sub(r"(\s*\n){3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _snap_to_boundary(text: str, raw_offset: int, hard_limit: int) -> int:
    """Mirrors snapToBoundary() in chunker.js — snap an overlap start forward
    to a sentence boundary, falling back to a word boundary, so windows never
    begin mid-word."""
    if raw_offset <= 0:
        return 0

    search_window_end = min(raw_offset + 80, hard_limit)
    for bp in SENTENCE_BOUNDARIES:
        idx = text.find(bp, raw_offset)
        if idx != -1 and idx < search_window_end:
            return idx + len(bp)

    space_idx = text.find(" ", raw_offset)
    if space_idx != -1 and space_idx < search_window_end:
        return space_idx + 1

    i = raw_offset
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return i if i > 0 else raw_offset


def chunk_text(text: str, target_tokens: int = TARGET_CHUNK_TOKENS,
               max_tokens: int = MAX_CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS):
    """Direct port of chunkText() from chunker.js. Returns a list[Chunk]."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    if not text:
        return []

    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text)]
    raw_paras = [p for p in raw_paras if len(p) >= MIN_CHUNK_CHARS]

    chunks = []
    buffer = ""
    buf_start = 0
    chunk_index = 0
    char_cursor = 0

    def flush(force_text=None):
        nonlocal buffer, buf_start, chunk_index
        content = (force_text if force_text is not None else buffer).strip()
        if len(content) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                text=content,
                token_count=estimate_tokens(content),
                char_start=buf_start,
                char_end=buf_start + len(content),
                chunk_index=chunk_index,
            ))
            chunk_index += 1
        buffer = ""
        buf_start = char_cursor

    for para in raw_paras:
        first_line = para.split("\n")[0]
        is_heading = bool(HEADING_RE.match(first_line))

        if is_heading and len(buffer) > 0:
            flush()

        if len(para) > max_chars:
            if len(buffer) > 0:
                flush()

            s = 0
            while s < len(para):
                e = s + target_chars
                if e < len(para):
                    search_from = e - int(target_chars * 0.2)
                    for bp in SENTENCE_BOUNDARIES:
                        idx = para.rfind(bp, 0, e)
                        if idx > search_from:
                            e = idx + len(bp)
                            break
                e = min(e, len(para))
                slice_ = para[s:e].strip()
                if len(slice_) >= MIN_CHUNK_CHARS:
                    chunks.append(Chunk(
                        text=slice_,
                        token_count=estimate_tokens(slice_),
                        char_start=char_cursor + s,
                        char_end=char_cursor + e,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1

                if e >= len(para):
                    break

                raw_next = e - overlap_chars
                s = _snap_to_boundary(para, raw_next, e) if raw_next > s else e

            char_cursor += len(para) + 2
            buf_start = char_cursor
            continue

        if len(buffer) > 0 and (len(buffer) + len(para) + 2) > max_chars:
            flush()

        buffer = buffer + "\n\n" + para if buffer else para
        char_cursor += len(para) + 2

        if len(buffer) >= target_chars:
            flush()

    flush()
    return chunks
