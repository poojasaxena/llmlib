
# llmlib/data/prepare_corpus.py
from __future__ import annotations

from pathlib import Path
import os
import random
import re
from typing import Dict, List, Optional

DATASET_DIR = os.environ["GLOBAL_DATASETS_DIR"]
DATA_ROOT = Path(DATASET_DIR) / "llm/mixed_text/"
RAW = DATA_ROOT / "raw"
OUT = DATA_ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

WS_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
DOC_TAG_RE = re.compile(r"</?doc>\s*", re.IGNORECASE)

ROLE_PREFIX_RE = re.compile(
    r"^\s*(human(?:\s*\d+)?|assistant|system|question|answer|q|a)\s*:\s*",
    re.IGNORECASE,
)

INLINE_ROLE_RE = re.compile(
    r"\b(?:human(?:\s*\d+)?|assistant|system)\s*:\s*",
    re.IGNORECASE,
)

INLINE_QA_RE = re.compile(r"\b(?:question|answer)\s*:\s*", re.IGNORECASE)

BRACKET_CITES_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\[\d+(?:\]\[\d+)+\]")

READMORE_RE = re.compile(r"^\s*read more\s*:\s*", re.IGNORECASE)

SECTION_JUNK_RE = re.compile(r"^\s*(see also|references|external links)\b[: ]?", re.I)

DISCOURSE_RE = re.compile(
    r"\b(?:in summary|as a result| im addition|in practise| in related news|on the other hand|moreover|additionally|similarly)\b[:,]?\s*",
    re.IGNORECASE,
)

OVERVIEW_RE = re.compile(
    r"\bhere\s+is\s+a\s+short\s+overview\s+of\b[:,]?\s*", re.IGNORECASE
)

DOMAIN_KEYWORDS = {
    "matriarch",
    "calves",
    "herds",
    "ivory",
    "poaching",
    "proboscidea",
    "savanna",
    "bush",
    "forest elephant",
    "elephant",
    "elephants",
    "loxodonta",
}


def _has_domain_keyword(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in DOMAIN_KEYWORDS)


def clean_doc(
    text: str,
    *,
    min_len: int = 200,
    max_len: int = 200_000,
    drop_if_no_domain: bool = True,
    strip_discourse: bool = True,
    strip_overview: bool = True,
) -> str:
    """
    Clean a whole document into a SINGLE LINE sample.
    This avoids 'newline == sample' bugs and prevents 120-char wrapped sources
    from fragmenting training examples.
    """
    if not text:
        return ""

    # Split into lines first so we can drop junk header lines cleanly
    raw_lines = text.splitlines()
    kept: List[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue

        # Drop obvious non-content section junk lines
        if SECTION_JUNK_RE.match(s):
            continue

        # Remove "Read more:" prefix
        s = READMORE_RE.sub("", s).strip()

        # Strip leading role prefixes (keep content)
        s = ROLE_PREFIX_RE.sub("", s).strip()

        if not s:
            continue

        kept.append(s)

    if not kept:
        return ""

    # Join back into one text blob
    s = " ".join(kept)

    # Normalize whitespace + remove control chars
    s = CONTROL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    s = DOC_TAG_RE.sub("", s).strip()

    # Strip inline scaffolding tags
    s = INLINE_ROLE_RE.sub("", s)
    s = INLINE_QA_RE.sub("", s)

    # Remove bracket citations
    s = BRACKET_CITES_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()

    # Strip discourse glue (don’t drop whole doc)
    if strip_discourse:
        s = DISCOURSE_RE.sub("", s).strip()
        s = WS_RE.sub(" ", s).strip()

    # Strip the “overview template” phrase
    if strip_overview:
        s = OVERVIEW_RE.sub("", s).strip()
        s = WS_RE.sub(" ", s).strip()

    # If still contains multiple role markers, likely stitched garbage → drop
    low = s.lower()
    markers = sum(
        low.count(m) for m in (" q:", " a:", " human", " assistant", " system")
    )
    if markers >= 2:
        return ""

    # Length guardrails (doc-level)
    if len(s) < min_len or len(s) > max_len:
        return ""

    # Topic gate
    if drop_if_no_domain and not _has_domain_keyword(s):
        return ""

    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 40:
        return ""

    return s


def chunk_text(s: str, *, chunk_chars: int, overlap_chars: int = 200) -> List[str]:
    """
    Simple char-based chunking with overlap. Produces one-line chunks.
    """
    if chunk_chars <= 0:
        return [s]
    if len(s) <= chunk_chars:
        return [s]

    chunks = []
    step = max(1, chunk_chars - max(0, overlap_chars))
    i = 0
    n = len(s)
    while i < n:
        chunk = s[i : i + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def prepare(
    raw_dir: Path = RAW,
    out_dir: Path = OUT,
    file_out: str = "elephant_human_90_10_corpus.txt",
    shuffle: bool = True,
    dedupe: bool = True,
    seed: int = 42,
    # NEW:
    chunk_chars: int = 0,  # 0 = one file -> one sample
    overlap_chars: int = 200,
) -> Dict[str, int]:
    files = sorted(raw_dir.glob("**/*.txt"))
    raw_subdirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    print(f"Raw input directory: {raw_dir}")
    if raw_subdirs:
        print("Raw base directories:")
        for d in raw_subdirs:
            print(f"  - {d}")
    print(f"Found raw files: {len(files)}")
    for i, f in enumerate(files[:20], 1):
        print(f"  {i}. {f}")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")

    rng = random.Random(seed)

    samples: List[str] = []
    removed_empty = 0
    removed_clean = 0

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        doc = clean_doc(text)
        if not doc:
            removed_clean += 1
            continue

        for piece in chunk_text(
            doc, chunk_chars=chunk_chars, overlap_chars=overlap_chars
        ):
            if piece:
                samples.append(piece)
            else:
                removed_empty += 1

    if dedupe:
        seen = set()
        uniq = []
        for l in samples:
            k = l.strip().lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(l)
        samples = uniq

    if shuffle:
        rng.shuffle(samples)

    out_path = out_dir / file_out
    out_path.write_text("\n".join(samples) + "\n", encoding="utf-8")

    chars = sum(len(l) for l in samples)
    print(f"Wrote {out_path} | lines: {len(samples)}, chars: {chars}")
    print(f"Removed by clean_doc(): {removed_clean} | empty chunks: {removed_empty}")
    return {"lines": len(samples), "chars": chars}



