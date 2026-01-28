# llmlib/data/dedupe_and_split.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Callable
import re
from collections import Counter

drops = Counter()

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
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
READMORE_RE = re.compile(r"^\s*read more\s*:\s*", re.IGNORECASE)

DISCOURSE_RE = re.compile(
    r"\b(?:in related news|on the other hand|moreover|additionally|similarly)\b[:,]?\s*",
    re.IGNORECASE,
)
OVERVIEW_RE = re.compile(
    r"\bhere\s+is\s+a\s+short\s+overview\s+of\b[:,]?\s*", re.IGNORECASE
)
QUESTION_START_RE = re.compile(
    r"^\s*(how|what|why|when|where|who)\b", re.IGNORECASE
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

# ---- NEW: template-key helpers (near-template cap) ----
# Remove digits + punctuation; keep letters/spaces only.
TEMPLATE_CLEAN_RE = re.compile(r"[^a-z\s]+", re.IGNORECASE)
DIGITS_RE = re.compile(r"\d+")

# Small, safe stopword list to reduce "in the", "of the" dominance in template keys.
# Keep it short; we do NOT want aggressive NLP here.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "from", "by", "with", "without",
    "of", "for", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "they", "their", "its",
    "can", "may", "often", "typically", "usually",
}

def _has_domain_keyword(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in DOMAIN_KEYWORDS)

def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().split())

def clean_line(
    line: str,
    *,
    drops: Optional[Counter] = None,
    drop_urls: bool = True,
    strip_discourse: bool = True,
    strip_overview: bool = True,
    require_domain: bool = False,
    min_chars: int = 10,
    max_chars: int = 12000,
    max_tokens: int = 0,
    token_counter: Optional[Callable[[str], int]] = None,
    drop_short_questions: bool = False,
    question_max_tokens: int = 80,
    question_max_chars: int = 200,
) -> str:
    s = line.strip()
    if not s:
        if drops is not None:
            drops["empty"] += 1
        return ""

    if drop_urls and URL_RE.search(s):
        if drops is not None:
            drops["url"] += 1
        return ""

    s = CONTROL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    s = DOC_TAG_RE.sub("", s).strip()
    if not s:
        if drops is not None:
            drops["empty_after_norm"] += 1
        return ""

    s = READMORE_RE.sub("", s).strip()
    if not s:
        if drops is not None:
            drops["readmore_only"] += 1
        return ""

    # strip scaffolding, keep content
    s = ROLE_PREFIX_RE.sub("", s).strip()
    s = INLINE_ROLE_RE.sub("", s)
    s = INLINE_QA_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    if not s:
        if drops is not None:
            drops["role_only"] += 1
        return ""

    s = BRACKET_CITES_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()

    if strip_discourse:
        s = DISCOURSE_RE.sub("", s).strip()
        s = WS_RE.sub(" ", s).strip()

    if strip_overview:
        s = OVERVIEW_RE.sub("", s).strip()
        s = WS_RE.sub(" ", s).strip()
    if not s:
        if drops is not None:
            drops["empty_after_strip"] += 1
        return ""

    # stitched garbage detector (multiple roles mid-line)
    low = s.lower()
    markers = sum(low.count(m) for m in (" q:", " a:", " human", " assistant", " system"))
    if markers >= 2:
        if drops is not None:
            drops["stitched_roles"] += 1
        return ""

    if len(s) < min_chars:
        if drops is not None:
            drops["too_short"] += 1
        return ""
    if len(s) > max_chars:
        if drops is not None:
            drops["too_long"] += 1
        return ""

    if max_tokens > 0:
        if token_counter is not None:
            tok_count = token_counter(s)
        else:
            tok_count = len(s.split())
        if tok_count > max_tokens:
            if drops is not None:
                drops["too_many_tokens"] += 1
            return ""

    if drop_short_questions:
        tok_count = len(s.split())
        if (s.endswith("?") or QUESTION_START_RE.match(s)) and (
            tok_count <= question_max_tokens or len(s) <= question_max_chars
        ):
            if drops is not None:
                drops["short_question"] += 1
            return ""

    if require_domain and not _has_domain_keyword(s):
        if drops is not None:
            drops["no_domain"] += 1
        return ""

    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 8:
        if drops is not None:
            drops["low_alpha"] += 1
        return ""

    return s

def _prefix_key(line: str, n_words: int) -> str:
    parts = line.lower().split()
    return " ".join(parts[:n_words])


def _iter_ngrams(words: list[str], n: int) -> list[str]:
    if n <= 0 or len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

def _template_key(line: str, n_words: int) -> str:
    """
    Near-template signature:
    - lowercase
    - remove digits
    - strip punctuation/symbols to spaces
    - drop a small set of stopwords
    - take first n_words
    """
    s = line.lower()
    s = DIGITS_RE.sub(" ", s)
    s = TEMPLATE_CLEAN_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    toks = [t for t in s.split() if t and t not in STOPWORDS and len(t) > 2]
    if not toks:
        return ""
    return " ".join(toks[:n_words])

def dedupe_and_filter_corpus(
    input_file: Path,
    output_file: Path,
    *,
    encoding: str = "utf-8",
    min_len: int = 10,
    max_len: int = 12000,
    max_tokens: int = 0,
    token_counter: Optional[Callable[[str], int]] = None,
    drop_short_questions: bool = False,
    question_max_tokens: int = 80,
    question_max_chars: int = 200,
    drop_urls: bool = True,
    strip_discourse: bool = True,
    strip_overview: bool = True,
    require_domain: bool = False,
    # prefix cap: prevent one template/prefix from dominating
    prefix_cap_words: int = 0,  # 0 disables
    prefix_cap_max: int = 0,  # 0 disables
    # ---- NEW: near-template cap (semantic-ish) ----
    template_cap_words: int = 0,  # 0 disables
    template_cap_max: int = 0,  # 0 disables
    # ---- NEW: n-gram cap (repeated phrasing) ----
    ngram_cap_n: int = 0,  # 0 disables
    ngram_cap_max: int = 0,  # 0 disables
) -> Tuple[int, int, int]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw_lines = [
        ln.rstrip("\n")
        for ln in input_file.open(encoding=encoding, errors="ignore")
        if ln.strip()
    ]
    original = len(raw_lines)

    cleaned = []
    drops_local = Counter()
    removed = 0
    for ln in raw_lines:
        c = clean_line(
            ln,
            drops=drops_local,
            drop_urls=drop_urls,
            strip_discourse=strip_discourse,
            strip_overview=strip_overview,
            require_domain=require_domain,
            min_chars=min_len,
            max_chars=max_len,
            max_tokens=max_tokens,
            token_counter=token_counter,
            drop_short_questions=drop_short_questions,
            question_max_tokens=question_max_tokens,
            question_max_chars=question_max_chars,
        )
        if not c:
            removed += 1
            continue
        cleaned.append(c)

    print(f"[clean] removed: {removed:,}/{original:,} ({(removed/max(1,original))*100:.2f}%)")
    print(f"[clean] kept:   {len(cleaned):,}/{original:,}")
    print("[clean] DROP REASONS:", drops_local.most_common(20))

    # normalized dedupe
    seen = set()
    unique = []
    dedup_removed = 0
    for ln in cleaned:
        k = _norm_key(ln)
        if k in seen:
            dedup_removed += 1
            continue
        seen.add(k)
        unique.append(ln)
    print(f"[dedupe] removed: {dedup_removed:,} | unique: {len(unique):,}")

    # optional prefix cap (after dedupe)
    if prefix_cap_words > 0 and prefix_cap_max > 0:
        counts: Dict[str, int] = {}
        capped = []
        dropped = 0
        for ln in unique:
            pk = _prefix_key(ln, prefix_cap_words)
            c = counts.get(pk, 0)
            if c >= prefix_cap_max:
                dropped += 1
                continue
            counts[pk] = c + 1
            capped.append(ln)
        print(f"[prefix-cap] cap={prefix_cap_max} words={prefix_cap_words} dropped: {dropped} kept: {len(capped)}")
        unique = capped

    # ---- NEW: near-template cap (catches Wikipedia-ish repeated phrasing) ----
    if template_cap_words > 0 and template_cap_max > 0:
        tcounts: Dict[str, int] = {}
        kept = []
        dropped = 0
        empty_key = 0

        for ln in unique:
            tk = _template_key(ln, template_cap_words)
            if not tk:
                empty_key += 1
                kept.append(ln)
                continue

            c = tcounts.get(tk, 0)
            if c >= template_cap_max:
                dropped += 1
                continue
            tcounts[tk] = c + 1
            kept.append(ln)

        print(
            f"[template-cap] cap={template_cap_max} words={template_cap_words} "
            f"dropped: {dropped} kept: {len(kept)} empty_key: {empty_key}"
        )
        unique = kept

    # ---- NEW: n-gram cap (after template cap) ----
    if ngram_cap_n > 0 and ngram_cap_max > 0:
        counts: Dict[str, int] = {}
        kept = []
        dropped = 0
        for ln in unique:
            words = ln.lower().split()
            ngrams = _iter_ngrams(words, ngram_cap_n)
            if not ngrams:
                kept.append(ln)
                continue
            if any(counts.get(ng, 0) >= ngram_cap_max for ng in ngrams):
                dropped += 1
                continue
            for ng in ngrams:
                counts[ng] = counts.get(ng, 0) + 1
            kept.append(ln)

        print(
            f"[ngram-cap] n={ngram_cap_n} cap={ngram_cap_max} "
            f"dropped: {dropped} kept: {len(kept)}"
        )
        unique = kept

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(unique) + "\n", encoding=encoding)
    return original, len(unique), len(unique)
