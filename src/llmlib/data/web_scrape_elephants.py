#!/usr/bin/env python3
"""
web_scrape_elephants.py  (FINAL / CLEAN)

Goals
- Be polite to Wikipedia (use MediaWiki API, not raw HTML scraping).
- Avoid 403/robots issues (rate limit + backoff).
- Produce CLEAN training lines:
  - one item per line
  - no Q:/A:/Human:/Assistant scaffolding
  - remove wiki citations like [12], [1][2]
  - drop section junk (References, See also, External links)
  - keep short-but-complete sentences (do NOT drop '?' or short lines blindly)

Output:
  ~/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw/web_scraped_elephants.txt

Usage:
  python web_scrape_elephants.py [OUT_DIR] [MAX_ITEMS]
Example:
  python web_scrape_elephants.py ~/PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw 1200
"""

from __future__ import annotations

import re
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests


# ----------------------------
# Cleaning utilities
# ----------------------------

RE_WS = re.compile(r"\s+")
RE_CITES = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")          # [12] or [1, 2]
RE_CITES_CHAIN = re.compile(r"\[\d+(?:\]\[\d+)*\]")            # [12][20]
RE_SECTION_JUNK = re.compile(r"^(see also|references|external links)\b[: ]?", re.I)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)

BAD_ANYWHERE = [
    "in related news",
    "on the other hand",
    # keep these *rare* if they exist; we drop them here to avoid loop attractors
    "moreover",
    "additionally",
    "similarly",
]

BAD_PREFIX_RE = re.compile(
    r"^\s*(human(?:\s*\d+)?|assistant|system|question|answer|q|a)\s*:\s*",
    re.I,
)


def clean_line(s: str, *, min_chars: int = 25, max_chars: int = 800) -> str:
    s = s.strip()
    if not s:
        return ""

    # remove obvious scaffolding prefixes (but keep the content!)
    s = BAD_PREFIX_RE.sub("", s).strip()
    if not s:
        return ""

    # normalize whitespace + remove control chars
    s = RE_WS.sub(" ", s)
    s = "".join(ch for ch in s if ord(ch) >= 32).strip()

    low = s.lower()
    if RE_SECTION_JUNK.match(s):
        return ""

    if RE_URL.search(s):
        return ""

    # drop "poison glue" anywhere
    if any(b in low for b in BAD_ANYWHERE):
        return ""

    # remove wiki citations
    s = RE_CITES.sub("", s)
    s = RE_CITES_CHAIN.sub("", s)
    s = RE_WS.sub(" ", s).strip()

    # basic sanity (keep short-but-good lines; don't drop '?' automatically)
    if len(s) < min_chars:
        # allow very short but complete sentences sometimes, e.g. "Elephants are herbivores."
        # We accept if it ends with punctuation and has enough letters.
        alpha = sum(ch.isalpha() for ch in s)
        if not (alpha >= 10 and s[-1] in ".!?"):
            return ""

    if len(s) > max_chars:
        return ""

    # Drop super fragmenty lines even if long (low alpha ratio)
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 12:
        return ""

    # Light normalize: capitalize first letter (optional)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s


def split_to_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitter.
    Keeps punctuation. Good enough for Wikipedia extracts.
    """
    text = RE_WS.sub(" ", text).strip()
    if not text:
        return []
    # split on .!? followed by space + capital (rough heuristic)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------
# Wikipedia API scraping
# ----------------------------

@dataclass
class WikiPage:
    title: str
    extract: str


class ElephantWikiScraper:
    """
    Uses MediaWiki API extracts endpoint:
      https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=...
    This is API-friendly and avoids HTML scraping.
    """

    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        *,
        delay: float = 1.2,
        jitter: float = 0.5,
        timeout: float = 15.0,
        max_retries: int = 5,
        seed: int = 42,
        user_agent: str = "ElephantGPTDataBot/1.0 (contact: local-script; purpose: research dataset building)",
    ):
        self.delay = delay
        self.jitter = jitter
        self.timeout = timeout
        self.max_retries = max_retries
        self.rng = random.Random(seed)
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": user_agent})

    def _sleep(self):
        time.sleep(self.delay + self.rng.random() * self.jitter)

    def fetch_extract(self, title: str) -> Optional[WikiPage]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "titles": title,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.sess.get(self.API_URL, params=params, timeout=self.timeout)
                # 429 or transient errors -> backoff
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                data = r.json()

                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    return None
                # pages is dict keyed by pageid
                page = next(iter(pages.values()))
                extract = page.get("extract", "") or ""
                title_out = page.get("title", title) or title

                self._sleep()
                return WikiPage(title=title_out, extract=extract)

            except Exception as e:
                last_err = e
                # exponential-ish backoff
                backoff = min(20.0, (2 ** (attempt - 1)) + self.rng.random())
                time.sleep(backoff)

        print(f"❌ Failed to fetch '{title}': {last_err}")
        return None

    def extract_clean_lines(self, page: WikiPage) -> List[str]:
        """
        Turn the extract into sentence-level lines.
        """
        text = page.extract
        if not text:
            return []

        # Drop very noisy "Contents" style lines if any
        raw_lines: List[str] = []
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            if RE_SECTION_JUNK.match(para):
                continue
            # split paragraph into sentences
            raw_lines.extend(split_to_sentences(para))

        cleaned = []
        for s in raw_lines:
            c = clean_line(s)
            if c:
                cleaned.append(c)

        return cleaned


def generate_static_facts() -> List[str]:
    """
    Small curated facts (safe, general, and not too quirky).
    Keep this modest to avoid dominating prefixes.
    """
    return [
        "Elephants are large herbivorous mammals found in Africa and Asia.",
        "African elephants are generally larger than Asian elephants.",
        "Elephants use their trunks for breathing, smelling, drinking, and grasping.",
        "Elephants communicate using vocalizations, touch, and body language.",
        "Elephant family groups are often led by an older female called a matriarch.",
        "Elephants can travel long distances to find food and water.",
        "Poaching and habitat loss are major threats to elephant populations.",
        "Elephants may use mud and water to help cool their bodies and protect their skin.",
        "Elephants play important roles in ecosystems by shaping vegetation and dispersing seeds.",
    ]


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    # Defaults compatible with your pipeline
    if len(sys.argv) >= 2:
        out_dir = Path(sys.argv[1]).expanduser()
    else:
        out_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw"

    max_items = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "web_scraped_elephants.txt"

    titles = [
        "Elephant",
        "African elephant",
        "Asian elephant",
        "Elephant cognition",
        "Elephant behaviour",
        "Elephant communication",
        "Elephant conservation",
        "Elephantidae",
        "Loxodonta",
        "Elephas maximus",
    ]

    scraper = ElephantWikiScraper(delay=1.1, jitter=0.6, max_retries=5)

    all_lines: List[str] = []

    # 1) add small curated facts (kept small)
    all_lines.extend([clean_line(x) for x in generate_static_facts() if clean_line(x)])

    # 2) fetch wiki extracts via API
    for t in titles:
        print(f"📡 Fetching Wikipedia extract: {t}")
        page = scraper.fetch_extract(t)
        if not page:
            continue
        lines = scraper.extract_clean_lines(page)
        print(f"   ✅ got {len(lines)} clean lines from '{page.title}'")
        all_lines.extend(lines)
        if len(all_lines) >= max_items:
            break

    # de-dupe (case-insensitive) but preserve order
    seen = set()
    final: List[str] = []
    for s in all_lines:
        k = s.strip().lower()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        final.append(s)
        if len(final) >= max_items:
            break

    out_file.write_text("\n".join(final) + "\n", encoding="utf-8")
    size_kb = out_file.stat().st_size / 1024

    print("\n✅ Scraping complete!")
    print(f"📁 Saved {len(final)} lines to: {out_file}")
    print(f"📊 File size: {size_kb:.1f} KB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
