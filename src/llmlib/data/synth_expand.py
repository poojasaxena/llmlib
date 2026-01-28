# llmlib/data/synth_expand.py
from __future__ import annotations
from pathlib import Path
import random
import re


ROLE_PREFIX_RE = re.compile(
    r"^\s*(human(?:\s*\d+)?|assistant|system|question|q|answer|a)\s*:\s*", re.IGNORECASE
)

DISCOURSE_RE = re.compile(
    r"\b(?:on the other hand|moreover|additionally)\b[:,]?\s*", re.IGNORECASE
)


def _clean_src_line(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    # avoid bootstrapping scaffolded lines into more scaffolded lines
    if ROLE_PREFIX_RE.match(s):
        return ""
    # remove heavy discourse glue from sources
    s = DISCOURSE_RE.sub("", s).strip()
    # drop very short fragments
    if len(s) < 30:
        return ""
    return s

import random
import re


WS_RE = re.compile(r"\s+")


def _rephrase_light(s: str, rng: random.Random) -> str:
    """
    Light rephrase without adding opener-prefix attractors.
    Prefer small internal edits; optionally add a short suffix.
    """
    out = s

    swaps = [
        (" are ", " tend to be "),
        (" can ", " may "),
        (" usually ", " often "),
        (" primarily ", " mostly "),
        (" important ", " significant "),
        (" large ", " sizable "),
        (" help ", " support "),
    ]
    rng.shuffle(swaps)
    for a, b in swaps[: rng.randint(0, 2)]:
        if a in out:
            out = out.replace(a, b, 1)

    # Optional small suffix (rare) instead of prefix (avoids “starter loops”)
    suffixes = [
        " This can vary by habitat and season.",
        " This depends on age and local conditions.",
        "",
        "",
        "",
        "",
    ]
    out = (out + rng.choice(suffixes)).strip()
    return out

   


def _definition_style(s: str, rng: random.Random) -> str:
    # turn "X ... is/are ..." into "Definition: X is ..."
    # heuristic: pick first 3–6 words as "topic"
    words = s.split()
    if len(words) < 8:
        return ""
    topic = " ".join(words[: rng.randint(3, 6)])
    templates = [
        f"{topic}: {s}",
        f"{topic} — {s}",
        f"{topic}. {s}",
    ]
    return rng.choice(templates)


def generate_synthetic_expansions(
    src_file: Path,
    output_file: Path,
    max_lines: int = 20_000,
    expansions_per_line: int = 2,
    seed: int = 42,
    encoding: str = "utf-8",
) -> int:
    """
    Safer synthetic expansion:
    - No Q/A scaffolding
    - No fixed phrase inflation ("I often", "you sometimes")
    - Adds small rephrases + occasional definition-style formatting
    """
    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    rng = random.Random(seed)

    src_lines = [
        ln.rstrip("\n") for ln in src_file.open(encoding=encoding) if ln.strip()
    ]
    synthetic: list[str] = []

    for ln in src_lines:
        base = _clean_src_line(ln)
        if not base:
            continue

        # Always keep original
        synthetic.append(base)

        # Add up to N expansions
        candidates = []
        candidates.append(_rephrase_light(base, rng))
        if rng.random() < 0.35:
            ds = _definition_style(base, rng)
            if ds:
                candidates.append(ds)

        # pick unique candidates (avoid duplicates)
        uniq = []
        seen = set()
        for c in candidates:
            c = c.strip()
            if not c or c == base:
                continue
            key = " ".join(c.lower().split())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        for c in uniq[:expansions_per_line]:
            synthetic.append(c)

        if len(synthetic) >= max_lines:
            break

    synthetic = synthetic[:max_lines]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(synthetic) + "\n", encoding=encoding)
    return len(synthetic)
