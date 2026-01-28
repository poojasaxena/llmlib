#!/usr/bin/env python3
"""
Data augmentation for elephant corpus (CLEAN + ANTI-TEMPLATE VERSION).

Goals:
- Generate useful variety without injecting style attractors.
- No chat scaffolding (Human/Assistant) and no Q:/A: blocks.
- Avoid discourse glue loops ("Moreover", "Additionally", etc.)
- One training item per line (plays well with combiner + dedupe/clean).
- EXTRA: Avoid dominant prefixes by enforcing an internal prefix-cap.

Output: plain text lines suitable for LM pretraining.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List, Dict, Tuple


class ElephantDataAugmenter:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Internal anti-template guard (separate from your pipeline prefix-cap)
        self.prefix_cap_words = 6     # how many starting words define a "prefix"
        self.prefix_cap_max = 2       # max lines allowed per prefix (inside augmentation)
        self._prefix_counts: Dict[str, int] = {}

        self.paraphrase_templates = self._load_paraphrase_templates()

    def _load_paraphrase_templates(self) -> Dict[str, List[str]]:
        # Keep many blanks so no single starter dominates
        return {
            "fact_starters": [
                "", "", "", "", "", "", "", "", "",  # mostly plain
                "Researchers note that",
                "Studies suggest that",
                "Field observations indicate that",
                "Evidence indicates that",
                "In some populations,",
                "In many habitats,",
            ],
        }

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for p in parts:
            s = p.strip()
            if len(s) >= 30:
                out.append(s)
        return out

    # -------------------------
    # Anti-template helpers
    # -------------------------
    def _norm_space(self, s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _prefix_key(self, s: str) -> str:
        toks = s.lower().split()
        return " ".join(toks[: self.prefix_cap_words])

    def _allow_by_prefix_cap(self, s: str) -> bool:
        key = self._prefix_key(s)
        n = self._prefix_counts.get(key, 0)
        if n >= self.prefix_cap_max:
            return False
        self._prefix_counts[key] = n + 1
        return True

    def _maybe_prepend_starter(self, s: str) -> str:
        starter = self.rng.choice(self.paraphrase_templates["fact_starters"]).strip()
        if not starter:
            return s
        # lower-case sentence start after starter for flow
        t = s[0].lower() + s[1:] if len(s) > 1 else s
        return f"{starter} {t}".strip()

    # -------------------------
    # Paraphrasing
    # -------------------------
    def paraphrase_sentences(self, sentences: List[str], num_variations: int = 2) -> List[str]:
        out: List[str] = []
        for s in sentences:
            if len(s) < 40:
                continue

            for _ in range(num_variations):
                # IMPORTANT: removed "expand" technique (it created repeated boilerplate endings)
                technique = self.rng.choice(["starter", "light_edit", "light_reorder"])

                if technique == "starter":
                    out.append(self._maybe_prepend_starter(s))

                elif technique == "light_reorder":
                    # mild restructuring without adding canned phrases
                    # If sentence has a comma, swap clauses sometimes.
                    if "," in s and self.rng.random() < 0.5:
                        parts = [p.strip() for p in s.split(",", 1)]
                        if len(parts) == 2 and all(len(p) > 15 for p in parts):
                            out.append(f"{parts[1]} {parts[0]}".strip())
                        else:
                            out.append(s)
                    else:
                        out.append(s)

                else:  # light_edit
                    t = s
                    # gentle replacements (small + varied)
                    replacements: List[Tuple[str, str]] = [
                        (" are ", " can be "),
                        (" use ", " rely on "),
                        (" typically ", " often "),
                        (" large ", " sizable "),
                        (" important ", " significant "),
                        (" helps ", " can help "),
                        (" may ", " can "),
                        (" usually ", " often "),
                    ]
                    self.rng.shuffle(replacements)
                    for a, b in replacements[: self.rng.randint(0, 2)]:
                        if a in t:
                            t = t.replace(a, b, 1)
                    out.append(t.strip())

        return out

    # -------------------------
    # Declarative "helper" lines (now rare + diverse + prefix-capped)
    # -------------------------
    def generate_declarative_from_statement(self, statement: str) -> List[str]:
        s = statement.strip()
        low = s.lower()

        # Diverse pools; many blanks so we often keep the original sentence as-is.
        pools = {
            "weigh": [
                "", "",
                "Elephant weight varies by species, sex, and age.",
                "Body mass differs across elephant species and life stages.",
                "Adult weight depends on species and condition.",
            ],
            "diet": [
                "", "",
                "Elephant diets shift with habitat and season.",
                "Diet composition changes across wet and dry periods.",
                "Feeding choices vary with resource availability.",
            ],
            "communicat": [
                "", "",
                "Elephants communicate through sound, touch, and posture.",
                "Communication is multi-channel, including vocal and tactile cues.",
                "Signals can be vocal, chemical, and body-based.",
            ],
            "trunk": [
                "", "",
                "The trunk combines strength with fine control.",
                "The trunk supports feeding, smell, and social contact.",
                "Trunk use enables exploration and manipulation.",
            ],
            "ears": [
                "", "",
                "Large ears help with cooling and signaling.",
                "Ear flapping supports thermoregulation and display.",
                "Ears contribute to temperature control in heat.",
            ],
        }

        key = None
        if "weigh" in low:
            key = "weigh"
        elif "eat" in low or "diet" in low or "forag" in low:
            key = "diet"
        elif "communicat" in low or "rumbl" in low or "infrasound" in low:
            key = "communicat"
        elif "trunk" in low:
            key = "trunk"
        elif "ears" in low:
            key = "ears"

        if key:
            starter = self.rng.choice(pools[key]).strip()
            line = f"{starter} {s}".strip() if starter else s
            line = self._norm_space(line)
            # internal prefix cap prevents one starter dominating
            if self._allow_by_prefix_cap(line):
                return [line]
            return []

        # generic fallback: extremely rare, and no "Elephant facts:" style label
        if self.rng.random() < 0.03:
            line = self._norm_space(s)
            if self._allow_by_prefix_cap(line):
                return [line]
        return []

    # -------------------------
    # Context variations (connector-light)
    # -------------------------
    def create_context_variations(self, sentences: List[str], max_pairs: int = 1200) -> List[str]:
        out: List[str] = []

        # Keep connectors minimal to avoid glue dominance
        connectors = ["", "", "", "", "", "Also,"]

        if len(sentences) < 2:
            return out

        idxs = list(range(len(sentences) - 1))
        self.rng.shuffle(idxs)
        idxs = idxs[:max_pairs]

        for i in idxs:
            a = sentences[i].strip()
            b = sentences[i + 1].strip()
            if len(a) < 40 or len(b) < 40:
                continue

            # keep only pairs relevant to elephants (to avoid cross-topic weirdness)
            if "elephant" not in a.lower() and "elephant" not in b.lower():
                continue

            conn = self.rng.choice(connectors)
            if conn:
                combined = f"{a} {conn} {b[0].lower() + b[1:] if len(b) > 1 else b}"
            else:
                combined = f"{a} {b}"

            combined = self._norm_space(combined)
            if len(combined) >= 40 and self._allow_by_prefix_cap(combined):
                out.append(combined)

        return out

    # -------------------------
    # Main augmentation
    # -------------------------
    def augment_corpus(self, input_file: Path, output_file: Path) -> int:
        print(f"📖 Reading corpus from: {input_file}")

        text = input_file.read_text(encoding="utf-8", errors="ignore")
        sentences = self._split_sentences(text)
        print(f"📝 Found {len(sentences)} sentences to augment")

        augmented: List[str] = []

        # 1) Paraphrase (bounded)
        print("  🔄 Creating paraphrases...")
        src = sentences[:350]  # slightly smaller than before to reduce repeated style
        augmented.extend(self.paraphrase_sentences(src, num_variations=2))

        # 2) Declarative helper lines (now MUCH smaller)
        print("  🧩 Creating declarative helper lines...")
        for s in sentences[:80]:  # was 300
            augmented.extend(self.generate_declarative_from_statement(s))

        # 3) Context variations (bounded + sampled)
        print("  🔗 Creating context variations...")
        augmented.extend(self.create_context_variations(sentences, max_pairs=1000))

        # final clean + dedupe
        cleaned: List[str] = []
        seen = set()
        for x in augmented:
            x = self._norm_space(x)
            if len(x) < 40:
                continue
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(x)

        self.rng.shuffle(cleaned)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(item + "\n")

        size_kb = output_file.stat().st_size / 1024
        print("✅ Augmentation complete!")
        print(f"📁 Generated {len(cleaned)} augmented lines")
        print(f"📊 Output size: {size_kb:.1f} KB")

        return len(cleaned)


def main():
    corpus_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out"
    input_file = corpus_dir / "elephant_human_90_10_corpus.txt"

    output_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "augmented_corpus.txt"

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return

    augmenter = ElephantDataAugmenter(seed=42)
    n = augmenter.augment_corpus(input_file, output_file)

    print("\n🎉 Data augmentation complete!")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Lines:  {n}")


if __name__ == "__main__":
    main()
