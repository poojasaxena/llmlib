#!/usr/bin/env python3
"""
advanced_synth_expand.py (CLEAN VERSION)
Generates diverse synthetic elephant-domain text WITHOUT chat/Q/A scaffolding.

Outputs:
- fact variations (single sentences)
- mini-dialogue paraphrased into a paragraph (no Human/Assistant tags)
- Q/A converted into declarative factual statements (no Q:/A:)
- descriptive passages (multi-sentence paragraphs in one line)

Write each item as ONE line (LM-friendly for your current pipeline).
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple


class AdvancedElephantDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.elephant_facts = self._load_base_facts()
        self.topics = self._load_topics()

    def _load_base_facts(self) -> List[str]:
        return [
            "African elephants are larger than Asian elephants.",
            "Elephants use their trunks for breathing, drinking, and communication.",
            "Baby elephants are called calves and weigh about 250 pounds at birth.",
            "Elephants have excellent memories and can remember other elephants for decades.",
            "Asian elephants have smaller ears than African elephants.",
            "Elephants are herbivores and eat up to 300 pounds of vegetation daily.",
            "Elephant tusks are elongated incisor teeth.",
            "Elephants live in matriarchal societies led by the oldest female.",
            "Elephants communicate through infrasonic sounds below human hearing range.",
            "Wild elephants can live 60–70 years in the wild.",
            "Elephants are found in Africa and Asia in grasslands and forests.",
            "Elephants play a crucial role in their ecosystem as a keystone species.",
            "Poaching for ivory remains a major threat to elephant populations.",
            "Elephants show empathy and may mourn dead companions.",
            "A group of elephants is called a herd or a parade.",
        ]

    def _load_topics(self) -> List[str]:
        return [
            "elephant behavior",
            "elephant habitat",
            "elephant conservation",
            "elephant anatomy",
            "elephant intelligence",
            "elephant families",
            "elephant migration",
            "elephant communication",
            "elephant diet",
            "elephant threats",
            "elephant reproduction",
            "elephant evolution",
        ]

    # --- generators ---

    def generate_fact_variations(
        self, base_facts: List[str], num_variations: int = 6
    ) -> List[str]:
        """
        Produce light stylistic variations without creating a single dominant prefix.
        """
        starters = [
            "",
            "",
            "",  # keep many plain
            "Notably, ",
            "In many cases, ",
            "Researchers note that ",
            "A useful fact is that ",
            "One key point is that ",
        ]
        out: List[str] = []
        for fact in base_facts:
            fact = fact.strip()
            if not fact.endswith("."):
                fact += "."
            for _ in range(num_variations):
                s = self.rng.choice(starters) + fact
                out.append(s.strip())
        return out

    def generate_mini_dialogue_paragraphs(self, num_items: int = 200) -> List[str]:
        """
        Turn a conversation idea into a short paragraph (single line), no role tags.
        """
        follow_ups = [
            "This can be compared to other large mammals in terms of social structure.",
            "Researchers study this using long-term field observation and tracking.",
            "Conservation work is often needed because habitat loss and poaching still occur.",
            "The details can vary across regions and between African and Asian elephants.",
        ]
        out: List[str] = []
        for _ in range(num_items):
            topic = self.rng.choice(self.topics)
            fact1 = self.rng.choice(self.elephant_facts)
            fact2 = self.rng.choice(self.elephant_facts)
            extra = self.rng.choice(follow_ups)

            paragraph = (
                f"Here is a short overview of {topic}. " f"{fact1} {fact2} {extra}"
            )
            out.append(paragraph.strip())
        return out

    def generate_declarative_qa(self, num_items: int = 300) -> List[str]:
        """
        Convert Q/A templates into declarative statements (no Q:/A:).
        """
        templates: List[Tuple[str, str]] = [
            (
                "average lifespan",
                "{etype} elephants typically live {lifespan} years in the wild.",
            ),
            (
                "adult weight",
                "Adult {etype} elephants often weigh between {weight_range}.",
            ),
            (
                "diet by season",
                "During the {season}, elephants primarily eat {food_items}.",
            ),
            (
                "long-distance communication",
                "Elephants use infrasonic calls that can travel several kilometers.",
            ),
            (
                "large ears",
                "Large ears help elephants regulate body temperature in hot climates.",
            ),
        ]
        elephant_types = ["African", "Asian", "forest", "savanna"]
        lifespans = ["50–70", "60–70", "40–60"]
        weight_ranges = ["4,000–7,000 kg", "3,000–5,000 kg", "2,700–4,000 kg"]
        seasons = ["dry season", "wet season", "winter", "summer"]
        food_items = ["grasses and bark", "fruits and leaves", "roots and branches"]

        out: List[str] = []
        for _ in range(num_items):
            _, a_tmpl = self.rng.choice(templates)
            s = a_tmpl
            s = s.replace("{etype}", self.rng.choice(elephant_types))
            s = s.replace("{lifespan}", self.rng.choice(lifespans))
            s = s.replace("{weight_range}", self.rng.choice(weight_ranges))
            s = s.replace("{season}", self.rng.choice(seasons))
            s = s.replace("{food_items}", self.rng.choice(food_items))
            out.append(s.strip())
        return out

    def generate_descriptive_passages(self, num_items: int = 100) -> List[str]:
        starters = [
            "In the African savanna, elephants roam across vast territories.",
            "Asian elephants inhabit dense forests and varied landscapes.",
            "Elephant herds are often led by an experienced matriarch.",
            "Conservation efforts for elephants face complex challenges worldwide.",
            "Elephant intelligence continues to interest researchers and wildlife observers.",
        ]
        out: List[str] = []
        for _ in range(num_items):
            starter = self.rng.choice(starters)
            facts = self.rng.sample(self.elephant_facts, self.rng.randint(2, 4))
            passage = starter + " " + " ".join(facts)
            out.append(passage.strip())
        return out

    def generate_all_synthetic_data(self, output_file: Path) -> int:
        all_data: List[str] = []

        # balance similar to your original intent
        all_data.extend(
            self.generate_fact_variations(self.elephant_facts, num_variations=6)
        )
        all_data.extend(self.generate_mini_dialogue_paragraphs(num_items=200))
        all_data.extend(self.generate_declarative_qa(num_items=300))
        all_data.extend(self.generate_descriptive_passages(num_items=100))

        self.rng.shuffle(all_data)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for item in all_data:
                item = item.strip()
                if not item:
                    continue
                # one item per line
                f.write(item + "\n")

        return len(all_data)


def main():
    # Match your pipeline style: write into RAW so prepare() can pick it up
    output_dir = (
        Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_advanced.txt"

    generator = AdvancedElephantDataGenerator(seed=42)
    n = generator.generate_all_synthetic_data(output_file)

    print(f"✅ Wrote {n} synthetic lines to: {output_file}")


if __name__ == "__main__":
    main()
