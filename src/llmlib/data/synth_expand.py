## llmlib/data/synth_expand.py
from pathlib import Path
from typing import Iterable


def generate_synthetic_expansions(
    src_file: Path,
    output_file: Path,
    max_lines: int = 20_000,
    encoding: str = "utf-8",
) -> int:
    """
    Generate simple synthetic expansions from a training corpus.

    Transformations applied:
    - Identity (original line)
    - Light paraphrase (pronoun + adverb tweaks)
    - Periodic QA-style wrapping

    Args:
        src_file: Path to source text file (e.g. train.txt)
        output_file: Path to write synthetic data
        max_lines: Maximum number of generated lines to write
        encoding: Text encoding

    Returns:
        Number of lines written
    """

    if not src_file.exists():
        raise FileNotFoundError(f"Source file not found: {src_file}")

    lines = [line.rstrip() for line in src_file.open(encoding=encoding) if line.strip()]

    synthetic: list[str] = []

    for i, line in enumerate(lines):
        # 1) original
        synthetic.append(line)

        # 2) light paraphrase
        synthetic.append(
            line.replace("I ", "I often ").replace(" you ", " you sometimes ")
        )

        # 3) periodic QA-style expansion
        if i % 3 == 0:
            synthetic.append(f"Question: {line}")
            synthetic.append(f"Answer: {line}")

        if len(synthetic) >= max_lines:
            break

    synthetic = synthetic[:max_lines]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(synthetic) + "\n", encoding=encoding)

    return len(synthetic)
