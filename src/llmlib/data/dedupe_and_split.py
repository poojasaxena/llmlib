import os
from pathlib import Path
from typing import Tuple


def dedupe_and_filter_corpus(
    input_file: Path,
    output_file: Path,
    min_len: int = 10,
    max_len: int = 500,
    encoding: str = "utf-8",
) -> Tuple[int, int, int]:
    """
    Deduplicate and filter a text corpus while preserving line order.

    Steps:
    - Strip whitespace
    - Remove empty lines
    - Deduplicate while preserving order
    - Filter lines by length
    - Write cleaned corpus to output_file

    Args:
        input_file: Path to input text file
        output_file: Path to write cleaned text
        min_len: Minimum allowed line length
        max_len: Maximum allowed line length
        encoding: File encoding

    Returns:
        (original_lines, unique_lines, filtered_lines)
    """

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read + strip + drop empty
    lines = [
        line.rstrip() for line in input_file.open(encoding=encoding) if line.strip()
    ]

    # Deduplicate (preserve order)
    seen = set()
    unique = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)

    # Length filter
    filtered = [line for line in unique if min_len <= len(line) <= max_len]

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(filtered) + "\n",
        encoding=encoding,
    )

    return len(lines), len(unique), len(filtered)
