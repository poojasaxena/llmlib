from pathlib import Path
from typing import Tuple


def _norm_key(s: str) -> str:
    # Match your overlap checker
    return " ".join(s.strip().lower().split())


def dedupe_and_filter_corpus(
    input_file: Path,
    output_file: Path,
    min_len: int = 10,
    max_len: int = 500,
    encoding: str = "utf-8",
) -> Tuple[int, int, int]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read + strip + drop empty
    lines = [
        line.rstrip("\n") for line in input_file.open(encoding=encoding) if line.strip()
    ]
    original_lines = len(lines)

    # Deduplicate by normalized key (preserve first occurrence)
    seen_keys = set()
    unique = []
    for line in lines:
        key = _norm_key(line)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(line)

    unique_lines = len(unique)

    # Length filter (apply to original line content)
    filtered = [line for line in unique if min_len <= len(line) <= max_len]
    filtered_lines = len(filtered)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(filtered) + "\n", encoding=encoding)

    return original_lines, unique_lines, filtered_lines
