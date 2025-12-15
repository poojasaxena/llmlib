from pathlib import Path
from typing import Iterable


def corpus_statistics(
    data_root: Path,
    files: Iterable[str] = ("train.txt", "val.txt", "test.txt"),
    encoding: str = "utf-8",
) -> dict[str, dict[str, int]]:
    """
    Compute statistics (lines, characters, words) for corpus files.

    Args:
        data_root: Path to folder containing corpus files
        files: Iterable of filenames to analyze
        encoding: Text encoding for reading files

    Returns:
        Dictionary mapping filename -> stats dict
        e.g., {"train.txt": {"lines": 123, "chars": 4567, "words": 890}}
    """
    stats: dict[str, dict[str, int]] = {}

    for name in files:
        p = data_root / name
        if not p.exists():
            print(f"Warning: {p} does not exist, skipping.")
            continue

        lines = [l.rstrip() for l in p.open(encoding=encoding) if l.strip()]
        char_count = sum(len(l) for l in lines)
        word_count = sum(len(l.split()) for l in lines)

        stats[name] = {
            "lines": len(lines),
            "chars": char_count,
            "words": word_count,
        }

        print(f"{name}: lines {len(lines)}, chars {char_count}, words {word_count}")

    return stats
