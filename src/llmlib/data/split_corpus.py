from pathlib import Path
import random
from typing import Tuple, Dict


def split_corpus(
    text_file: Path,
    output_dir: Path,
    splits: Tuple[float, float, float] = (0.98, 0.01, 0.01),
    seed: int = 42,
    encoding: str = "utf-8",
) -> Dict[str, Dict[str, int]]:
    """
    Split a corpus into train/val/test sets.

    Steps:
    - Read and clean lines
    - Shuffle deterministically
    - Split according to ratios
    - Write train.txt / val.txt / test.txt
    - Compute basic statistics

    Args:
        text_file: Path to cleaned corpus (one sample per line)
        output_dir: Directory to write train/val/test files
        splits: (train, val, test) ratios — must sum to 1.0
        seed: RNG seed for reproducibility
        encoding: Text encoding

    Returns:
        Dictionary with stats per split:
        {
            "train": {"lines": int, "chars": int, "words": int},
            "val":   {...},
            "test":  {...}
        }
    """

    if not text_file.exists():
        raise FileNotFoundError(f"Corpus not found: {text_file}")

    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {splits}")

    # Read corpus
    lines = [
        line.rstrip() for line in text_file.open(encoding=encoding) if line.strip()
    ]

    if len(lines) < 3:
        raise ValueError("Corpus too small to split")

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    n_val = max(1, int(n * splits[1]))
    n_test = max(1, int(n * splits[2]))
    n_train = n - n_val - n_test

    train = lines[:n_train]
    val = lines[n_train : n_train + n_val]
    test = lines[n_train + n_val :]

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / "train.txt",
        "val": output_dir / "val.txt",
        "test": output_dir / "test.txt",
    }

    for split, path in paths.items():
        data = {"train": train, "val": val, "test": test}[split]
        path.write_text("\n".join(data) + "\n", encoding=encoding)

    # Stats
    def stats(lines):
        return {
            "lines": len(lines),
            "chars": sum(len(l) for l in lines),
            "words": sum(len(l.split()) for l in lines),
        }

    return {
        "train": stats(train),
        "val": stats(val),
        "test": stats(test),
    }
