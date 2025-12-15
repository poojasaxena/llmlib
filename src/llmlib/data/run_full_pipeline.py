#!/usr/bin/env python3
"""
Full Elephant GPT preprocessing pipeline:
raw/ -> clean -> dedupe -> split -> synthetic expansion -> corpus stats
"""

from pathlib import Path
import os
from llmlib.data.prepare_corpus import prepare
from llmlib.data.dedupe_and_split import dedupe_and_filter_corpus
from llmlib.data.split_corpus import split_corpus
from llmlib.data.synth_expand import generate_synthetic_expansions
from llmlib.data.corpus_stats import corpus_statistics

def main():
    DATASET_DIR = os.environ["GLOBAL_DATASETS_DIR"]
    DATA_ROOT = Path(DATASET_DIR) / "llm/mixed_text/"
    RAW = DATA_ROOT / "raw"
    OUT = DATA_ROOT / "out"
    OUT.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Prepare & clean raw corpus
    print("=== Step 1: Prepare raw corpus ===")
    prep_stats = prepare(
        raw_dir=RAW,
        out_dir=OUT,
        file_out="elephant_human_90_10_corpus.txt",
        shuffle=True,
        dedupe=True,
    )

    # Paths
    input_file = OUT / "elephant_human_90_10_corpus.txt"
    dedup_file = OUT / "elephant_human_90_10_corpus.dedup.txt"

    # 2️⃣ Dedupe and filter
    print("=== Step 2: Dedupe and filter corpus ===")
    orig, uniq, filt = dedupe_and_filter_corpus(
        input_file=input_file,
        output_file=dedup_file,
        min_len=10,
        max_len=500,
    )
    print(f"Orig: {orig} | Unique: {uniq} | Filtered: {filt}")

    # 3️⃣ Split corpus into train/val/test
    print("=== Step 3: Split corpus ===")
    stats = split_corpus(
        text_file=dedup_file,
        output_dir=OUT,
        splits=(0.98, 0.01, 0.01),
        seed=42,
    )
    print("Split statistics:")
    for split_name, count in stats.items():
        print(f"{split_name}: {count}")

    # 4️⃣ Generate synthetic expansions
    print("=== Step 4: Generate synthetic expansions ===")
    synthetic_file = OUT / "synthetic_generated.txt"
    count = generate_synthetic_expansions(
        src_file=OUT / "train.txt",
        output_file=synthetic_file,
        max_lines=20_000,
    )
    print(f"Wrote {count} synthetic lines to {synthetic_file}")

    # 5️⃣ Corpus statistics
    print("=== Step 5: Corpus statistics ===")
    corpus_stats = corpus_statistics(OUT)
    for split, stats in corpus_stats.items():
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()
