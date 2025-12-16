#!/usr/bin/env python3
"""
Revised Elephant GPT preprocessing pipeline:
raw/ -> clean -> synthetic expansion -> dedupe -> split -> corpus stats
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
        dedupe=False,  # dedupe will happen after synthetic expansion
    )

    # Paths
    clean_file = OUT / "elephant_human_90_10_corpus.txt"
    synthetic_file = OUT / "synthetic_generated.txt"
    combined_file = OUT / "corpus_with_synthetic.txt"
    dedup_file = OUT / "corpus_final_dedup.txt"

    # 2️⃣ Generate synthetic expansions from cleaned corpus
    print("=== Step 2: Generate synthetic expansions ===")
    synth_count = generate_synthetic_expansions(
        src_file=clean_file,
        output_file=synthetic_file,
        max_lines=20_000,
    )
    print(f"Wrote {synth_count} synthetic lines to {synthetic_file}")

    # 3️⃣ Combine cleaned corpus + synthetic data
    print("=== Step 3: Combine human + synthetic corpus ===")
    with combined_file.open("w", encoding="utf-8") as out_f:
        for path in [clean_file, synthetic_file]:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    out_f.write(line)

    # 4️⃣ Dedupe and filter combined corpus
    print("=== Step 4: Dedupe and filter combined corpus ===")
    orig, uniq, filt = dedupe_and_filter_corpus(
        input_file=combined_file,
        output_file=dedup_file,
        min_len=10,
        max_len=500,
    )
    print(f"Orig: {orig} | Unique: {uniq} | Filtered: {filt}")

    # 5️⃣ Split corpus into train/val/test
    print("=== Step 5: Split corpus ===")
    stats = split_corpus(
        text_file=dedup_file,
        output_dir=OUT,
        splits=(0.8, 0.1, 0.1),
        seed=42,
    )
    print("Split statistics:")
    for split_name, count in stats.items():
        print(f"{split_name}: {count}")

    # 6️⃣ Corpus statistics
    print("=== Step 6: Corpus statistics ===")
    corpus_stats = corpus_statistics(OUT)
    for split, stats in corpus_stats.items():
        print(f"{split}: {stats}")

    # 7️⃣ Cleanup unnecessary files
    print("=== Step 7: Cleanup intermediate files ===")
    files_to_keep = {
        OUT / "elephant_human_90_10_corpus.txt",
        OUT / "train.txt",
        OUT / "val.txt",
        OUT / "test.txt",
    }

    for f in OUT.iterdir():
        if f.is_file() and f not in files_to_keep:
            f.unlink()
            print(f"Deleted: {f.name}")

    print("✅ Pipeline finished. Kept train/val/test and original corpus.")


if __name__ == "__main__":
    main()
