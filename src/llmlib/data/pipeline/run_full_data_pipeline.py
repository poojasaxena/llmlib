#!/usr/bin/env python3

## llmlib/data/pipeline/run_full_data_pipeline.py
"""
Enhanced Elephant GPT preprocessing pipeline:
raw/ -> clean -> synthetic expansion -> advanced synthetic -> web scraping -> augmentation -> dedupe -> split -> corpus stats
llmlib/data/pipeline/run_full_data_pipeline.py
"""

from pathlib import Path
import os
import subprocess
import sys
from llmlib.data.prepare_corpus import prepare
from llmlib.data.dedupe_and_split import dedupe_and_filter_corpus
from llmlib.data.split_corpus import split_corpus
from llmlib.data.synth_expand import generate_synthetic_expansions
from llmlib.data.corpus_stats import corpus_statistics


def run_script_if_exists(script_name, data_dir, *args):
    """Run a data generation script if it exists."""
    script_path = Path(__file__).parent.parent / script_name
    if script_path.exists():
        print(f"Running {script_name}...")
        try:
            result = subprocess.run([
                sys.executable, str(script_path), str(data_dir), *args
            ], check=True, capture_output=True, text=True)
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {script_name} failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            return False
    else:
        print(f"📝 {script_name} not found, skipping...")
        return False


def collect_data_files(data_dir):
    """Collect all available data files for combination."""
    files_to_combine = []
    
    # Base corpus file
    base_corpus = data_dir / "elephant_human_90_10_corpus.txt"
    if base_corpus.exists():
        files_to_combine.append(base_corpus)
        print(f"📄 Found base corpus: {base_corpus.name}")
    
    # Look for various data files in both out/ and raw/ directories
    data_patterns = [
        "synthetic_generated.txt",
        "advanced_synthetic*.txt", 
        "web_scraped*.txt",
        "augmented*.txt",
        "*synthetic*.txt",
        "*augmented*.txt", 
        "*scraped*.txt"
    ]
    
    # Check output directory
    for pattern in data_patterns:
        matching_files = list(data_dir.glob(pattern))
        for file in matching_files:
            if file not in files_to_combine and file.is_file() and file.stat().st_size > 0:
                files_to_combine.append(file)
                print(f"📄 Found additional data: {file.name}")
    
    # Also check raw directory for generated files
    raw_dir = data_dir.parent / "raw"
    if raw_dir.exists():
        for pattern in data_patterns:
            matching_files = list(raw_dir.glob(pattern))
            for file in matching_files:
                if file not in files_to_combine and file.is_file() and file.stat().st_size > 0:
                    files_to_combine.append(file)
                    print(f"📄 Found additional data in raw/: {file.name}")
    
    return files_to_combine


def main():
    DATASET_DIR = os.environ["GLOBAL_DATASETS_DIR"]
    DATA_ROOT = Path(DATASET_DIR) / "llm/mixed_text/"
    RAW = DATA_ROOT / "raw"
    OUT = DATA_ROOT / "out"
    OUT.mkdir(parents=True, exist_ok=True)

    print("🐘 Enhanced Elephant LLM Data Pipeline Starting...")
    print(f"📁 Data directory: {OUT}")

    # 1️⃣ Prepare & clean raw corpus
    print("\n=== Step 1: Prepare raw corpus ===")
    prep_stats = prepare(
        raw_dir=RAW,
        out_dir=OUT,
        file_out="elephant_human_90_10_corpus.txt",
        shuffle=True,
        dedupe=False,  # dedupe will happen after all data is combined
    )

    # Paths
    clean_file = OUT / "elephant_human_90_10_corpus.txt"
    synthetic_file = OUT / "synthetic_generated.txt"
    
    # 2️⃣ Generate synthetic expansions from cleaned corpus
    print("\n=== Step 2: Generate basic synthetic expansions ===")
    if clean_file.exists():
        synth_count = generate_synthetic_expansions(
            src_file=clean_file,
            output_file=synthetic_file,
            max_lines=20_000,
        )
        print(f"Wrote {synth_count} synthetic lines to {synthetic_file}")
    else:
        print(f"⚠️  Base corpus not found at {clean_file}")

    # 3️⃣ Run advanced data generation scripts
    print("\n=== Step 3: Advanced data generation ===")
    
    # Run advanced synthetic expansion
    run_script_if_exists("advanced_synth_expand.py", OUT, str(clean_file), "10000")
    
    # Run web scraping
    run_script_if_exists("web_scrape_elephants.py", OUT, "1000")
    
    # Run data augmentation on existing corpus
    run_script_if_exists("augment_corpus.py", OUT, str(clean_file))

    # 4️⃣ Collect and combine ALL available data files
    print("\n=== Step 4: Combine all available data sources ===")
    all_data_files = collect_data_files(OUT)
    
    if not all_data_files:
        print("❌ No data files found to combine!")
        return
        
    print(f"📊 Combining {len(all_data_files)} data sources:")
    for i, file in enumerate(all_data_files, 1):
        lines = sum(1 for _ in file.open('r', encoding='utf-8'))
        size_mb = file.stat().st_size / (1024*1024)
        print(f"  {i}. {file.name} ({lines:,} lines, {size_mb:.1f}MB)")

    combined_file = OUT / "corpus_all_sources_combined.txt"
    total_lines = 0
    
    with combined_file.open("w", encoding="utf-8") as out_f:
        for file_path in all_data_files:
            print(f"📖 Processing {file_path.name}...")
            file_lines = 0
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Only write non-empty lines
                            out_f.write(line + "\n")
                            file_lines += 1
                            total_lines += 1
                print(f"   Added {file_lines:,} lines")
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {e}")
    
    print(f"✅ Combined corpus written: {total_lines:,} total lines")

    # 5️⃣ Dedupe and filter combined corpus
    print("\n=== Step 5: Dedupe and filter combined corpus ===")
    dedup_file = OUT / "corpus_final_dedup.txt"
    
    orig, uniq, filt = dedupe_and_filter_corpus(
        input_file=combined_file,
        output_file=dedup_file,
        min_len=10,
        max_len=500,
    )
    print(f"📊 Deduplication results:")
    print(f"   Original lines: {orig:,}")
    print(f"   Unique lines: {uniq:,}")
    print(f"   Final filtered: {filt:,}")
    print(f"   Reduction: {((orig-filt)/orig*100):.1f}%")

    # 6️⃣ Split corpus into train/val/test
    print("\n=== Step 6: Split corpus ===")
    stats = split_corpus(
        text_file=dedup_file,
        output_dir=OUT,
        splits=(0.8, 0.1, 0.1),
        seed=42,
    )
    print("📊 Split statistics:")
    for split_name, split_data in stats.items():
        if isinstance(split_data, dict) and 'lines' in split_data:
            print(f"   {split_name}: {split_data['lines']:,} lines")
        elif isinstance(split_data, (int, float)):
            print(f"   {split_name}: {split_data:,} lines")
        else:
            print(f"   {split_name}: {split_data}")

    # 7️⃣ Corpus statistics
    print("\n=== Step 7: Final corpus statistics ===")
    corpus_stats = corpus_statistics(OUT)
    for split, stats in corpus_stats.items():
        print(f"{split}: {stats}")

    # 8️⃣ Cleanup intermediate files (keep only essentials)
    print("\n=== Step 8: Cleanup intermediate files ===")
    files_to_keep = {
        OUT / "corpus_final_dedup.txt",            # Complete final corpus
        OUT / "train.txt",                         # Training set
        OUT / "val.txt",                          # Validation set
        OUT / "test.txt",                         # Test set
    }

    cleanup_count = 0
    for f in OUT.iterdir():
        if f.is_file() and f not in files_to_keep:
            f.unlink()
            print(f"🗑️  Deleted: {f.name}")
            cleanup_count += 1
    
    if cleanup_count == 0:
        print("   No intermediate files to clean up")
    
    print("\n🎉 Enhanced pipeline completed successfully!")
    print(f"📁 Final files available in: {OUT}")
    print("📊 Key outputs:")
    for key_file in ["train.txt", "val.txt", "test.txt"]:
        file_path = OUT / key_file
        if file_path.exists():
            lines = sum(1 for _ in file_path.open('r', encoding='utf-8'))
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   {key_file}: {lines:,} lines ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
