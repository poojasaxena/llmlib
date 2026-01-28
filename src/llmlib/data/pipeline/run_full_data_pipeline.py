#!/usr/bin/env python3

## llmlib/data/pipeline/run_full_data_pipeline.py
"""
Enhanced Elephant GPT preprocessing pipeline:
raw/ -> clean -> synthetic expansion -> advanced synthetic -> web scraping -> augmentation -> dedupe -> split -> corpus stats
llmlib/data/pipeline/run_full_data_pipeline.py
"""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
from llmlib.data.prepare_corpus import prepare
from llmlib.data.dedupe_and_split import dedupe_and_filter_corpus
from llmlib.data.split_corpus import split_corpus
from llmlib.data.synth_expand import generate_synthetic_expansions
from llmlib.data.corpus_stats import corpus_statistics
from llmlib.tokenization.registry import load_tokenizer


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


def collect_data_files(
    data_dir,
    *,
    include_generated: bool = True,
    base_corpus_name: str = "elephant_human_90_10_corpus.txt",
    data_patterns: list[str] | None = None,
):
    """Collect available data files for combination."""
    files_to_combine = []
    
    # Base corpus file
    base_corpus = data_dir / base_corpus_name
    if base_corpus.exists():
        files_to_combine.append(base_corpus)
        print(f"📄 Found base corpus: {base_corpus.name}")
    
    if not include_generated:
        return files_to_combine

    # Look for various data files in both out/ and raw/ directories
    if data_patterns is None:
        data_patterns = [
            "synthetic_generated.txt",
            "synthetic_advanced*.txt",
            "web_scraped*.txt",
            "augmented*.txt",
            "*synthetic*.txt",
            "*augmented*.txt",
            "*scraped*.txt",
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


def _load_pipeline_config(config_path: Path | None) -> dict:
    if not config_path:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("data_pipeline", {})


def _tuple_or_default(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data pipeline (optionally skip cleanup)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate files at the end of the pipeline",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Disable all data generation (synthetic, web scraping, augmentation)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to project config JSON with data_pipeline section",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    pipeline_cfg = _load_pipeline_config(config_path)

    full_cfg = None
    if config_path:
        with config_path.open("r", encoding="utf-8") as f:
            full_cfg = json.load(f)

    min_len = int(pipeline_cfg.get("min_len", 10))
    max_len = int(pipeline_cfg.get("max_len", 12000))
    max_tokens = int(pipeline_cfg.get("max_tokens", 0))
    max_tokens_use_tokenizer = bool(
        pipeline_cfg.get("max_tokens_use_tokenizer", True)
    )
    drop_short_questions = bool(pipeline_cfg.get("drop_short_questions", False))
    question_max_tokens = int(pipeline_cfg.get("question_max_tokens", 80))
    question_max_chars = int(pipeline_cfg.get("question_max_chars", 200))
    splits = _tuple_or_default(pipeline_cfg.get("splits"), (0.8, 0.1, 0.1))
    seed = int(pipeline_cfg.get("seed", 42))
    max_synth_lines = int(pipeline_cfg.get("max_synth_lines", 20_000))
    advanced_synth_count = int(pipeline_cfg.get("advanced_synth_count", 10_000))
    web_scrape_max_items = int(pipeline_cfg.get("web_scrape_max_items", 1_000))
    drop_urls = bool(pipeline_cfg.get("drop_urls", True))
    strip_discourse = bool(pipeline_cfg.get("strip_discourse", True))
    strip_overview = bool(pipeline_cfg.get("strip_overview", True))
    require_domain = bool(pipeline_cfg.get("require_domain", False))
    prefix_cap_words = int(pipeline_cfg.get("prefix_cap_words", 0))
    if prefix_cap_words == 0:
        prefix_cap_words = int(pipeline_cfg.get("template_cap_words", 0))

    prefix_cap_max = int(pipeline_cfg.get("prefix_cap_max", 0))
    if prefix_cap_max == 0:
        prefix_cap_max = int(pipeline_cfg.get("template_cap_max", 0))

    ngram_cap_n = int(pipeline_cfg.get("ngram_cap_n", 0))
    ngram_cap_max = int(pipeline_cfg.get("ngram_cap_max", 0))

    print(f"🔧 Prefix-cap config: words={prefix_cap_words}, max={prefix_cap_max}")
    data_generation = bool(pipeline_cfg.get("data_generation", True))
    base_corpus_name = pipeline_cfg.get(
        "base_corpus_name", "elephant_human_90_10_corpus.txt"
    )
    synthetic_file_name = pipeline_cfg.get(
        "synthetic_file_name", "synthetic_generated.txt"
    )
    combined_file_name = pipeline_cfg.get(
        "combined_file_name", "corpus_all_sources_combined.txt"
    )
    dedup_file_name = pipeline_cfg.get("dedup_file_name", "corpus_final_dedup.txt")
    data_patterns = pipeline_cfg.get("data_patterns")
    prepare_shuffle = bool(pipeline_cfg.get("prepare_shuffle", True))
    prepare_dedupe = bool(pipeline_cfg.get("prepare_dedupe", False))
    cleanup_keep_files = pipeline_cfg.get(
        "cleanup_keep_files",
        ["corpus_final_dedup.txt", "train.txt", "val.txt", "test.txt"],
    )
    cleanup_generated_on_start = bool(
        pipeline_cfg.get("cleanup_generated_on_start", False)
    )
    cleanup_out_on_start = bool(pipeline_cfg.get("cleanup_out_on_start", False))
    cleanup_raw_generated_on_start = bool(
        pipeline_cfg.get("cleanup_raw_generated_on_start", False)
    )
    cleanup_generated_patterns = pipeline_cfg.get(
        "cleanup_generated_patterns",
        [
            "synthetic_generated.txt",
            "synthetic_advanced*.txt",
            "web_scraped*.txt",
            "augmented*.txt",
            "*synthetic*.txt",
            "*augmented*.txt",
            "*scraped*.txt",
        ],
    )
    cleanup_raw_generated_patterns = pipeline_cfg.get(
        "cleanup_raw_generated_patterns",
        [
            "augmented_corpus.txt",
            "synthetic_advanced.txt",
            "synthetic_advanced*.txt",
            "web_scraped*.txt",
            "augmented*.txt",
            "*synthetic*.txt",
            "*augmented*.txt",
            "*scraped*.txt",
        ],
    )
    key_output_files = pipeline_cfg.get(
        "key_output_files", ["train.txt", "val.txt", "test.txt"]
    )
    dataset_subdir = pipeline_cfg.get("dataset_subdir", "llm/mixed_text")
    raw_subdir = pipeline_cfg.get("raw_subdir", "raw")
    out_subdir = pipeline_cfg.get("out_subdir", "out")

    DATASET_DIR = os.environ["GLOBAL_DATASETS_DIR"]
    DATA_ROOT = Path(DATASET_DIR) / dataset_subdir
    RAW = DATA_ROOT / raw_subdir
    OUT = DATA_ROOT / out_subdir
    OUT.mkdir(parents=True, exist_ok=True)

    print("🐘 Enhanced Elephant LLM Data Pipeline Starting...")
    print(f"📁 Data directory: {OUT}")

    if cleanup_out_on_start:
        print("\n=== Pre-step: Cleanup out/ directory ===")
        removed = 0
        for f in OUT.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
                print(f"🗑️  Deleted: {f.name}")
        if removed == 0:
            print("   No files to remove")

    if cleanup_generated_on_start:
        print("\n=== Pre-step: Cleanup old generated files ===")
        removed = 0
        for pattern in cleanup_generated_patterns:
            for f in OUT.glob(pattern):
                if f.is_file():
                    f.unlink()
                    removed += 1
                    print(f"🗑️  Deleted: {f.name}")
        if removed == 0:
            print("   No generated files to remove")

    if cleanup_raw_generated_on_start:
        print("\n=== Pre-step: Cleanup generated files in raw/ ===")
        removed = 0
        for pattern in cleanup_raw_generated_patterns:
            for f in RAW.glob(pattern):
                if f.is_file():
                    f.unlink()
                    removed += 1
                    print(f"🗑️  Deleted: {f.name}")
        if removed == 0:
            print("   No generated files to remove")

    # 1️⃣ Prepare & clean raw corpus
    print("\n=== Step 1: Prepare raw corpus ===")
    prep_stats = prepare(
        raw_dir=RAW,
        out_dir=OUT,
        file_out=base_corpus_name,
        shuffle=prepare_shuffle,
        dedupe=prepare_dedupe,
    )

    # Paths
    clean_file = OUT / base_corpus_name
    synthetic_file = OUT / synthetic_file_name
    
    generation_enabled = data_generation and (not args.no_generation)

    if not generation_enabled:
        print("\n=== Step 2: Data generation disabled (skipping) ===")
        print("=== Step 3: Data generation disabled (skipping) ===")
    else:
        # 2️⃣ Generate synthetic expansions from cleaned corpus
        print("\n=== Step 2: Generate basic synthetic expansions ===")
        if clean_file.exists():
            synth_count = generate_synthetic_expansions(
                src_file=clean_file,
                output_file=synthetic_file,
                max_lines=max_synth_lines,
            )
            print(f"Wrote {synth_count} synthetic lines to {synthetic_file}")
        else:
            print(f"⚠️  Base corpus not found at {clean_file}")

        # 3️⃣ Run advanced data generation scripts
        print("\n=== Step 3: Advanced data generation ===")

        # Run advanced synthetic expansion
        run_script_if_exists(
            "advanced_synth_expand.py", OUT, str(clean_file), str(advanced_synth_count)
        )

        # Run web scraping
        run_script_if_exists(
            "web_scrape_elephants.py", OUT, str(web_scrape_max_items)
        )

        # Run data augmentation on existing corpus
        run_script_if_exists("augment_corpus.py", OUT, str(clean_file))

    # 4️⃣ Collect and combine ALL available data files
    print("\n=== Step 4: Combine all available data sources ===")
    all_data_files = collect_data_files(
        OUT,
        include_generated=generation_enabled,
        base_corpus_name=base_corpus_name,
        data_patterns=data_patterns,
    )
    
    if not all_data_files:
        print("❌ No data files found to combine!")
        return
        
    print(f"📊 Combining {len(all_data_files)} data sources:")
    for i, file in enumerate(all_data_files, 1):
        lines = sum(1 for _ in file.open('r', encoding='utf-8'))
        size_mb = file.stat().st_size / (1024*1024)
        print(f"  {i}. {file.name} ({lines:,} lines, {size_mb:.1f}MB)")

    combined_file = OUT / combined_file_name
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
    dedup_file = OUT / dedup_file_name
    
    token_counter = None
    if max_tokens > 0 and max_tokens_use_tokenizer and full_cfg is not None:
        try:
            tokenizer = load_tokenizer(full_cfg)
            token_counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
            print("🔧 Token-cap uses tokenizer encode()")
        except Exception as e:
            print(f"⚠️  Tokenizer load failed for token-cap: {e}")

    orig, uniq, filt = dedupe_and_filter_corpus(
        input_file=combined_file,
        output_file=dedup_file,
        min_len=min_len,
        max_len=max_len,
        max_tokens=max_tokens,
        token_counter=token_counter,
        drop_short_questions=drop_short_questions,
        question_max_tokens=question_max_tokens,
        question_max_chars=question_max_chars,
        drop_urls=drop_urls,
        strip_discourse=strip_discourse,
        strip_overview=strip_overview,
        require_domain=require_domain,
        prefix_cap_words=prefix_cap_words,
        prefix_cap_max=prefix_cap_max,
        ngram_cap_n=ngram_cap_n,
        ngram_cap_max=ngram_cap_max,
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
        splits=splits,
        seed=seed,
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
    if args.cleanup:
        print("\n=== Step 8: Cleanup intermediate files ===")
        files_to_keep = {OUT / name for name in cleanup_keep_files}

        cleanup_count = 0
        for f in OUT.iterdir():
            if f.is_file() and f not in files_to_keep:
                f.unlink()
                print(f"🗑️  Deleted: {f.name}")
                cleanup_count += 1
        
        if cleanup_count == 0:
            print("   No intermediate files to clean up")
    else:
        print("\n=== Step 8: Cleanup skipped (use --cleanup to enable) ===")
    
    print("\n🎉 Enhanced pipeline completed successfully!")
    print(f"📁 Final files available in: {OUT}")
    print("📊 Key outputs:")
    for key_file in key_output_files:
        file_path = OUT / key_file
        if file_path.exists():
            lines = sum(1 for _ in file_path.open('r', encoding='utf-8'))
            size_mb = file_path.stat().st_size / (1024*1024)
            print(f"   {key_file}: {lines:,} lines ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
