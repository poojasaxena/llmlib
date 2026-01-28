#!/usr/bin/env python3
"""
Data Sanity Check Utilities for LLM Training

Comprehensive data validation tools including:
- Train/validation overlap detection
- Token length statistics and analysis
- Truncation impact assessment
- Data quality warnings and recommendations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from llmlib.utils.path_util import get_data_split_path
from llmlib.utils.config_util import load_config
from llmlib.tokenization.registry import load_tokenizer
from llmlib.utils.logger import get_logger
from llmlib.data.health_check import run_health_check

logger = get_logger(__name__)


def _norm_line(s: str) -> str:
    """Normalize a text line for comparison (lowercase, single spaces)."""
    return " ".join(s.strip().lower().split())


def check_train_val_overlap(train_path: Path, val_path: Path) -> dict:
    """Check for exact text overlap between training and validation sets.
    
    Returns:
        dict: Statistics about the overlap including counts and percentages
    """
    print("\n🔍 Checking train/validation overlap...")
    
    train_lines = [_norm_line(x) for x in train_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    val_lines   = [_norm_line(x) for x in val_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    train_set = set(train_lines)
    val_set   = set(val_lines)
    overlap   = train_set & val_set

    print(f"📊 Train lines: {len(train_lines)} (unique: {len(train_set)})")
    print(f"📊 Val lines: {len(val_lines)} (unique: {len(val_set)})")
    print(f"⚠️  Exact overlap unique lines: {len(overlap)}")
    
    overlap_pct = 0.0
    if len(val_set) > 0:
        overlap_pct = 100.0 * len(overlap) / len(val_set)
        print(f"📈 Overlap % of validation (unique): {overlap_pct:.2f}%")
        
        if overlap_pct > 5.0:  # Warning threshold
            print(f"⚠️  WARNING: High overlap detected ({overlap_pct:.2f}% > 5%)!")
            print("   This may lead to overfitting or inflated validation scores.")
        elif overlap_pct > 0:
            print(f"ℹ️  Minor overlap detected ({overlap_pct:.2f}%)")
        else:
            print("✅ No overlap detected - good data separation!")
    
    return {
        'train_total': len(train_lines),
        'train_unique': len(train_set),
        'val_total': len(val_lines),
        'val_unique': len(val_set),
        'overlap_count': len(overlap),
        'overlap_pct': overlap_pct
    }


def token_length_stats(tokenizer, path: Path, name: str) -> np.ndarray:
    """Analyze token length statistics for a dataset.
    
    Returns:
        np.ndarray: Array of token lengths for each line
    """
    print(f"\n📏 Token length analysis for {name}...")
    
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    lens = np.array([len(tokenizer.encode(x)) for x in lines], dtype=np.int32)

    print(f"📊 {name}: lines={len(lines)}")
    print(f"📊 {name}: token_len mean={lens.mean():.1f}, median={np.median(lens):.1f}, "
          f"p90={np.percentile(lens, 90):.1f}, p99={np.percentile(lens, 99):.1f}, max={lens.max()}")
    
    return lens


def check_truncation_impact(train_lens: np.ndarray, val_lens: np.ndarray, max_seq_len: int) -> dict:
    """Check how many sequences will be truncated based on max_seq_length.
    
    Returns:
        dict: Truncation statistics
    """
    print(f"\n📐 Sequence length configuration: max_seq_length = {max_seq_len}")
    
    # Check truncation impact (-2 for BOS/EOS tokens)
    effective_max = max_seq_len - 2
    train_truncated = np.sum(train_lens >= effective_max)
    val_truncated = np.sum(val_lens >= effective_max)
    
    total_train = len(train_lens)
    total_val = len(val_lens)
    
    train_truncated_pct = 100 * train_truncated / total_train if total_train > 0 else 0
    val_truncated_pct = 100 * val_truncated / total_val if total_val > 0 else 0
    
    print(f"✂️  Training sequences that will be truncated: {train_truncated}/{total_train} ({train_truncated_pct:.1f}%)")
    print(f"✂️  Validation sequences that will be truncated: {val_truncated}/{total_val} ({val_truncated_pct:.1f}%)")
    
    if train_truncated_pct > 10.0:  # > 10% truncation
        print("⚠️  WARNING: High truncation rate in training data!")
        print("   Consider increasing max_seq_length or preprocessing your data.")
    elif train_truncated_pct > 5.0:
        print("ℹ️  Moderate truncation rate detected. Monitor training performance.")
    
    return {
        'max_seq_len': max_seq_len,
        'effective_max': effective_max,
        'train_truncated': train_truncated,
        'train_total': total_train,
        'train_truncated_pct': train_truncated_pct,
        'val_truncated': val_truncated,
        'val_total': total_val,
        'val_truncated_pct': val_truncated_pct
    }


def run_data_sanity_checks(cfg: dict, tokenizer) -> dict:
    """Run comprehensive data sanity checks.
    
    Args:
        cfg: Project configuration dictionary
        tokenizer: Loaded tokenizer instance
        
    Returns:
        dict: Complete results of all sanity checks
    """
    print("\n" + "="*60)
    print("🔬 DATA SANITY CHECKS")
    print("="*60)
    
    # Get data paths
    train_path = get_data_split_path(cfg, "data_file")
    val_path = get_data_split_path(cfg, "val_file")
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Run all checks
    overlap_stats = check_train_val_overlap(train_path, val_path)
    train_lens = token_length_stats(tokenizer, train_path, "TRAIN")
    val_lens = token_length_stats(tokenizer, val_path, "VAL")
    
    max_seq_len = cfg["project_metadata"]["max_seq_length"]
    truncation_stats = check_truncation_impact(train_lens, val_lens, max_seq_len)
    
    print("="*60)
    print("✅ Data sanity checks completed!")
    print("="*60 + "\n")
    
    # Return comprehensive results
    return {
        'overlap': overlap_stats,
        'truncation': truncation_stats,
        'train_token_stats': {
            'mean': float(train_lens.mean()),
            'median': float(np.median(train_lens)),
            'p90': float(np.percentile(train_lens, 90)),
            'p99': float(np.percentile(train_lens, 99)),
            'max': int(train_lens.max())
        },
        'val_token_stats': {
            'mean': float(val_lens.mean()),
            'median': float(np.median(val_lens)),
            'p90': float(np.percentile(val_lens, 90)),
            'p99': float(np.percentile(val_lens, 99)),
            'max': int(val_lens.max())
        }
    }


def _load_nested_project_config(config_path: Path) -> dict:
    """Load and validate project configuration."""
    cfg = load_config(caller_file=str(config_path), config_filename=config_path.name)
    for key in ("model_config", "training_config", "project_metadata"):
        if key not in cfg:
            logger.error(f"Missing '{key}' in project_config: {config_path}")
            raise ValueError(f"Missing '{key}' in project_config: {config_path}")
    return cfg


def main():
    """Main CLI entry point for standalone data sanity checks."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive data sanity checks for LLM training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run data sanity checks
    llmlib-data-checks --config config.json
    
    # Run checks with JSON output
    llmlib-data-checks --config config.json --output results.json
        """
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to project_config.json"
    )
    parser.add_argument(
        "--output", type=str, help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run additional line-level health checks",
    )
    parser.add_argument(
        "--health-min-words",
        default=3,
        type=int,
        help="Health check: threshold for 'very short' line",
    )
    parser.add_argument(
        "--health-prefix-words",
        default=6,
        type=int,
        help="Health check: prefix length (words) for dominance",
    )
    parser.add_argument(
        "--health-topk",
        default=30,
        type=int,
        help="Health check: Top-K items to display",
    )
    parser.add_argument(
        "--health-near-dup",
        action="store_true",
        help="Health check: run near-duplicate detection (simhash)",
    )
    parser.add_argument(
        "--health-near-dup-hamming",
        default=3,
        type=int,
        help="Health check: max hamming distance for near-dup",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config and tokenizer
    cfg = _load_nested_project_config(config_path)
    tokenizer = load_tokenizer(cfg)
    
    # Run data sanity checks
    results = run_data_sanity_checks(cfg, tokenizer)

    # Optional: run line-level health checks
    if args.health_check:
        train_path = get_data_split_path(cfg, "data_file")
        val_path = get_data_split_path(cfg, "val_file")
        run_health_check(
            train_path=train_path,
            val_path=val_path,
            min_words=args.health_min_words,
            prefix_words=args.health_prefix_words,
            topk=args.health_topk,
            near_dup=args.health_near_dup,
            near_dup_hamming=args.health_near_dup_hamming,
        )
    
    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        with output_path.open('w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {output_path}")
    
    print("✅ Data sanity checks completed!")


if __name__ == '__main__':
    main()
