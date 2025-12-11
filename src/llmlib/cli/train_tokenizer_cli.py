#!/usr/bin/env python3
"""
Train & save a BPE tokenizer from the command line.

Usage examples:

# Use project config in current directory (project_config.json)
python -m llmlib.cli.train_tokenizer_cli

# Specify project config path
python -m llmlib.cli.train_tokenizer_cli --config /path/to/project_config.json

# Choose tokenizer type and vocab size (overrides config)
python -m llmlib.cli.train_tokenizer_cli --tokenizer-type byte_bpe --vocab-size 8000

# Dry-run (don't write tokenizer file)
python -m llmlib.cli.train_tokenizer_cli --no-save --vocab-size 200

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from llmlib.tokenization.training.bpe_training import train_and_save_tokenizer
from llmlib.tokenization.training.bpe_training import TOKENIZER_REGISTRY


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Train and save a BPE tokenizer for a project.")
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("project_config.json"),
        help="Path to the project config JSON (default: ./project_config.json)",
    )
    p.add_argument(
        "--tokenizer-type",
        "-t",
        type=str,
        default=None,
        choices=list(TOKENIZER_REGISTRY.keys()),
        help="Tokenizer type name (overrides project config).",
    )
    p.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=None,
        help="Target vocabulary size (overrides project config or tokenizer_config).",
    )
    p.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum pair frequency for merges (passed to tokenizer.train).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic behaviour (optional).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Train but do not write tokenizer file to disk (dry run).",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Optional explicit path to save tokenizer JSON (overrides project config).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle corpus before training (may help some datasets).",
    )
    return p.parse_args(argv)


def load_project_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Project config not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        cfg = load_project_config(args.config)
    except Exception as e:
        print(f"[ERROR] Could not load project config: {e}", file=sys.stderr)
        return 2

    print("[INFO] Project config loaded from:", args.config.resolve())

    # If tokenizer_type not provided, trainer will fall back to config.tokenizer_config.type or require tokenizer_class
    try:
        tokenizer, saved_path = train_and_save_tokenizer(
            cfg,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            seed=args.seed,
            save=not args.no_save,
            out_path=args.out_path,
            shuffle_corpus=args.shuffle,
            extra_train_kwargs={},  # place to pass more params if desired
        )
    except Exception as e:
        print(f"[ERROR] Tokenizer training failed: {e}", file=sys.stderr)
        return 3

    print("[OK] Trained tokenizer")
    if not args.no_save:
        print(f"     Saved to: {saved_path}")
    else:
        print("     (dry-run: not saved)")

    # show some summary info if available
    try:
        info = {"model_type": getattr(tokenizer, "model_type", None)}
        vocab_len = len(getattr(tokenizer, "vocab", {}))
        print(f"     Type: {info['model_type']}, Vocab size: {vocab_len}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
