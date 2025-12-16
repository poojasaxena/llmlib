## llmlib/tokenization/training/bpe_training.py
from __future__ import annotations
from pathlib import Path
import json
import random
from typing import Iterable, List, Type, Optional, Dict, Tuple

# tokenizer implementations
from llmlib.tokenization.char_bpe_tokenizer import CharBPETokenizer
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer

# utils for IO & saving
from llmlib.tokenization.registry import save_tokenizer
from llmlib.utils.tokenizer_io import get_tokenizer_path
from llmlib.utils.path_util import get_data_file_path

# Registry so we can select by name in config
TOKENIZER_REGISTRY: Dict[str, Type] = {
    "char_bpe": CharBPETokenizer,
    "byte_bpe": ByteBPETokenizer,
}


def read_corpus(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_and_save_tokenizer(
    project_config: dict,
    *,
    tokenizer_type: Optional[str] = None,
    tokenizer_class: Optional[Type] = None,
    vocab_size: Optional[int] = None,
    min_freq: int = 2,
    seed: Optional[int] = None,
    save: bool = True,
    out_path: Optional[Path] = None,
    shuffle_corpus: bool = False,
    extra_train_kwargs: Optional[Dict] = None,
):
    """
    Generic trainer for BPE tokenizers.

    Parameters
    ----------
    project_config : dict
        Project config containing dataset + (optionally) tokenizer_config.
    tokenizer_type : Optional[str]
        Short name from TOKENIZER_REGISTRY (e.g. "char_bpe" or "byte_bpe").
        If provided, overrides tokenizer_class.
    tokenizer_class : Optional[Type]
        A class with `.train(texts, vocab_size, **kwargs)` and `.to_dict()` / `.from_dict()`.
    vocab_size : Optional[int]
        Target vocabulary size. If None, read from project_config["tokenizer_config"]["vocab_size"].
    min_freq : int
        Minimum pair frequency filter (propagated in extra_train_kwargs if tokenizer supports it).
    seed : Optional[int]
        Random seed (for deterministic behaviour if tokenizer uses randomness).
    save : bool
        If True, persist tokenizer JSON to path from project_config (or out_path).
    out_path : Optional[Path]
        If provided, override destination path (useful for tests).
    shuffle_corpus : bool
        Shuffle input lines before training.
    extra_train_kwargs : Optional[Dict]
        Extra kwargs forwarded to tokenizer_class.train(...).

    Returns
    -------
    tokenizer_instance, Path
        The trained tokenizer instance and the path where it was saved (or intended to be saved).
    """
    if seed is not None:
        random.seed(seed)

    # resolve tokenizer class
    if tokenizer_type is not None:
        if tokenizer_type not in TOKENIZER_REGISTRY:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}. Known: {list(TOKENIZER_REGISTRY)}")
        tokenizer_class = TOKENIZER_REGISTRY[tokenizer_type]

    if tokenizer_class is None:
        raise ValueError("tokenizer_class or tokenizer_type must be provided")

    # read corpus
    data_file_path = get_data_file_path(project_config)
    texts = read_corpus(data_file_path)
    if shuffle_corpus:
        random.shuffle(texts)

    # resolve vocab_size
    if vocab_size is None:
        vocab_size = project_config.get("tokenizer_config", {}).get("vocab_size")
    if vocab_size is None:
        raise ValueError("vocab_size must be provided either as arg or in project_config['tokenizer_config']['vocab_size']")

    extra_train_kwargs = extra_train_kwargs or {}
    # ensure min_freq present if tokenizer supports it
    if "min_freq" not in extra_train_kwargs:
        extra_train_kwargs["min_freq"] = min_freq

    # Train
    tokenizer_config = project_config.get("tokenizer_config", {})
    special_tokens = tokenizer_config.get("special_tokens", [])

    # Train
    texts_str = "\n".join(texts)
    tokenizer = tokenizer_class.train(
    texts_str,
    vocab_size=vocab_size,
    special_tokens=special_tokens,  
    **extra_train_kwargs
)
    # Save
    if out_path is not None:
        tok_path = Path(out_path)
    else:
        tok_path = get_tokenizer_path(project_config)

    if save:
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        # use loader.save_tokenizer when possible (centralizes format)
        try:
            save_tokenizer(tokenizer, project_config)
            saved_path = tok_path
        except Exception:
            # fallback: write tokenizer.to_dict()
            with tok_path.open("w", encoding="utf-8") as f:
                json.dump(tokenizer.to_dict(), f, indent=2, ensure_ascii=False)
            saved_path = tok_path
    else:
        saved_path = tok_path

    return tokenizer, saved_path
