## llmlib/test/tokenization/test_bpe_training.py
"""
Unit tests for BPE tokenizer training.

Tests the full pipeline:
config → corpus → training → encode/decode
"""
from pathlib import Path
import tempfile

from llmlib.tokenization.training.bpe_training import train_and_save_tokenizer


def test_train_and_save_tokenizer_byte_bpe(tmp_path: Path):
    # --- create fake corpus ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_file = data_dir / "train.txt"

    train_file.write_text(
        "hello elephants\n" "where do elephants live?\n" "elephants live in africa\n",
        encoding="utf-8",
    )

    # --- minimal config ---
    cfg = {
        "project_metadata": {
            "data_path": str(data_dir),
            "data_file": "train.txt",
            "tokenizer_save_path": str(tmp_path / "tokenizer.json"),
        },
        "tokenizer_config": {
            "type": "byte_bpe",
            "vocab_size": 256,
            "min_freq": 1,
            "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        },
    }

    tokenizer, out_path = train_and_save_tokenizer(
        cfg,
        tokenizer_type="byte_bpe",
        vocab_size=256,
        min_freq=1,
        save=False,  # IMPORTANT: no disk writes in unit tests
    )

    # --- sanity checks ---
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")

    text = "Hello elephants!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert isinstance(ids, list)
    assert isinstance(decoded, str)
    assert len(decoded) > 0
