# tiny_transformer_lab/tokenizer.py

import torch

# Simple character-level tokenizer for tiny experiments
CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?")
VOCAB = {ch: i for i, ch in enumerate(CHARS)}
INV_VOCAB = {i: ch for ch, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def encode(text: str) -> torch.Tensor:
    """Convert text to a tensor of token ids."""
    return torch.tensor(
        [VOCAB[ch] for ch in text.lower() if ch in VOCAB],
        dtype=torch.long,
    )


def decode(ids: torch.Tensor) -> str:
    """Convert a tensor of token ids back to text."""
    return "".join(INV_VOCAB[int(i)] for i in ids)
