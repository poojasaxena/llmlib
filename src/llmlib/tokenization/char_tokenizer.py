# llmlib/src/llmlib/tokenization/char_tokenizer.py

import torch
from typing import List

CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?")
VOCAB = {ch: i for i, ch in enumerate(CHARS)}
INV_VOCAB = {i: ch for ch, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
MODEL_TYPE = "char"


class CharTokenizer:
    model_type = "char"

    def __init__(self, vocab=VOCAB, inv_vocab=INV_VOCAB):
        self.vocab = vocab
        self.inv_vocab = inv_vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [
            self.vocab[ch]
            for ch in text.lower()
            if ch in self.vocab
        ]

    def decode(self, ids):
        return "".join(self.inv_vocab[int(i)] for i in ids)

    def to_dict(self):
        return {
            "MODEL_TYPE": self.model_type,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            vocab=data["vocab"],
            inv_vocab={i: ch for ch, i in data["vocab"].items()}
        )
