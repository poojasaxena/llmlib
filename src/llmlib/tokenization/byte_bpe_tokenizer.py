## llmlib/src/llmlib/tokenization/byte_bpe_tokenizer.py
import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

MODEL_TYPE = "byte_bpe"

class ByteBPETokenizer:
    """
    A clean, modern byte-level BPE tokenizer.

    Key features:
    - Works on raw bytes (0–255), ensuring stable encoding of any text.
    - Learns BPE merges from training text.
    - Provides encode/decode.
    - Provides to_dict()/from_dict() for IO integration.
    - No filesystem operations (handled by llmlib.utils.io).
    """

    model_type = "byte_bpe"

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        byte_fallback: bool = True):
        self.vocab = vocab  # {token: id}
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges  # list of (a, b)
        self.byte_fallback = byte_fallback

        # Build fast lookup table for merges
        self.merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.model_type = "byte_bpe"
    # -------------------------------------------------------------------------
    # Utility — converting raw text <-> initial byte tokens
    # -------------------------------------------------------------------------

    @staticmethod
    def text_to_bytes(text: str) -> List[str]:
        """Convert python string to list of byte tokens ('<0x##>')."""
        return [f"<0x{b:02x}>" for b in text.encode("utf-8")]

    @staticmethod
    def bytes_to_text(byte_tokens: List[str]) -> str:
        """Convert list of '<0x##>' tokens back to string."""
        byte_values = []
        for tok in byte_tokens:
            if tok.startswith("<0x") and tok.endswith(">"):
                try:
                    b = int(tok[3:-1], 16)
                    byte_values.append(b)
                except ValueError:
                    pass
        return bytes(byte_values).decode("utf-8", errors="replace")

    # -------------------------------------------------------------------------
    # BPE training
    # -------------------------------------------------------------------------

    @staticmethod
    def get_pair_counts(tokens: List[str]) -> Counter:
        """Count adjacent token pairs in the token sequence."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    @staticmethod
    def apply_merge(tokens: List[str], merge_pair: Tuple[str, str]) -> List[str]:
        """Apply a merge to a list of tokens."""
        merged = []
        i = 0
        a, b = merge_pair

        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 3000):
        """
        Train a byte-level BPE tokenizer from raw text.
        """

        # 1) Turn into byte tokens
        tokens = cls.text_to_bytes(text)

        # 2) Start with byte-level vocab
        vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
        merges = []

        # 3) Learn merges (simple greedy BPE loop)
        while len(vocab) + 1 < vocab_size:
            pair_counts = cls.get_pair_counts(tokens)
            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            # Merge it
            tokens = cls.apply_merge(tokens, best_pair)
            merges.append(best_pair)

            # update vocab
            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

        return cls(vocab=vocab, merges=merges, byte_fallback=True)

    # -------------------------------------------------------------------------
    # Encoding + Decoding
    # -------------------------------------------------------------------------

    def tokenize_bytes(self, text: str) -> List[str]:
        """Turn input text into byte tokens, then apply merges."""
        tokens = self.text_to_bytes(text)

        # Apply merges in order
        for a, b in self.merges:
            tokens = self.apply_merge(tokens, (a, b))

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.tokenize_bytes(text)
        ids = []

        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            elif self.byte_fallback:
                # fallback to literal byte tokens (should not happen often)
                for b in tok.encode("utf-8"):
                    ids.append(self.vocab.get(f"<0x{b:02x}>", 0))
            else:
                ids.append(0)  # unk fallback

        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        toks = [self.id_to_token.get(i, "") for i in ids]

        # BPE merges create combined tokens, so we must break back down
        byte_tokens = []
        for tok in toks:
            # If token looks like byte-level
            if tok.startswith("<0x") and tok.endswith(">"):
                byte_tokens.append(tok)
            else:
                # This is a merged token — break into bytes
                # by splitting every 4 chars: "<0x??>"
                for i in range(0, len(tok), 6):  # length of "<0x##>"
                    sub = tok[i : i + 6]
                    if sub.startswith("<0x") and sub.endswith(">"):
                        byte_tokens.append(sub)

        return self.bytes_to_text(byte_tokens)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "byte_fallback": self.byte_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            vocab=data["vocab"],
            merges=[tuple(x) for x in data["merges"]],
            byte_fallback=data.get("byte_fallback", True),
        )
