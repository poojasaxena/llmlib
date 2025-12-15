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
        special_tokens: List[str] | None = None,
        byte_fallback: bool = True,
    ):
        self.special_tokens = special_tokens or []
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.byte_fallback = byte_fallback

        # Build fast lookup table for merges
        self.merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.model_type = "byte_bpe"
    # -------------------------------------------------------------------------
    # Utility — converting raw text <-> initial byte tokens
    # -------------------------------------------------------------------------

    @staticmethod
    def text_to_bytes(text: str | list[str]) -> List[str]:
        if isinstance(text, list):
            text = "\n".join(text)
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
            if tokens[i] == "<wb>" or tokens[i + 1] == "<wb>":
                continue

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
    def train(cls, text: str | list[str], vocab_size: int = 3000, min_freq: int = 2, **kwargs):
        """
        Train a byte-level BPE tokenizer from raw text.
        """
        if isinstance(text, list):
            text = "\n".join(text)
        # 1) Turn into byte tokens
        tokens = []
        for word in re.findall(r"\S+|\s+", text):
            word_bytes = cls.text_to_bytes(word)
            tokens.extend(word_bytes + ["<wb>"])

        # 2) Start with byte-level vocab
        special_tokens = kwargs.get(
            "special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"]
        )

        vocab = {}
        idx = 0

        # 1) Reserve special tokens FIRST
        for tok in special_tokens:
            vocab[tok] = idx
            idx += 1

        # 2) Add byte-level tokens
        # 2) Add ALL byte-level tokens (0x00–0xFF)
        for b in range(256):
            tok = f"<0x{b:02x}>"
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1

        # vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
        merges = []

        # 3) Learn merges (simple greedy BPE loop)
        while len(vocab) < vocab_size:
            pair_counts = cls.get_pair_counts(tokens)
            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < min_freq:
                break

            # Merge it
            tokens = cls.apply_merge(tokens, best_pair)
            merges.append(best_pair)

            # update vocab
            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                vocab[new_token] = idx
                idx += 1


        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            byte_fallback=True,
        )

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
        tokens = self.tokenize_bytes(text)
        ids = []

        if add_special_tokens and "<bos>" in self.vocab:
            ids.append(self.vocab["<bos>"])

        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                ids.append(self.vocab.get("<unk>", 0))

        if add_special_tokens and "<eos>" in self.vocab:
            ids.append(self.vocab["<eos>"])

        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""

        toks = [self.id_to_token.get(i, "") for i in ids]
        specials = set(self.special_tokens)
        toks = [t for t in toks if t not in specials]

        # BPE merges create combined tokens, so we must break back down
        byte_tokens = []
        for tok in toks:
            if tok == "<wb>":
                continue

            # If token looks like byte-level
            if re.fullmatch(r"<0x[0-9a-fA-F]{2}>", tok):
                byte_tokens.append(tok)
            else:
                # Merged token → split into bytes
                byte_tokens.extend(
                    re.findall(r"<0x[0-9a-fA-F]{2}>", tok)
                )

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
