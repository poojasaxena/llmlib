# llmlib/src/llmlib/tokenization/char_bpe_tokenizer.py

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import json
from pathlib import Path
from typing import Union

MODEL_TYPE = "char_bpe"

@dataclass
class CharBPETokenizerConfig:
    """
    Simple config for the toy BPE tokenizer.
    """

    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    word_end: str = "</w>"


class CharBPETokenizer:
    """
    Very small educational BPE tokenizer:
      - trains from raw text lines
      - learns merges up to a target vocab size
      - can encode/decode, and save/load to JSON

    This is intentionally simple and slow – good for understanding.
    """
    model_type = "char_bpe"

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        config: CharBPETokenizerConfig | None = None,
    ):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        # lower rank = earlier merge = higher priority
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.config = config or CharBPETokenizerConfig()
        self.model_type = "char_bpe"

    # -------- TRAINING --------

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int,
        config: CharBPETokenizerConfig | None = None,
        min_freq: int = 2) -> "CharBPETokenizer":
        """
        Train a BPE tokenizer on a list of text lines.

        - Start from characters + special tokens.
        - Repeatedly find the most frequent pair and merge.
        - Stop when vocab_size reached or pair frequency < min_freq.
        """
        config = config or CharBPETokenizerConfig()

        # 1) Split into words and represent as char sequences + word_end marker
        corpus_words: List[List[str]] = []
        for text in texts:
            for word in text.strip().split():
                if not word:
                    continue
                symbols = list(word) + [config.word_end]
                corpus_words.append(symbols)

        # 2) Initial vocab: special tokens + all characters + word_end
        vocab_tokens = {
            config.unk_token,
            config.pad_token,
            config.bos_token,
            config.eos_token,
        }
        for w in corpus_words:
            vocab_tokens.update(w)

        merges: List[Tuple[str, str]] = []

        def get_pair_freqs(words: List[List[str]]) -> Counter[Tuple[str, str]]:
            freqs: Counter[Tuple[str, str]] = Counter()
            for w in words:
                if len(w) < 2:
                    continue
                for a, b in zip(w, w[1:]):
                    freqs[(a, b)] += 1
            return freqs

        # 3) Iteratively learn merges
        while len(vocab_tokens) < vocab_size:
            pair_freqs = get_pair_freqs(corpus_words)
            if not pair_freqs:
                break

            (best_pair, best_freq) = pair_freqs.most_common(1)[0]
            if best_freq < min_freq:
                # no pair appears often enough, stop
                break

            # New symbol = just string concatenation of the pair
            new_symbol = "".join(best_pair)

            merges.append(best_pair)
            vocab_tokens.add(new_symbol)

            # 4) Apply the merge to all words in the corpus
            new_corpus_words: List[List[str]] = []
            for w in corpus_words:
                if len(w) < 2:
                    new_corpus_words.append(w)
                    continue

                merged: List[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                        merged.append(new_symbol)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                new_corpus_words.append(merged)

            corpus_words = new_corpus_words

        # 5) Build final vocab mapping
        vocab_list = sorted(vocab_tokens)
        vocab: Dict[str, int] = {tok: i for i, tok in enumerate(vocab_list)}

        return cls(vocab=vocab, merges=merges, config=config)

    # -------- ENCODE / DECODE --------

    def _bpe_tokenize_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single word.
        """
        symbols = list(word) + [self.config.word_end]
        if not self.merges:
            return symbols

        # Greedy pair-merging based on merge_ranks
        while True:
            pair_indices = defaultdict(list)  # pair -> list of positions
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merge_ranks:
                    pair_indices[pair].append(i)

            if not pair_indices:
                break

            # Choose pair with smallest rank (highest priority)
            best_pair = min(pair_indices.keys(), key=lambda p: self.merge_ranks[p])
            new_symbol = "".join(best_pair)

            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to a list of token IDs.
        """
        tokens: List[str] = []

        if add_special_tokens and self.config.bos_token in self.vocab:
            tokens.append(self.config.bos_token)

        for word in text.strip().split():
            word_tokens = self._bpe_tokenize_word(word)
            tokens.extend(word_tokens)

        if add_special_tokens and self.config.eos_token in self.vocab:
            tokens.append(self.config.eos_token)

        ids = [
            self.vocab.get(tok, self.vocab.get(self.config.unk_token, 0))
            for tok in tokens
        ]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of IDs back to a (rough) text string.
        """
        tokens = [self.id_to_token.get(i, self.config.unk_token) for i in ids]

        words: List[str] = []
        current_word = ""

        for tok in tokens:
            # Drop special tokens if requested
            if skip_special_tokens and tok in {
                self.config.unk_token,
                self.config.pad_token,
                self.config.bos_token,
                self.config.eos_token,
            }:
                continue

            # Word-end marker logic
            if tok.endswith(self.config.word_end):
                # Could be merged like "hello</w>"
                piece = tok[: -len(self.config.word_end)]
                current_word += piece
                words.append(current_word)
                current_word = ""
            elif tok == self.config.word_end:
                words.append(current_word)
                current_word = ""
            else:
                # Regular subword
                if tok.endswith(self.config.word_end):
                    piece = tok[: -len(self.config.word_end)]
                else:
                    piece = tok
                current_word += piece

        if current_word:
            words.append(current_word)

        return " ".join(words)

    # -------- SERIALIZATION --------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save tokenizer config, vocab and merges to a JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        obj = {
            "vocab": self.vocab,
            "merges": self.merges,
            "config": asdict(self.config),
            "model_type": self.model_type,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharBPETokenizer":
        """
        Load tokenizer from a JSON file created by `.save`.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        return cls.from_dict(obj)


    def to_dict(self) -> dict:
        """
        Standard serializable dict for this tokenizer.
        Keeps merges as lists for JSON friendliness.
        """
        return {
            "model_type": self.model_type,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "config": asdict(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharBPETokenizer":
        """
        Construct from a dict produced by to_dict().
        """
        config = CharBPETokenizerConfig(**data.get("config", {}))
        merges = [tuple(pair) for pair in data.get("merges", [])]
        return cls(vocab=data["vocab"], merges=merges, config=config)
