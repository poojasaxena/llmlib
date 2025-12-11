from pathlib import Path
import json

from llmlib.io import get_tokenizer_path, get_data_file_path
from llmlib.modern_bpe_tokenizer import ModernByteBPETokenizer


def read_corpus(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def train_and_save_tokenizer(config: dict, tokenizer_class=ModernByteBPETokenizer):
    """
    Train a BPE tokenizer on the dataset defined in project config
    and save it to the path defined in project metadata.
    """
    data_file_path = get_data_file_path(config)
    texts = read_corpus(data_file_path)

    # Train tokenizer
    vocab_size = config["tokenizer_config"]["vocab_size"]
    tokenizer = tokenizer_class.train(texts, vocab_size=vocab_size)

    # Save tokenizer
    tok_path = get_tokenizer_path(config)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    with tok_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2, ensure_ascii=False)

    return tokenizer, tok_path
