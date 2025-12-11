# tokenization/loader.py
import json
from pathlib import Path
from .char_tokenizer import CharTokenizer
from .char_bpe_tokenizer import CharBPETokenizer
from .byte_bpe_tokenizer import ByteBPETokenizer
from llmlib.utils.tokenizer_io import load_tokenizer_path, get_tokenizer_path


TOKENIZER_REGISTRY = {
    "char": CharTokenizer,
    "char_bpe": CharBPETokenizer,
    "byte_bpe": ByteBPETokenizer,
}


def load_tokenizer(config):
    """
    Load a tokenizer from a JSON file.
    Auto-detects the tokenizer type based on the "model_type" key in the JSON
    """
    tok_path = load_tokenizer_path(config)

    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {tok_path}")

    with tok_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model_type = data.get("model_type")
    cls = TOKENIZER_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown tokenizer type: {model_type}")

    if hasattr(cls, "from_dict"):
        return cls.from_dict(data)
    if hasattr(cls, "load"):
        return cls.load(tok_path)
    raise ValueError(f"Tokenizer class {cls} has no from_dict/load")


def save_tokenizer(tokenizer, project_config: dict):
    tok_path = get_tokenizer_path(project_config)

    if hasattr(tokenizer, "to_dict"):
        data = tokenizer.to_dict()
    else:
        raise ValueError("Tokenizer must implement to_dict()")
    
    with tok_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return tok_path
