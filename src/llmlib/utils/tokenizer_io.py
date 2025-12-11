## Tokenizer I/O Utilities
# llmlib/src/llmlib/utils/tokenizer_io.py

from pathlib import Path
from llmlib.utils.config_util import _meta
from llmlib.utils.path_util import get_global_model_dir



def get_tokenizer_path(project_config: dict, create_dir: bool = True) -> Path:
    """
    Return the full absolute path to the tokenizer file defined in:
        project_metadata["tokenizer_save_path"]

    This mirrors get_model_dir() but for tokenizers.
    """
    meta = _meta(project_config)

    if "tokenizer_save_path" not in meta:
        raise KeyError(
            "project_metadata must contain 'tokenizer_save_path' "
            "for tokenizer-related operations."
        )

    rel_path = meta["tokenizer_save_path"]
    base = get_global_model_dir()

    full_path = base / rel_path

    if create_dir:
        full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def load_tokenizer_path(project_config: dict) -> Path:
    """
    Resolve the tokenizer path without creating directories.
    Useful for read-only access in inference or demo scripts.
    """
    return get_tokenizer_path(project_config, create_dir=False)