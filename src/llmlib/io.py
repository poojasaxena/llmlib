import json
import os
from pathlib import Path

import torch

from llmlib.tiny_config import TinyConfig
from llmlib.tiny_model import TinyTransformerModel
from llmlib.tokenizer import VOCAB_SIZE as TOKENIZER_VOCAB_SIZE
from llmlib.bpe_tokenizer import BPETokenizer 
from llmlib.modern_gpt import ModernGPTConfig, ModernGPTModel


# ---------------------------------------------------------------------
# 1. Project config loading
# ---------------------------------------------------------------------


def _meta(cfg: dict) -> dict:
    """Return project_metadata sub-dict if present, otherwise the whole dict."""
    return cfg.get("project_metadata", cfg)


def _train_cfg(cfg: dict) -> dict:
    """Return training_config sub-dict if present, otherwise the whole dict."""
    return cfg.get("training_config", cfg)


def load_project_config(caller_file: str, config_filename: str = "config.json") -> dict:
    """
    Load a configuration file located in the same directory as the caller.
    The filename is flexible (default: config.json).
    """
    project_dir = Path(caller_file).resolve().parent
    config_path = project_dir / config_filename

    if not config_path.exists():
        raise FileNotFoundError(f"{config_filename} not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# 2. Global model directory helpers
# ---------------------------------------------------------------------
def get_global_model_dir() -> Path:
    """
    Return the base directory where all models are stored.

    Priority:
    1. $GLOBAL_MODELS_DIR (global, recommended)
    2. Fallback: ~/.llm_models
    """
    env_root = os.environ.get("GLOBAL_MODELS_DIR")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / ".llm_models"


def get_model_dir(project_config: dict, create: bool = True) -> Path:
    """
    Compute the directory where this project's model should be saved.

    Uses:
    - GLOBAL_MODELS_DIR as root
    - project_config["model_save_path"] as optional subfolder
      (e.g. "llm/TinyTransformer_Program")
    - project_config["model_name"] as final folder
    """
    base = get_global_model_dir()  # e.g. ~/.llm_models or whatever you set

    meta = _meta(project_config)
    sub = meta.get("model_save_path", "")
    model_name = meta["model_name"]


    model_dir = base
    if sub:
        model_dir = model_dir / sub
    model_dir = model_dir / model_name

    if create:
        model_dir.mkdir(parents=True, exist_ok=True)

    return model_dir


def get_model_paths(project_config: dict, create_dir: bool = True):
    """
    Convenience helper returning:
    - model_dir : folder for this model
    - ckpt_path : checkpoint file path (model.pt)
    - cfg_path  : saved config path (tiny_config.json)
    """
    model_dir = get_model_dir(project_config, create=create_dir)
    ckpt_path = model_dir / "model.pt"
    cfg_path = model_dir / "tiny_config.json"
    return model_dir, ckpt_path, cfg_path


# ---------------------------------------------------------------------
# 2b. Tokenizer paths (parallel to model paths)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# 3. Global datasets directory helpers
# ---------------------------------------------------------------------
def get_global_datasets_dir() -> Path:
    """
    Return the base directory where all datasets are stored.

    Priority:
    1. $GLOBAL_DATASETS_DIR
    2. Fallback: ~/datasets
    """
    env_root = os.environ.get("GLOBAL_DATASETS_DIR")
    if env_root:
        return Path(env_root).expanduser()
    return Path.home() / "datasets"


def get_data_dir(project_config: dict) -> Path:
    """
    Base directory for this experiment's data.

    Uses:
    - GLOBAL_DATASETS_DIR as root
    - project_config["data_path"] as subfolder (e.g. "llm/greetings")
    """
    base = get_global_datasets_dir()
    meta = _meta(project_config)
    sub = meta.get("data_path", "")
    if sub:
        return base / sub
    return base

def get_data_file_path(project_config: dict) -> Path:
    """
    Full path to the data file defined in project_config["data_file"],
    inside the data directory.
    """
    data_dir = get_data_dir(project_config)
    meta = _meta(project_config)
    file_name = meta["data_file"]
    full_path = data_dir / file_name

    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found at: {full_path}")

    return full_path


# ---------------------------------------------------------------------
# 4. Save / load model
# ---------------------------------------------------------------------
def save_tiny_model(model: TinyTransformerModel, project_config: dict) -> Path:
    """
    Save model weights and a nested config snapshot for reproducibility.

    - model.pt          : weights
    - tiny_config.json  : {
          "model_config": { ... },
          "training_config": { ... },
          "project_metadata": { ... }
      }

    Returns the checkpoint path.
    """
    _, ckpt_path, cfg_path = get_model_paths(project_config)

    # 1) Save model weights
    torch.save(model.state_dict(), ckpt_path)

    # 2) Build config snapshot
    # Assume model.config is a TinyConfig instance
    tiny_cfg: TinyConfig = model.config  # type: ignore[attr-defined]

    model_config = {
        "vocab_size": tiny_cfg.vocab_size,
        "d_model": tiny_cfg.d_model,
        "n_heads": tiny_cfg.n_heads,
        "n_layers": tiny_cfg.n_layers,
        "max_position_embeddings": tiny_cfg.max_position_embeddings,
        "dropout": tiny_cfg.dropout,
    }

    # include only keys that exist in project_config
    training_config = _train_cfg(project_config)
    project_metadata = _meta(project_config)

    full_cfg = {
        "model_config": model_config,
        "training_config": training_config,
        "project_metadata": project_metadata,
    }

    # 3) Save tiny_config.json
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, sort_keys=True)

    return ckpt_path


def load_tiny_model(
    project_config: dict,
    device: str = "cpu",
    eval_mode: bool = False,
) -> TinyTransformerModel:
    """
    Load a TinyTransformerModel from disk using the given project_config.

    Looks for tiny_config.json next to model.pt and supports:
    - NEW format (nested: model_config / training_config / project_metadata)
    - OLD format (flat, like project_config.json)
    """
    _, ckpt_path, cfg_path = get_model_paths(project_config, create_dir=False)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")

    # 1) Load tiny_config.json
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # 2) Extract model_config (support new + old style)
    if "model_config" in cfg_dict:
        # NEW STYLE
        model_cfg_dict = cfg_dict["model_config"]
    else:
        # OLD STYLE FALLBACK:
        # tiny_config.json is just a flat config copy
        model_cfg_dict = cfg_dict

    # 3) Build TinyConfig
    tiny_cfg = TinyConfig(**model_cfg_dict)

    # 4) Build model + load weights
    model = TinyTransformerModel(tiny_cfg)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    if eval_mode:
        model.eval()

    return model


# ---------------------------------------------------------------------
# 5. Load tokenizer (parallel to load_tiny_model)
# ---------------------------------------------------------------------


def load_tokenizer(project_config: dict):
    """
    Load either an old BPETokenizer (Project 4) or a new
    ModernByteBPETokenizer (Project 5). Auto-detects based on JSON keys.
    """
    tok_path = load_tokenizer_path(project_config)

    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at: {tok_path}\n"
            "Run the tokenizer training script first."
        )

    import json
    from llmlib.bpe_tokenizer import BPETokenizer  # old v1
    from llmlib.modern_bpe_tokenizer import ModernByteBPETokenizer  # new v2

    with tok_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Auto-detect tokenizer type ---
    if "model_type" in data and data["model_type"] == "modern_byte_bpe":
        # New Project 5 tokenizer
        return ModernByteBPETokenizer.from_dict(data)
    else:
        # Old Project 4 tokenizer
        return BPETokenizer.from_dict(data)


# ---------------------------------------------------------------------
# 6. Save / load Modern GPT model (v2)
# ---------------------------------------------------------------------


def save_modern_gpt_model(model: ModernGPTModel, project_config: dict) -> Path:
    """
    Save ModernGPTModel weights + config snapshot.

    Files:
    - model.pt
    - modern_gpt_config.json
    """
    model_dir, ckpt_path, _ = get_model_paths(project_config)
    cfg_path = model_dir / "modern_gpt_config.json"

    torch.save(model.state_dict(), ckpt_path)

    cfg = model.config
    model_config = {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "max_position_embeddings": cfg.max_position_embeddings,
        "dropout": cfg.dropout,
    }

    training_config = _train_cfg(project_config)
    project_metadata = _meta(project_config)

    full_cfg = {
        "model_config": model_config,
        "training_config": training_config,
        "project_metadata": project_metadata,
    }

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, sort_keys=True)

    return ckpt_path


def load_modern_gpt_model(
    project_config: dict,
    device: str = "cpu",
    eval_mode: bool = False,
) -> ModernGPTModel:
    """
    Load ModernGPTModel from disk (v2 architecture).
    """
    model_dir, ckpt_path, _ = get_model_paths(project_config, create_dir=False)
    cfg_path = model_dir / "modern_gpt_config.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    model_cfg_dict = cfg_dict["model_config"]
    mg_cfg = ModernGPTConfig(**model_cfg_dict)

    model = ModernGPTModel(mg_cfg)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    if eval_mode:
        model.eval()

    return model


def save_tokenizer(tokenizer, project_config: dict):
    tok_path = get_tokenizer_path(project_config)

    # Detect new vs old class by attribute
    if hasattr(tokenizer, "model_type"):
        # New modern tokenizer
        data = tokenizer.to_dict()
    else:
        # Old tokenizer
        data = tokenizer.to_dict()  # old version already had this method

    with tok_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return tok_path
