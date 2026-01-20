## Path Utilities for llmlib
# llmlib/src/llmlib/utils/path_util.py

from pathlib import Path
import json
import os
from .config_util import _meta

# ---------------------------------------------------------------------
# 1. Global datasets directory helpers
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


def get_data_split_path(project_config: dict, split: str) -> Path:
    """
    Return full path to a dataset split file (train / val / test).

    split must be one of: 'data_file', 'val_file', 'test_file'
    """
    data_dir = get_data_dir(project_config)
    meta = _meta(project_config)

    if split not in meta:
        raise KeyError(f"'{split}' not found in project_metadata")

    full_path = data_dir / meta[split]

    if not full_path.exists():
        raise FileNotFoundError(f"{split} not found at: {full_path}")

    return full_path


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


def short_path(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)  


# ---------------------------------------------------------------------
# 3. Checkpoint path helpers
# ---------------------------------------------------------------------
def resolve_checkpoint_path(
    project_config: dict,
    ckpt: str,
    default_filename: str = "best.pt",
) -> Path:
    if not ckpt or not str(ckpt).strip():
        raise ValueError("Empty checkpoint path")

    raw = str(ckpt).strip()
    p = Path(raw).expanduser()

    # Allow passing just a directory name like "gpt-bpe-v6"
    if p.suffix == "" and "/" not in raw and "\\" not in raw:
        p = Path(raw) / default_filename

    # 1) Absolute
    if p.is_absolute():
        return p.resolve()

    base = get_global_model_dir()
    meta = _meta(project_config)
    sub = meta.get(
        "model_save_path", ""
    )  # e.g. "llm/language_models/elephantdomain_gpt/"

    tried = []

    # 2) relative to global models dir
    cand1 = (base / p).resolve()
    tried.append(cand1)

    # 3) shorthand under model_save_path
    if sub:
        cand2 = (base / sub / p).resolve()
        tried.append(cand2)

    # 4) relative to current working dir (optional, but can help local experiments)
    cand3 = (Path.cwd() / p).resolve()
    tried.append(cand3)

    for c in tried:
        if c.exists():
            return c

    msg = "resume_from checkpoint not found. Tried:\n" + "\n".join(
        f"  - {t}" for t in tried
    )
    raise FileNotFoundError(msg)
