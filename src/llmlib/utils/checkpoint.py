## Checkpointing Utilities for llmlib
# llmlib/src/llmlib/utils/checkpoint.py


from pathlib import Path
import json
import torch
from typing import Type

from llmlib.utils.path_util import get_model_dir  # your refactored path helper

def save_model(model, project_config: dict, config_filename: str = "model_config.json") -> Path:
    """
    Save any PyTorch model with a nested config snapshot.

    Args:
        model: PyTorch model with `model.config` attributes
        project_config: project metadata dict
        config_filename: name of JSON config file (default: model_config.json)

    Returns:
        Path to saved checkpoint (.pt)
    """

    model_dir = get_model_dir(project_config)
    ckpt_path = model_dir / "model.pt"
    cfg_path = model_dir / config_filename

    # 1) Save weights
    torch.save(model.state_dict(), ckpt_path)

    # 2) Build config snapshot
    cfg = model.config
    if hasattr(cfg, "__dict__"):
        model_config = vars(cfg)
    else:
        # fallback for plain dict config
        model_config = dict(cfg)

    # training + project metadata
    training_config = project_config.get("training_config", {})
    project_metadata = project_config.get("project_metadata", {})

    full_cfg = {
        "model_config": model_config,
        "training_config": training_config,
        "project_metadata": project_metadata,
    }

    # 3) Save config JSON
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(full_cfg, f, indent=2, sort_keys=True)

    return ckpt_path


def load_model(
    model_cls: Type,
    config_cls: Type,
    project_config: dict,
    config_filename: str = "model_config.json",
    device: str = "cpu",
    eval_mode: bool = False,
):
    """
    Load any PyTorch model saved with save_model.

    Args:
        model_cls: PyTorch model class (e.g., TinyTransformerModel, ModernGPTModel)
        config_cls: Model's config class (e.g., TinyConfig, ModernGPTConfig)
        project_config: project metadata dict
        config_filename: name of the JSON config file
        device: "cpu" or "cuda"
        eval_mode: call model.eval() if True

    Returns:
        Loaded model instance on the requested device
    """

    model_dir = get_model_dir(project_config)
    ckpt_path = model_dir / "model.pt"
    cfg_path = model_dir / config_filename

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load JSON config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    # Support old + new format
    if "model_config" in cfg_dict:
        model_cfg_dict = cfg_dict["model_config"]
    else:
        model_cfg_dict = cfg_dict  # fallback for old-style flat configs

    # Build config object
    config_obj = config_cls(**model_cfg_dict)

    # Instantiate model
    model = model_cls(config_obj)

    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    if eval_mode:
        model.eval()

    return model
