## Configuration Utilities for llmlib
# llmlib/src/llmlib/utils/config_util.py

import json
import os
from pathlib import Path

def _meta(cfg: dict) -> dict:
    """Return project_metadata sub-dict if present, otherwise the whole dict."""
    return cfg.get("project_metadata", cfg)


def _train_cfg(cfg: dict) -> dict:
    """Return training_config sub-dict if present, otherwise the whole dict."""
    return cfg.get("training_config", cfg)


def load_config(caller_file: str, config_filename: str = "config.json") -> dict:
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


def validate_nested_config(cfg: dict, config_path: Path | str = "") -> None:
    """
    Validate that a config has the expected nested structure for CLI tools.
    
    Args:
        cfg: The configuration dictionary to validate
        config_path: Optional path for better error messages
    """
    required_keys = ("model_config", "training_config", "project_metadata")
    for key in required_keys:
        if key not in cfg:
            path_info = f" in {config_path}" if config_path else ""
            raise ValueError(f"Missing required key '{key}'{path_info}")


def load_nested_config(caller_file: str, config_filename: str = "config.json") -> dict:
    """
    Load and validate a nested configuration file for CLI tools.
    Combines load_config and validate_nested_config for convenience.
    """
    config_path = Path(caller_file).resolve().parent / config_filename
    cfg = load_config(caller_file, config_filename)
    validate_nested_config(cfg, config_path)
    return cfg

