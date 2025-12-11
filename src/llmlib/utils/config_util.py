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

