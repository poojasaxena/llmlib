## Checkpointing Utilities for llmlib
# llmlib/src/llmlib/utils/checkpoint.py


from pathlib import Path
import json
import torch
from typing import Type
from .path_util import get_model_dir
from llmlib.utils.path_util import resolve_checkpoint_path

def save_model(model, 
               project_config: dict, 
               config_filename: str = "model_config.json",
               ckpt_name:str='model.pt') -> Path:
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
    ckpt_path = model_dir / ckpt_name
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
    ckpt_filename: str = "model.pt",
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
    ckpt_path = model_dir / ckpt_filename
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


def resume_checkpoint_if_available(
    *,
    cfg,
    train_cfg,
    model,
    optimizer,
    scheduler,
    device,
    logger,
):
    """
    Resume training state if a checkpoint is provided.

    Supports:
    - full checkpoints (model + optimizer + scheduler)
    - legacy checkpoints (model weights only)

    Returns:
        start_step (int)
        best_val (float)
        best_step (int)
    """
    start_step = 0
    best_val = float("inf")
    best_step = -1

    resume_from = train_cfg.get("resume_from")
    if not resume_from:
        logger.info("[modern-gpt-train] No resume_from specified - starting fresh training")
        return start_step, best_val, best_step

    resume_path = resolve_checkpoint_path(cfg, resume_from)
    if not resume_path.exists():
        raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")

    logger.info(f"[modern-gpt-train] 🔄 RESUMING FROM CHECKPOINT:")
    logger.info(f"[modern-gpt-train]    Config setting: '{resume_from}'")  
    logger.info(f"[modern-gpt-train]    Resolved path: {resume_path}")
    logger.info(f"[modern-gpt-train]    File size: {resume_path.stat().st_size / (1024*1024):.1f} MB")

    ckpt = torch.load(resume_path, map_location=device)

    # ---- New format: full checkpoint ----
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])

        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        best_step = int(ckpt.get("best_step", -1))

        # Optional RNG restore
        if ckpt.get("rng_state") is not None:
            torch.random.set_rng_state(ckpt["rng_state"])
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

        logger.info(
            f"[modern-gpt-train] ✅ FULL CHECKPOINT RESTORED:"
            f"\n                      Step: {start_step}"
            f"\n                      Best validation loss: {best_val:.4f}"
            f"\n                      Best step: {best_step}"
            f"\n                      Optimizer state: {'✅' if ckpt.get('optimizer') else '❌'}"
            f"\n                      Scheduler state: {'✅' if ckpt.get('scheduler') else '❌'}"
            f"\n                      RNG state: {'✅' if ckpt.get('rng_state') else '❌'}"
        )

    # ---- Old format: weights only ----
    else:
        model.load_state_dict(ckpt)
        logger.info("[modern-gpt-train] ⚠️  WEIGHTS-ONLY CHECKPOINT RESTORED:")
        logger.info("[modern-gpt-train]     (No optimizer/scheduler/step info available)")
        logger.info("[modern-gpt-train]     Training will start from step 0 with fresh optimizer")

    return start_step, best_val, best_step


def pack_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    step,
    best_val,
    best_step,
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "best_val": best_val,
        "best_step": best_step,
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
