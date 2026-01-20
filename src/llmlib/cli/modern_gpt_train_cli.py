# llmlib/cli/modern_gpt_train_cli.py

from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim

from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel
from llmlib.utils.path_util import get_data_split_path
from llmlib.utils.config_util import load_config
from llmlib.utils.checkpoint import save_model
from llmlib.tokenization.registry import load_tokenizer
from llmlib.utils.logger import get_logger
from llmlib.utils.path_util import get_model_dir, resolve_checkpoint_path, short_path
from llmlib.training.lr_schedulers import build_lr_scheduler

logger = get_logger(__name__)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _load_nested_project_config(config_path: Path) -> dict:
    cfg = load_config(caller_file=str(config_path), config_filename=config_path.name)
    for key in ("model_config", "training_config", "project_metadata"):
        if key not in cfg:
            logger.error(f"Missing '{key}' in project_config: {config_path}")
            raise ValueError(f"Missing '{key}' in project_config: {config_path}")
    return cfg


def _select_device(device_arg: str | None) -> str:
    if device_arg in ("cpu", "cuda"):
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# modern-gpt-train
# ---------------------------------------------------------
def modern_gpt_train() -> None:
    """
    Train a Modern GPT model from the command line.

        modern-gpt-train --config /path/to/project_config.json [--device cpu|cuda]
    """
    parser = argparse.ArgumentParser(description="Train a Modern GPT model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to project_config.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    logger.info(f"[modern-gpt-train] Using device: {device}")

    # 1) Load nested config
    cfg       = _load_nested_project_config(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg  = cfg["project_metadata"]

    es_cfg       = train_cfg.get("early_stopping", {})
    es_enabled   = es_cfg.get("enabled", False)
    es_patience  = es_cfg.get("patience_evals", 0)
    es_min_delta = es_cfg.get("min_delta", 0.0)

    no_improve_evals = 0

    ### Normalization
    # --- Normalize vocab size naming ---
    if "vocab_size" in model_cfg and "num_embeddings" in model_cfg:
        if model_cfg["vocab_size"] != model_cfg["num_embeddings"]:
            raise ValueError(
                f"Config mismatch: vocab_size={model_cfg['vocab_size']} "
                f"but num_embeddings={model_cfg['num_embeddings']}"
            )

    vocab_size = model_cfg.get("vocab_size", model_cfg.get("num_embeddings"))
    if vocab_size is None:
        raise ValueError(
            "model_config must define either 'vocab_size' or 'num_embeddings'"
        )

    # Write back normalized form (optional but recommended)
    model_cfg["vocab_size"] = vocab_size

    # 2) Load dataset
    data_path = get_data_split_path(cfg, "data_file")
    logger.info(f"[modern-gpt-train] Using data file: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = load_tokenizer(cfg)  # pass the full project config

    assert model_cfg["vocab_size"] == len(tokenizer.vocab), (
        f"Config vocab_size={model_cfg['vocab_size']} "
        f"but tokenizer vocab={len(tokenizer.vocab)}"
        )

    logger.info(f"[modern-gpt-train] Tokenizer vocab size: {len(tokenizer.vocab)}")

    ## Sanity checks
    print("PAD:", tokenizer.token_to_id("<pad>"))
    print("BOS:", tokenizer.token_to_id("<bos>"))
    print("EOS:", tokenizer.token_to_id("<eos>"))

    encoded_data = [tokenizer.encode(t) for t in text.splitlines() if t]

    if not encoded_data:
        raise ValueError(f"No non-empty lines found in dataset: {data_path}")

    max_seq_len = meta_cfg["max_seq_length"]
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    train_steps = train_cfg["train_steps"]
    eval_interval = train_cfg["eval_interval"]

    # 3) Build ModernGPTConfig + model
    config = ModernGPTConfig(
        vocab_size              = len(tokenizer.vocab),
        d_model                 = model_cfg["d_model"],
        n_heads                 = model_cfg["n_heads"],
        n_layers                = model_cfg["n_layers"],
        max_position_embeddings = max_seq_len,
        dropout                 = model_cfg["dropout"],
    )

    model = ModernGPTModel(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Learning rate scheduler
    scheduler = build_lr_scheduler(
        optimizer,
        base_lr=learning_rate,
        train_steps=train_steps,
        warmup_steps=train_cfg.get("warmup_steps", 0),
        scheduler_cfg=train_cfg.get("lr_scheduler"),
    )


    pad_id = tokenizer.token_to_id("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    sched_cfg = train_cfg.get("lr_scheduler", {"type": "constant"})

    print("\n============================================================")
    print("🚀 modern-gpt-train")
    print("============================================================")
    print(f"Model Name       : {meta_cfg['model_name']}")
    print(f"Vocabulary Size  : {tokenizer.vocab_size}")
    print(f"d_model          : {model_cfg['d_model']}")
    print(f"n_heads          : {model_cfg['n_heads']}")
    print(f"n_layers         : {model_cfg['n_layers']}")
    print(f"dropout          : {model_cfg['dropout']}")
    print(f"max_seq_length   : {max_seq_len}")
    print(f"batch_size       : {batch_size}")
    print(f"learning_rate    : {learning_rate}")
    print(f"train_steps      : {train_steps}")
    print(f"LR Scheduler     : {sched_cfg}")
    print("============================================================\n")

    ## 4) Batching helper
    def make_batch(bs: int) -> torch.Tensor:
        idx = torch.randint(0, len(encoded_data), (bs,))
        batch = [
        [tokenizer.token_to_id("<bos>")] + encoded_data[i][:max_seq_len-2] + [tokenizer.token_to_id("<eos>")]
        for i in idx
        ]
        batch = [torch.tensor(x, dtype=torch.long) for x in batch]

        max_len = max(len(x) for x in batch)
        padded = [
        torch.cat([
            x,
            torch.full((max_len - len(x),), pad_id, dtype=torch.long)
        ])
        for x in batch
        ]

        return torch.stack(padded, dim=0)

    best_val = float("inf")
    best_step = -1
    model_dir = get_model_dir(cfg, create=True)
    best_path = model_dir / "best.pt"   
    last_path = model_dir / "last.pt"

    ### Resume from last checkpoint if exists
    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = resolve_checkpoint_path(cfg, resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {resume_path}")

        logger.info(f"[modern-gpt-train] Resuming weights from: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state)
        print(f"[modern-gpt-train] Loaded resume weights from: {resume_path}")

    # 5) Training loop

    # Encode validation set
    val_path = get_data_split_path(cfg, "val_file")
    if val_path is None:
        raise ValueError("Validation file path is not specified in the config.")

    with val_path.open("r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]
    val_encoded = [tokenizer.encode(l) for l in val_lines]

    # ---- B) Initialize best_val from resumed model (or fresh model) ----
    init_val = compute_val_loss(model, tokenizer, val_encoded, max_seq_len, device)

    best_val = init_val
    best_step = 0
    torch.save(model.state_dict(), best_path)
    print(f"[modern-gpt-train] Initial val={best_val:.4f} saved as best.pt")

    model.train()
    for step in range(1, train_steps+1):
        input_ids = make_batch(batch_size).to(device)
        pad_id = tokenizer.token_to_id("<pad>")
        full_attention_mask = (input_ids != pad_id).long()

        targets = input_ids[:, 1:]
        inputs = input_ids[:, :-1]

        attention_mask = full_attention_mask[:, :-1]
        logits = model(inputs, attention_mask=attention_mask)

        assert inputs.shape == attention_mask.shape

        loss = criterion(logits.reshape(-1, config.vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if step % eval_interval == 0:
            # print(f"Step {step}, Loss: {loss.item():.4f}")
            val_loss = compute_val_loss(model, tokenizer, val_encoded, max_seq_len, device)
            print(f"Step {step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

            # Always save last
            torch.save(model.state_dict(), last_path)

            improved = (best_val - val_loss) > es_min_delta
            if improved:
                best_val = val_loss
                best_step = step
                no_improve_evals = 0
                torch.save(model.state_dict(), best_path)
                print(f"[modern-gpt-train] New best saved at step {step} (val={best_val:.4f})")
            else:
                no_improve_evals += 1

            # Early stopping decision (only checked on eval steps)
            if es_enabled and no_improve_evals >= es_patience:
                print(
                    f"[modern-gpt-train] Early stopping at step {step}. "
                    f"Best val={best_val:.4f} at step {best_step}."
                )
                break
    print(f"[modern-gpt-train] Training complete. Best val={best_val:.4f} at step {best_step}.")

    # 6) Save model
    # Load best weights before final save (canonical model.pt)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        model.eval()
    ckpt_path = save_model(model, cfg)
    print(f"\n[modern-gpt-train] Model saved to: {ckpt_path}")

    # Save tokenizer alongside model
    tokenizer_path = ckpt_path.parent / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)
    print(f"[modern-gpt-train] Tokenizer saved to: {tokenizer_path}")

    cfg["model_config"]["vocab_size"] = len(tokenizer.vocab)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"[modern-gpt-train] Updated config saved to: {config_path}")

@torch.no_grad()
def compute_val_loss(
    model,
    tokenizer,
    val_encoded,
    max_seq_len,
    device):

    model.eval()

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    losses = []

    for seq in val_encoded:
        # Add BOS/EOS + truncate
        ids = [bos_id] + seq[: max_seq_len - 2] + [eos_id]

        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)

        mask = (x != pad_id).long()
        logits = model(x, attention_mask=mask)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
