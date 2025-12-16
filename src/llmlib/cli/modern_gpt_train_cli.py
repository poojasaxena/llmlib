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
from llmlib.utils.path_util import short_path
from llmlib.utils.logger import get_logger

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
    cfg = _load_nested_project_config(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg = cfg["project_metadata"]

    # 2) Load dataset
    data_path = get_data_split_path(cfg, "data_file")
    logger.info(f"[modern-gpt-train] Using data file: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = load_tokenizer(cfg)  # pass the full project config

    assert model_cfg["num_embeddings"] == len(tokenizer.vocab), (
        f"Config num_embeddings={model_cfg['num_embeddings']} "
        f"but tokenizer vocab={len(tokenizer.vocab)}"
    )

    logger.info(f"[modern-gpt-train] Tokenizer vocab size: {len(tokenizer.vocab)}")

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
        vocab_size=len(tokenizer.vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_position_embeddings=max_seq_len,
        dropout=model_cfg["dropout"],
    )

    model = ModernGPTModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
        pad_id = tokenizer.token_to_id("<pad>") 

        padded = [
        torch.cat([
            x,
            torch.full((max_len - len(x),), pad_id, dtype=torch.long)
        ])
        for x in batch
        ]

        return torch.stack(padded, dim=0)


    # 5) Training loop

    # Encode validation set
    val_path = get_data_split_path(cfg, "val_file")
    if val_path is None:
        raise ValueError("Validation file path is not specified in the config.")

    with val_path.open("r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]
    val_encoded = [tokenizer.encode(l) for l in val_lines]

    model.train()
    for step in range(train_steps):
        input_ids = make_batch(batch_size).to(device)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        logits = model(inputs)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            # print(f"Step {step}, Loss: {loss.item():.4f}")
            val_loss = compute_val_loss(model, tokenizer, val_encoded, max_seq_len, device)
            print(f"Step {step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    # 6) Save model
    ckpt_path = save_model(model, cfg)
    print(f"\n[modern-gpt-train] Model saved to: {ckpt_path}")

    # Save tokenizer alongside model
    tokenizer_path = ckpt_path.parent / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)
    print(f"[modern-gpt-train] Tokenizer saved to: {tokenizer_path}")

    cfg["model_config"]["num_embeddings"] = len(tokenizer.vocab)
    save_model(model, cfg)  # optional: overwrite config with correct vocab


def compute_val_loss(model, tokenizer, val_encoded, max_seq_len, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for line in val_encoded:
            if len(line) < 2:
                continue
            # Truncate to max_seq_len
            line = line[: max_seq_len + 1]  # +1 because we shift

            input_ids = torch.tensor(line[:-1], dtype=torch.long).unsqueeze(0).to(device)
            targets = torch.tensor(line[1:], dtype=torch.long).unsqueeze(0).to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, len(tokenizer.vocab)), targets.view(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)
