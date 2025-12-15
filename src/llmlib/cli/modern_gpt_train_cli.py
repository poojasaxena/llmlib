# llmlib/cli/main_cli.py

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel
from llmlib.utils.path_util import get_data_file_path
from llmlib.utils.config_util import load_config
from llmlib.utils.checkpoint import save_model, load_model
from llmlib.tokenization.char_tokenizer import CharTokenizer


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _load_nested_project_config(config_path: Path) -> dict:
    cfg = load_config(caller_file=str(config_path), config_filename=config_path.name)
    for key in ("model_config", "training_config", "project_metadata"):
        if key not in cfg:
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
    print(f"[modern-gpt-train] Using device: {device}")

    # 1) Load nested config
    cfg = _load_nested_project_config(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg = cfg["project_metadata"]

    # 2) Load dataset
    data_path = get_data_file_path(cfg)
    print(f"[modern-gpt-train] Using data file: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer()
    encoded_data = [tokenizer.encode(t) for t in text.splitlines() if t]

    if not encoded_data:
        raise ValueError(f"No non-empty lines found in dataset: {data_path}")

    max_seq_len = meta_cfg["max_seq_length"]
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    train_steps = train_cfg["train_steps"]
    eval_interval = train_cfg.get("eval_interval", 50)

    # 3) Build ModernGPTConfig + model
    config = ModernGPTConfig(
        vocab_size=tokenizer.vocab_size(),
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
    print(f"Vocabulary Size  : {tokenizer.vocab_size()}")
    print(f"d_model          : {model_cfg['d_model']}")
    print(f"n_heads          : {model_cfg['n_heads']}")
    print(f"n_layers         : {model_cfg['n_layers']}")
    print(f"dropout          : {model_cfg['dropout']}")
    print(f"max_seq_length   : {max_seq_len}")
    print(f"batch_size       : {batch_size}")
    print(f"learning_rate    : {learning_rate}")
    print(f"train_steps      : {train_steps}")
    print("============================================================\n")

    # 4) Batching helper
    def make_batch(bs: int) -> torch.Tensor:
        idx = torch.randint(0, len(encoded_data), (bs,))
        batch = [encoded_data[i] for i in idx]
        batch = [x[:max_seq_len] for x in batch]
        max_len = max(len(x) for x in batch)
        padded = [
            torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
            for x in batch
        ]
        return torch.stack(padded, dim=0)

    # 5) Training loop
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
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # 6) Save model
    ckpt_path = save_model(model, cfg)
    print(f"\n[modern-gpt-train] Model saved to: {ckpt_path}")
