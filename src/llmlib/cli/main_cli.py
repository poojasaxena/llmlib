#!/usr/bin/env python
"""
Command-line interface for tiny GPT experiments.

Exposes two console scripts (via pyproject.toml):

- tiny-gpt-train  --config path/to/project_config.json
- tiny-gpt-infer  --config path/to/project_config.json [--prompt "hello"]

Both commands expect a nested project_config.json with:
{
  "model_config": {...},
  "training_config": {...},
  "project_metadata": {...}
}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from llmlib.io import (
    load_project_config,
    get_data_file_path,
    save_tiny_model,
    load_tiny_model,
)
from llmlib.tokenization.tokenizer import encode, decode, VOCAB_SIZE
from llmlib.tiny_config import TinyConfig
from llmlib.tiny_model import TinyTransformerModel


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _load_nested_project_config(config_path: Path) -> dict:
    """
    Load a nested project_config.json given its file path.

    We reuse llmlib.io.load_project_config by passing:
    - caller_file   = full path to the config file
    - config_filename = the actual filename
    """
    cfg = load_project_config(
        caller_file=str(config_path),
        config_filename=config_path.name,
    )

    # minimal sanity checks
    for key in ("model_config", "training_config", "project_metadata"):
        if key not in cfg:
            raise ValueError(f"Missing '{key}' in project_config: {config_path}")

    return cfg


def _select_device(device_arg: str | None) -> str:
    if device_arg in ("cpu", "cuda"):
        return device_arg
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# tiny-gpt-train
# ---------------------------------------------------------
def tiny_gpt_train() -> None:
    """
    Train a tiny GPT model from the command line.

        tiny-gpt-train --config /path/to/project_config.json [--device cpu|cuda]

    Uses:
    - project_metadata.data_path / data_file  for training text
    - model_config + VOCAB_SIZE to build TinyConfig
    - training_config for batch_size, learning_rate, train_steps, etc.
    """
    parser = argparse.ArgumentParser(description="Train a tiny GPT model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to project_config.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    print(f"[tiny-gpt-train] Using device: {device}")

    # 1) Load nested config
    cfg = _load_nested_project_config(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg = cfg["project_metadata"]

    # 2) Load dataset
    data_path = get_data_file_path(cfg)
    print(f"[tiny-gpt-train] Using data file: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    encoded_data = [encode(t) for t in text.splitlines() if t]

    if not encoded_data:
        raise ValueError(f"No non-empty lines found in dataset: {data_path}")

    max_seq_len = meta_cfg["max_seq_length"]
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    train_steps = train_cfg["train_steps"]
    eval_interval = train_cfg.get("eval_interval", 50)

    # 3) Build TinyConfig + model
    tiny_config = TinyConfig(
        vocab_size=VOCAB_SIZE,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_position_embeddings=max_seq_len,
        dropout=model_cfg["dropout"],
    )

    model = TinyTransformerModel(tiny_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\n============================================================")
    print("🚀 tiny-gpt-train")
    print("============================================================")
    print(f"Model Name       : {meta_cfg['model_name']}")
    print(f"Vocabulary Size  : {VOCAB_SIZE}")
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
        # random sequences
        idx = torch.randint(0, len(encoded_data), (bs,))
        batch = [encoded_data[i] for i in idx]

        # truncate
        batch = [x[:max_seq_len] for x in batch]

        # pad
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
        loss = criterion(
            logits.view(-1, tiny_config.vocab_size),
            targets.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # 6) Save model
    ckpt_path = save_tiny_model(model, cfg)
    print(f"\n[tiny-gpt-train] Model saved to: {ckpt_path}")


# ---------------------------------------------------------
# tiny-gpt-infer
# ---------------------------------------------------------
def _generate_text(
    model: TinyTransformerModel,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    input_ids = encode(prompt)
    if input_ids.numel() == 0:
        return ""

    input_ids = input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated[-max_seq_len:].unsqueeze(0)

        with torch.no_grad():
            logits = model(context)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated = torch.cat(
            [generated, torch.tensor([next_id], device=device)], dim=0
        )

        next_char = decode(torch.tensor([next_id]))
        if next_char in [".", "?", "!"]:
            break

    return decode(generated.cpu())


def tiny_gpt_infer() -> None:
    """
    Run inference with a saved tiny GPT model.

        tiny-gpt-infer --config /path/to/project_config.json [--prompt "hello"] [--device cpu|cuda]

    If --prompt is omitted, it will ask interactively.
    """
    parser = argparse.ArgumentParser(description="Run inference with a tiny GPT model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to project_config.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to feed the model (if omitted, read from stdin).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference (default: auto)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    print(f"[tiny-gpt-infer] Using device: {device}")

    cfg = _load_nested_project_config(config_path)
    meta_cfg = cfg["project_metadata"]

    max_seq_len = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg.get("max_new_tokens", 40)

    # Load model
    model = load_tiny_model(cfg, device=device, eval_mode=True)

    # Prompt
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt (e.g. 'hello'): ").strip()
        if not prompt:
            prompt = "hello."

    print("\n============================================================")
    print("🧪 tiny-gpt-infer")
    print("============================================================")
    print(f"Model Name     : {meta_cfg['model_name']}")
    print(f"Max Seq Length : {max_seq_len}")
    print(f"Max New Tokens : {max_new_tokens}")
    print(f"Device         : {device}")
    print("============================================================\n")

    output = _generate_text(
        model,
        prompt=prompt,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
    )

    print("---")
    print(f"Prompt : {prompt}")
    print(f"Output : {output}")
