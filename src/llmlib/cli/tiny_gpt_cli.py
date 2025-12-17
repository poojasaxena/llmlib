#!/usr/bin/env python
"""
Command-line interface for Modern GPT experiments using Byte-level BPE tokenizer.

Exposes console scripts via pyproject.toml:

- modern-gpt-train  --config /path/to/project_config.json
- modern-gpt-infer  --config /path/to/project_config.json [--prompt "hello"]

Both commands expect a nested project_config.json:
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

from llmlib.utils.path_util import get_data_file_path
from llmlib.utils.config_util import load_config
from llmlib.utils.checkpoint import save_model, load_model
from llmlib.tokenization.byte_bpe_tokenizer import ByteBPETokenizer
from llmlib.modeling.modern_gpt import ModernGPTConfig, ModernGPTModel


# -----------------------
# Utilities
# -----------------------
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


# -----------------------
# modern-gpt-train
# -----------------------
def tiny_gpt_train() -> None:
    parser = argparse.ArgumentParser(description="Train a Tiny GPT model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to project_config.json"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "auto"], default="auto"
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    device = _select_device(args.device)
    print(f"[modern-gpt-train] Using device: {device}")

    cfg = _load_nested_project_config(config_path)
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg = cfg["project_metadata"]

    # Load raw data
    data_path = get_data_file_path(cfg)
    print(f"[modern-gpt-train] Using data file: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        text = f.read()

    tokenizer_cfg = cfg["tokenizer_config"]
    if tokenizer_cfg is None:
        raise ValueError("Missing 'tokenizer_config' in project_config.json")

    # Train Byte BPE tokenizer on train set only
    tokenizer = ByteBPETokenizer.train(
        text=text,
        vocab_size=tokenizer_cfg["vocab_size"],
        min_freq=tokenizer_cfg.get("min_freq", 2),
        special_tokens=tokenizer_cfg.get("special_tokens"),
    )
    print(f"[modern-gpt-train] Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")

    assert tokenizer.vocab_size == tokenizer_cfg["vocab_size"], (
        f"Tokenizer vocab mismatch: "
        f"{tokenizer.vocab_size} != {tokenizer_cfg['vocab_size']}"
    )

    # Encode full dataset for simplicity
    encoded_data = tokenizer.encode(text)

    max_seq_len = meta_cfg["max_seq_length"]
    batch_size = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    train_steps = train_cfg["train_steps"]
    eval_interval = train_cfg.get("eval_interval", 50)

    # Build model
    modern_cfg = ModernGPTConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_position_embeddings=max_seq_len,
        dropout=model_cfg["dropout"],
    )
    model = ModernGPTModel(modern_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\n[modern-gpt-train] Starting training...\n")

    # Simple batching helper
    def make_batch(bs: int):
        idx = torch.randint(0, len(encoded_data) - max_seq_len, (bs,))
        batch = [encoded_data[i : i + max_seq_len] for i in idx]
        return torch.tensor(batch, dtype=torch.long)

    # Training loop
    model.train()
    for step in range(train_steps):
        input_ids = make_batch(batch_size).to(device)
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        logits = model(inputs)
        loss = criterion(logits.view(-1, len(tokenizer.vocab)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # Save model + tokenizer
    save_path = save_model(model, cfg)
    tokenizer_path = save_path.parent / "tokenizer.json"
    with tokenizer_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(tokenizer.to_dict(), f)
    print(f"[modern-gpt-train] Model + tokenizer saved at: {save_path.parent}")


# -----------------------
# modern-gpt-infer
# -----------------------
def _generate_text(
    model: ModernGPTModel,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int) -> str:
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    generated = input_ids.copy()

    for _ in range(max_new_tokens):
        context = torch.tensor(
            [generated[-max_seq_len:]], dtype=torch.long, device=device
        )
        with torch.no_grad():
            logits = model(context)
            next_logits = logits[0, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        next_char = tokenizer.decode([next_id])
        if next_char in [".", "!", "?"]:
            break
    return tokenizer.decode(generated)


def tiny_gpt_infer() -> None:
    
    parser = argparse.ArgumentParser(
        description="Run inference with a Tiny GPT model."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "auto"], default="auto"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prompt mode (REPL).",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Resolve config + device
    # --------------------------------------------------
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = _select_device(args.device)
    cfg = _load_nested_project_config(config_path)

    meta_cfg       = cfg["project_metadata"]
    max_seq_len    = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg.get("max_new_tokens", 40)

    # --------------------------------------------------
    # Resolve model directory (single source of truth)
    # --------------------------------------------------
    from llmlib.utils.path_util import get_model_dir

    model_dir = get_model_dir(cfg, create=False)

    print("\n============================================================")
    print("🧠 Modern GPT Inference")
    print("============================================================")
    print(f"Project config     : {config_path}")
    print(f"Model directory    : {model_dir}")
    print(f"Checkpoint         : {model_dir / 'model.pt'}")
    print(f"Model config       : {model_dir / 'model_config.json'}")
    print(f"Tokenizer          : {model_dir / 'tokenizer.json'}")
    print(f"Device             : {device}")
    print("============================================================\n")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = load_model(
        model_cls=ModernGPTModel,
        config_cls=ModernGPTConfig,
        project_config=cfg,
        device=device,
        eval_mode=True,
    )

    # --------------------------------------------------
    # Load tokenizer (FROM MODEL DIR)
    # --------------------------------------------------
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at expected location: {tokenizer_path}"
        )

    import json

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = ByteBPETokenizer.from_dict(json.load(f))

    # --------------------------------------------------
    # Prompt + generation
    # --------------------------------------------------
    def _run_once(prompt: str):
        output = _generate_text(
        model,
        tokenizer,
        prompt,
        max_seq_len,
        max_new_tokens,
        )
        print("\n------------------------------------------------------------")
        print(f"Prompt : {prompt}")
        print(f"Output : {output}")
        print("------------------------------------------------------------\n")


    if args.interactive:
        print("[modern-gpt-infer] Interactive mode (Ctrl+C to exit)\n")
        try:
            while True:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue
                if prompt.lower() in {"exit", "quit"}:
                    break
                _run_once(prompt)
        except KeyboardInterrupt:
            print("\n[modern-gpt-infer] Bye 👋")
    else:
        prompt = args.prompt or input("Enter prompt: ").strip() or "hello."
        _run_once(prompt)

    print("[modern-gpt-infer] Generating text...\n")

    output = _generate_text(
        model,
        tokenizer,
        prompt,
        max_seq_len,
        max_new_tokens,
    )

    print("------------------------------------------------------------")
    print(f"Prompt : {prompt}")
    print(f"Output : {output}")
    print("------------------------------------------------------------")


def modern_gpt_infer_old() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a Modern GPT model."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "auto"], default="auto"
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    device = _select_device(args.device)

    cfg = _load_nested_project_config(config_path)
    meta_cfg = cfg["project_metadata"]
    max_seq_len = meta_cfg["max_seq_length"]
    max_new_tokens = meta_cfg.get("max_new_tokens", 40)

    # Load model
    model = load_model(
        model_cls      = ModernGPTModel,
        config_cls     = ModernGPTConfig,
        project_config = cfg,
        device         = device,
        eval_mode      = True
    )

    # Load tokenizer
    tokenizer_path = Path(
        meta_cfg.get("tokenizer_path", config_path.parent / "tokenizer.json")
    )
    import json

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = ByteBPETokenizer.from_dict(json.load(f))

    prompt = args.prompt or input("Enter prompt: ").strip() or "hello."
    print("\n[modern-gpt-infer] Generating text...\n")
    output = _generate_text(model, tokenizer, prompt, max_seq_len, max_new_tokens)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
