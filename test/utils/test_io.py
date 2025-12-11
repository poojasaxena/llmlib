#!/usr/bin/env python
"""
Quick smoke test for llmlib:
- loads project_config
- finds data file
- builds TinyConfig from tokenizer + project_config
- trains 1 tiny forward pass
- saves and reloads the model
"""
import os
import torch

from llmlib.io import (
    load_project_config,
    get_data_file_path,
    save_tiny_model,
    load_tiny_model,
)
from llmlib.tokenization.tokenizer import encode, VOCAB_SIZE
from llmlib.tiny_config import TinyConfig
from llmlib.tiny_model import TinyTransformerModel


def main():
    # 1) Adjust this path to your train.py location

    learn_root = os.environ.get("LEARNING_ROOT")
    if learn_root is None:
        raise RuntimeError("LEARNING_ROOT environment variable is not set")

    cfg = load_project_config(
        caller_file=f"{learn_root}/NLP_and_LLMs/Transformers_Fundamentals/course3_building-a-mini_gpt/projects/1_tiny_transformer_intro/train.py",
        config_filename="project_config.json",
    )

    print("Loaded project_config:")
    print(cfg)

    # unpack nested config
    model_cfg = cfg["model_config"]
    train_cfg = cfg["training_config"]
    meta_cfg = cfg["project_metadata"]

    # 2) Check that data file is found
    data_path = get_data_file_path(cfg)
    print(f"Data file found at: {data_path}")

    # 3) Build TinyConfig
    config = TinyConfig(
        vocab_size=VOCAB_SIZE,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_position_embeddings=meta_cfg["max_seq_length"],
        dropout=model_cfg["dropout"],
    )
    print("TinyConfig OK:", config)

    # 4) Build model and do one forward pass
    model = TinyTransformerModel(config)
    model.eval()

    dummy_ids = encode("hello.")[: meta_cfg["max_seq_length"]].unsqueeze(0)
    with torch.no_grad():
        logits = model(dummy_ids)
    print("Forward pass OK. logits shape:", logits.shape)

    # 5) Save + reload model
    ckpt_path = save_tiny_model(model, cfg)
    print(f"Model saved to: {ckpt_path}")

    reloaded = load_tiny_model(cfg, device="cpu", eval_mode=True)
    with torch.no_grad():
        logits2 = reloaded(dummy_ids)
    print("Reloaded model forward OK. logits2 shape:", logits2.shape)


def test_io_smoke():
    """Pytest-compatible wrapper around the smoke test."""
    main()

if __name__ == "__main__":
    main()
