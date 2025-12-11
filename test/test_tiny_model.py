import torch

from llmlib.tokenizer import VOCAB_SIZE, encode
from llmlib.tiny_config import TinyConfig
from llmlib.tiny_model import TinyTransformerModel


def _build_tiny_model(d_model: int = 16, n_heads: int = 2, n_layers: int = 2):
    """Helper to build a tiny model with given hyperparameters."""
    cfg = TinyConfig(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_position_embeddings=32,
        dropout=0.1,
    )
    model = TinyTransformerModel(cfg)
    return cfg, model


def test_tiny_transformer_forward_shape():
    """
    Basic sanity test:
    - can we run a forward pass?
    - do we get the expected output shape (batch, seq_len, vocab_size)?
    """

    config = TinyConfig(
        vocab_size=200,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_position_embeddings=32,
        dropout=0.1,
    )

    model = TinyTransformerModel(config)

    batch_size = 2
    seq_len = 5

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    # shape check
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_tiny_transformer_no_nan_inf():
    """
    Check that the forward pass does not produce NaN or Inf values.
    """

    config = TinyConfig(
        vocab_size=200,
        d_model=16,
        n_heads=2,
        n_layers=2,
        max_position_embeddings=32,
        dropout=0.1,
    )

    model = TinyTransformerModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)

    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"


def test_d_model_affects_param_count():
    """
    Changing d_model should change the number of parameters
    (sanity check that config is actually wired into the model).
    """
    _, model_small = _build_tiny_model(d_model=16)
    _, model_big = _build_tiny_model(d_model=32)

    n_params_small = sum(p.numel() for p in model_small.parameters())
    n_params_big = sum(p.numel() for p in model_big.parameters())

    assert n_params_big > n_params_small
