## Configuration class for a Tiny Transformer model.
from transformers import PretrainedConfig


class TinyConfig(PretrainedConfig):
    """
    Configuration class for the Tiny Transformer model.

    This docstring documents the constructor arguments and attributes of TinyConfig,
    which encapsulates the hyperparameters required to instantiate a Tiny Transformer
    model. Instances of this class are typically passed to model constructors so the
    model architecture and runtime behavior can be reproduced or configured programmatically.

    Parameters
    ----------
    vocab_size : int, optional (default=200)
        Size of the vocabulary — the number of unique token ids the model expects.
    d_model : int, optional (default=16)
        Embedding Dimensionality (the embedding size / model width).
    n_heads : int, optional (default=2)
        Number of attention heads in each multi-head attention layer.
    n_layers : int, optional (default=2)
        Number of transformer encoder/decoder layers (stack depth).
    max_position_embeddings : int, optional (default=32)
        Maximum sequence length that this model might ever be used with. This defines
        the size of positional embeddings.
    dropout_rate : float, optional (default=0.1)
        Dropout probability applied to attention scores, feed-forward layers, and embeddings
        where dropout is used.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the parent PretrainedConfig initializer
        (e.g., id2label, label2id, bos_token_id, eos_token_id).

    Notes
    -----
    - Ensure that the combination of d_model and n_heads is compatible with any
      implementation constraints (for example, some implementations require that
      d_model is divisible by n_heads).
    - This configuration class extends PretrainedConfig, so it supports standard
      configuration serialization/deserialization utilities provided by the
      Transformers-style ecosystem.

    Example
    -------
    >>> config = TinyConfig(vocab_size=1000, d_model=64, n_heads=4, n_layers=6)
    >>> model  = TinyTransformerModel(config)
    """

    ## model_type (str): The model type identifier for the Tiny Transformer model.
    model_type = 'tiny_transformer'

    def __init__(
        self,
        vocab_size=0,    ## number of trainable tokens or vocabulary size
        d_model=16,      ## embedding dimension
        n_heads=2,       ## number of attention heads
        n_layers=2,      ## number of transformer layers
        max_position_embeddings=32, ## maximum sequence length
        dropout=0.1,    ## dropout rate
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.max_position_embeddings = max_position_embeddings
        self.dropout    = dropout