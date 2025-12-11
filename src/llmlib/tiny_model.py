import math
import torch
import torch.nn as nn

from transformers import PreTrainedModel
from llmlib.tiny_config import TinyConfig

## Self-Attention Module, implements multi-head self-attention (heart of the transformer)
class TinySelfAttention(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = self.d_model // self.n_heads

        ## One projection to get Q, K, V together: (d_model) -> (3 * d_model)
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model) # (b, s, 3 * d_model)
        self.out_proj = nn.Linear(self.d_model,     self.d_model) # (b, s, d_model)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
            batch    : batch size
            seq_len  : number of tokens in a sequence
            d_model  : embedding dimension (size of the embedding vector for each token)
        returns : (batch, seq_len, d_model)
        """

        bsz, seq_len, _ = x.size()  # x.size = (batch, seq_len, d_model)

        # Project to Q, K, V: (b, s, d_model) -> (b, s, 3 * d_model)
        qkv = self.qkv_proj(x)  # (b, s, 3*d_model)

        # Split last dimension into (3, n_heads, head_dim),since d_model = n_heads * head_dim
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)  # (b, s, 3, n_heads, head_dim)

        # Rearrange, move the 3 and heads up front:
        # Since attention for head h is computed as: scores_h = Q_h @ K_h^T and for this:
        # dim1: heads
        # dim2: tokens
        # dim3: head dimension
        # Hence, (b, s, 3, n_heads, head_dim) -> (3, b, n_heads, s, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, n_heads, s, head_dim)

        # Compute scaled dot-product attention, Q @ K^T
        # scores: (b, n_heads, s, s)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (GPT-2 style): prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # shape to broadcast: (1, 1, s, s), (b, n_heads, s, s)
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

        # Score has shape (b, n_heads, s, s), where:
        # dim1: batch
        # dim2: heads
        # dim3: tokens (query)
        # dim4: tokens (key)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Softmax to get attention weights
        ### in attention, each token i must distribute its attention across all tokens j
        ### hence, softmax is applied along the last dimension (over keys, i.e., -1)
        attn_weights = torch.softmax(scores, dim=-1)

        ## attn_weights: how much should I pay attention to each previous token
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output: (b, n_heads, s, head_dim)
        ## The embedding is the raw word meaning.
        ## V is the processed meaning the model decided each token should contribute in attention.
        ### V is a transformed version of the token’s embedding, reduced to head_dim, 
        ### specialized for attention, and NOT the raw embedding or the word itself.
        attn_output = torch.matmul(attn_weights, v)  # (b, n_heads, s, head_dim)

        # Merge heads back: (b, s, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (b, s, n_heads, head_dim)
        attn_output = attn_output.view(bsz, seq_len, self.d_model) 

        # Final linear projection + dropout
        y = self.out_proj(attn_output)
        y = self.resid_dropout(y)
        return y # (b, s, d_model)

## Feed-Forward Network
class TinyFeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) for the Tiny Transformer.
    Self-attention lets each token gather info from others; the TinyFeedForward MLP
    then lets each token nonlinearly transform that info with a 2-layer neural net
    applied independently to each token.
    x → Linear → GELU → Dropout → Linear → Dropout
    """
    def __init__(self, config: TinyConfig):
        super().__init__()

        self.d_model = config.d_model
        hidden_dim = 4 * self.d_model  # standard transformer MLP expansion

        self.fc1        = nn.Linear(self.d_model, hidden_dim) # (d_model -> hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, self.d_model) # (hidden_dim -> d_model)
        self.dropout    = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x): # x: (batch, seq_len, d_model)
        x = self.fc1(x) # (batch, seq_len, hidden_dim)
        x = self.activation(x) # non-linearity
        x = self.dropout(x)     # dropout
        x = self.fc2(x)         # (batch, seq_len, d_model)    
        x = self.dropout(x)
        return x


## Transformer Block
class TinyBlock(nn.Module):
    """
    x -> LayerNorm -> Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm
    """
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = TinySelfAttention(config)
        self.mlp = TinyFeedForward(config)

    def forward(self, x):
        # Attention + residual
        # attn_out contains “token interactions” learned by the model
        attn_out = self.attn(self.ln1(x)) # first LayerNorm and Self-Attention

        ## Residual connection
        x = x + attn_out

        # Feed-forward + residual
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        return x


## Main Tiny Transformer Model
### Builds the complete GPT architecture using embeddings, many TinyBlocks, and the final output head.
class TinyTransformerModel(PreTrainedModel):
    """ Tiny Transformer Model implementing a GPT-like architecture."""
    config_class = TinyConfig

    def __init__(self, config: TinyConfig):
        super().__init__(config)

        self.d_model                 = config.d_model
        self.vocab_size              = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings

        # Token embeddings + positional embeddings
        self.wte = nn.Embedding(config.vocab_size, config.d_model)  # (batch, seq, d_model)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.d_model)  # (1, seq, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([TinyBlock(config) for _ in range(config.n_layers)])

        # Final LayerNorm (GPT-2 style)
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head for language modeling
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights (important!)
        self.post_init()

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        """
        _, seq_len = input_ids.size()

        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.max_position_embeddings}"
            )

        # 1. Token embeddings
        token_embeds = self.wte(input_ids)    # (b, s, d_model)

        # 2. Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) # (1, s)
        pos_embeds = self.wpe(positions)      # (1, s, d_model)

        # 3. Add them
        x = token_embeds + pos_embeds         # (b, s, d_model)

        # 4. Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # 5. Final Layer Normamlization
        x = self.ln_f(x)                     # (b, s, d_model)

        # 6. Output logits over vocabulary
        logits = self.lm_head(x)             # (b, s, vocab_size)

        return logits
