## llmlib/modeling/modern_gpt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Config
# ------------------------------


@dataclass
class ModernGPTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 8
    max_position_embeddings: int = 512
    dropout: float = 0.1


# ------------------------------
# RMSNorm
# ------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # RMS = sqrt(mean(x^2, dim=-1, keepdim=True))
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


# ------------------------------
# RoPE helpers
# ------------------------------


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k expected shapes: (B, H, T, Dh) where Dh is head_dim.
    Returns rotated q, k (same shapes).
    """
    B, H, T, Dh = q.shape
    device = q.device

    # pair-wise frequencies for half-dim
    dim = torch.arange(0, Dh, 2, device=device).float()
    inv_freq = 1.0 / (10000 ** (dim / Dh))  # (Dh/2,)

    positions = torch.arange(T, device=device).float()  # (T,)
    freqs = torch.einsum("t,d->td", positions, inv_freq)  # (T, Dh/2)

    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh/2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh/2)

    q_even = q[..., ::2]  # (B,H,T,Dh/2)
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]

    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
    k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).flatten(-2)

    return q_rot, k_rot


# ------------------------------
# Multi-head self-attention with RoPE
# ------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # head dim must be divisible by 2 for RoPE (even number of features per head)
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask: boolean (1, 1, T, T) will be created on the fly for max seq len
        causal = torch.tril(
            torch.ones(
                config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool
            )
        ).view(
            1, 1, config.max_position_embeddings, config.max_position_embeddings
        )
        self.register_buffer("mask", causal, persistent=False)

    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape  # batch, seq, dim

        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, T, H, Dh)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE on q, k
        q, k = apply_rotary_pos_emb(q, k)

        # attention scores
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # causal mask (boolean)
        mask = self.mask[:, :, :T, :T]  # (1,1,T,T)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # padding mask
        if attention_mask is not None:
            # attention_mask: (B, T) where 1 = keep, 0 = pad
            pad_mask = attention_mask[:, None, None, :].to(torch.bool)  # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(~pad_mask, float("-inf"))
        

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        out = self.out_proj(out)
        out = self.dropout(out)
        return out


# ------------------------------
# SwiGLU feed-forward
# ------------------------------


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden_dim = int(4 * config.d_model)  # typical FF dim
        self.w1 = nn.Linear(config.d_model, hidden_dim)  # gate
        self.w2 = nn.Linear(config.d_model, hidden_dim)  # value
        self.w3 = nn.Linear(hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        # SwiGLU: silu(gate) * value
        out = F.silu(gate) * value
        out = self.w3(out)
        out = self.dropout(out)
        return out


# ------------------------------
# Decoder block
# ------------------------------


class DecoderBlock(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ffn = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ------------------------------
# Modern GPT model
# ------------------------------


class ModernGPTModel(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        input_ids: (B, T) or (T,) - if 1D, will be treated as batch size 1
        returns: logits (B, T, vocab_size)
        """
        # Handle both 1D and 2D input tensors
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension: (T,) -> (1, T)
        elif input_ids.dim() > 2:
            raise ValueError(f"input_ids should be 1D or 2D, got {input_ids.dim()}D tensor with shape {input_ids.shape}")
        
        B, T = input_ids.shape

        x = self.tok_emb(input_ids)  # (B, T, D)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits
