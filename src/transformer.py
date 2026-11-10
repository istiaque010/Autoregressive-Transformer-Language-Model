"""Pre-LayerNorm Transformer decoder block (Phase 5)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """LayerNorm -> causal MHA -> residual -> LayerNorm -> MLP -> residual."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
