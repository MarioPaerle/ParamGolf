import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class AttentionGate(nn.Module):
    """Base class for attention gates."""
    hook = "output"

    def forward(self, attn_out: Tensor, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError


class OutputInputWeightedGate(AttentionGate):
    """
    Your idea: cat(attn_out[:gd//2], x_input[:gd//2]) → sigmoid gate on attn_out.
    gate_dim=12 → 6 dims from output, 6 from input.
    mode="scalar" → 1 scalar per token (cheapest, ~13 params)
    mode="headwise" → H scalars averaged → still 1 gate applied to full D
    """
    hook = "output"

    def __init__(self, gate_dim: int = 12, num_heads: int = 1, mode: str = "scalar"):
        super().__init__()
        assert gate_dim % 2 == 0
        assert mode in ("scalar", "headwise")
        self.gate_dim = gate_dim
        self.half = gate_dim // 2
        self.mode = mode
        out_dim = 1 if mode == "scalar" else num_heads
        self.W = nn.Parameter(torch.zeros(out_dim, gate_dim))

    def forward(self, attn_out: Tensor, x: Tensor, **kwargs) -> Tensor:
        # attn_out, x: (B, T, D)
        half = self.half
        inp = torch.cat([attn_out[..., :half], x[..., :half]], dim=-1)  # (B,T,gate_dim)
        g = torch.sigmoid(F.linear(inp, self.W))                         # (B,T,1 or H)
        if self.mode == "headwise":
            g = g.mean(dim=-1, keepdim=True)  # (B,T,1)
        return attn_out * g