"""
Attention Is All You Need — Transformer Implementation.

From-scratch PyTorch implementation of the Transformer architecture
as described in Vaswani et al. (2017).

Paper: https://arxiv.org/abs/1706.03762

This module implements multi-head scaled dot-product attention,
the core mechanism of the Transformer architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Reference: Section 3.2.1 of the paper.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # (batch, heads, seq_len, d_k)
        key: torch.Tensor,      # (batch, heads, seq_len, d_k)
        value: torch.Tensor,    # (batch, heads, seq_len, d_v)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (B, H, L_q, d_k)
            key: Key tensor of shape (B, H, L_k, d_k)
            value: Value tensor of shape (B, H, L_k, d_v)
            mask: Optional mask tensor

        Returns:
            output: Attention-weighted values (B, H, L_q, d_v)
            attn_weights: Attention weights (B, H, L_q, L_k)
        """
        d_k = query.size(-1)

        # Compute attention scores: QK^T / sqrt(d_k)
        # (B, H, L_q, d_k) @ (B, H, d_k, L_k) → (B, H, L_q, L_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (for decoder self-attention and padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (key positions)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # (B, H, L_q, L_k) @ (B, H, L_k, d_v) → (B, H, L_q, d_v)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Reference: Section 3.2.2 of the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # W^Q, W^K, W^V ∈ R^{d_model × d_k}
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection W^O ∈ R^{h·d_v × d_model}
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,    # (B, L_q, d_model)
        key: torch.Tensor,      # (B, L_k, d_model)
        value: torch.Tensor,    # (B, L_k, d_model)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.

        Args:
            query: (B, L_q, d_model)
            key: (B, L_k, d_model)
            value: (B, L_k, d_model)
            mask: Optional attention mask

        Returns:
            output: (B, L_q, d_model)
            attn_weights: (B, H, L_q, L_k)
        """
        batch_size = query.size(0)

        # 1. Linear projections: (B, L, d_model) → (B, L, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads: (B, L, d_model) → (B, H, L, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 4. Concatenate heads: (B, H, L, d_k) → (B, L, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 5. Final linear projection
        output = self.W_o(attn_output)

        return output, attn_weights
