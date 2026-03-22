"""
Transformer — Encoder/Decoder Layers and Components.

Implements positional encoding, feed-forward network,
encoder layer, and decoder layer.

Reference: Sections 3.3, 3.4, 3.5 of the paper.
"""

import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Reference: Section 3.5 of the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register as buffer (not a parameter, but saved with model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (B, L, d_model)

        Returns:
            Positionally encoded tensor of shape (B, L, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Two linear transformations with a ReLU activation in between.
    Reference: Section 3.3 of the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Consists of:
    1. Multi-head self-attention + Add & Norm
    2. Feed-forward network + Add & Norm

    Reference: Section 3.1 of the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Self-attention with residual connection & layer norm
        attn_output, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Feed-forward with residual connection & layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention + Add & Norm
    2. Multi-head cross-attention (over encoder output) + Add & Norm
    3. Feed-forward network + Add & Norm

    Reference: Section 3.1 of the paper.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Cross-attention (Q from decoder, K/V from encoder)
        cross_output, _ = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(cross_output))

        # 3. Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x
