"""Tests for Transformer implementation correctness."""

import torch
import pytest

from papers.transformer.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
)
from papers.transformer.layers import (
    PositionalEncoding,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
)
from papers.transformer.model import Transformer


class TestScaledDotProductAttention:
    def test_output_shape(self):
        attn = ScaledDotProductAttention()
        B, H, L, d_k = 2, 8, 10, 64
        Q = torch.randn(B, H, L, d_k)
        K = torch.randn(B, H, L, d_k)
        V = torch.randn(B, H, L, d_k)
        output, weights = attn(Q, K, V)
        assert output.shape == (B, H, L, d_k)
        assert weights.shape == (B, H, L, L)

    def test_attention_weights_sum_to_one(self):
        attn = ScaledDotProductAttention(dropout=0.0)
        Q = K = V = torch.randn(1, 1, 5, 32)
        _, weights = attn(Q, K, V)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        output, weights = mha(x, x, x)
        assert output.shape == (2, 10, 512)
        assert weights.shape == (2, 8, 10, 10)


class TestTransformerModel:
    def test_forward_pass(self):
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
        )
        src = torch.randint(0, 1000, (2, 10))
        tgt = torch.randint(0, 1000, (2, 8))
        logits = model(src, tgt)
        assert logits.shape == (2, 8, 1000)

    def test_causal_mask(self):
        mask = Transformer.generate_causal_mask(5)
        assert mask.shape == (5, 5)
        # Upper triangle should be False (masked)
        assert mask[0, 4] == False  # noqa
        # Diagonal should be True (attend)
        assert mask[2, 2] == True  # noqa
