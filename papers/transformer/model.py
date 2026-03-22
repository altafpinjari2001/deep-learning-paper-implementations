"""
Transformer — Full Model.

Complete Transformer architecture combining encoder and decoder
stacks with token embeddings and output projection.

Reference: Figure 1 of the paper.
"""

import torch
import torch.nn as nn

from .layers import (
    PositionalEncoding,
    EncoderLayer,
    DecoderLayer,
)


class Transformer(nn.Module):
    """
    Full Transformer model (Encoder-Decoder).

    Architecture (from the paper):
    - Input Embedding → Positional Encoding → N × Encoder Layers
    - Output Embedding → Positional Encoding → N × Decoder Layers
    - Linear → Softmax

    Default hyperparameters from the paper:
    - d_model = 512
    - num_heads = 8
    - d_ff = 2048
    - num_layers = 6
    - dropout = 0.1
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding (shared)
        self.positional_encoding = PositionalEncoding(
            d_model, max_len, dropout
        )

        # Encoder stack: N identical layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder stack: N identical layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token IDs (B, L_src)
            src_mask: Source attention mask

        Returns:
            Encoder output (B, L_src, d_model)
        """
        # Embed + scale + positional encoding
        x = self.src_embedding(src) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder output.

        Args:
            tgt: Target token IDs (B, L_tgt)
            encoder_output: Encoder output (B, L_src, d_model)
            src_mask: Source attention mask
            tgt_mask: Target (causal) attention mask

        Returns:
            Decoder output (B, L_tgt, d_model)
        """
        x = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode → decode → project.

        Args:
            src: Source tokens (B, L_src)
            tgt: Target tokens (B, L_tgt)
            src_mask: Source mask
            tgt_mask: Target causal mask

        Returns:
            Logits over target vocabulary (B, L_tgt, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        logits = self.output_projection(decoder_output)

        return logits

    @staticmethod
    def generate_causal_mask(size: int) -> torch.Tensor:
        """
        Generate causal (look-ahead) mask for decoder.

        Prevents attending to future positions.
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0  # True = attend, False = masked
