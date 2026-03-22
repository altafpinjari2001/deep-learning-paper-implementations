"""
Vision Transformer (ViT) — From-scratch implementation.

Paper: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
Reference: https://arxiv.org/abs/2010.11929

Key idea: Split an image into fixed-size patches, linearly embed them,
add position embeddings, and feed to a standard Transformer encoder.
"""

import torch
import torch.nn as nn

from ..transformer.attention import MultiHeadAttention
from ..transformer.layers import FeedForward


class PatchEmbedding(nn.Module):
    """
    Split image into patches and project to embedding dimension.

    For a 224×224 image with 16×16 patches:
    - Number of patches = (224/16)² = 196
    - Each patch is flattened to 16×16×3 = 768 dimensions

    Reference: Section 3.1 of the paper.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        # Using Conv2d with kernel_size=patch_size is equivalent
        # to splitting into patches and applying a linear layer
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        # (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) → (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    """Single Vision Transformer block (Pre-norm variant)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: LayerNorm → Attention → Residual
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_out

        # Pre-norm: LayerNorm → MLP → Residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT).

    Architecture:
    1. Split image into patches
    2. Linear embedding + positional embedding + [CLS] token
    3. N × Transformer encoder blocks
    4. MLP classification head on [CLS] token

    Default: ViT-Base (B/16) configuration
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embedding.num_patches

        # Learnable [CLS] token — prepended to patch sequence
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # Learnable position embeddings
        # +1 for [CLS] token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        batch_size = x.size(0)

        # 1. Patch embedding: (B, C, H, W) → (B, N, D)
        x = self.patch_embedding(x)

        # 2. Prepend [CLS] token: (B, N+1, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. Add positional embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 4. Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # 5. Extract [CLS] token output
        x = self.norm(x[:, 0])  # Only [CLS] token

        # 6. Classification head
        x = self.head(x)

        return x
