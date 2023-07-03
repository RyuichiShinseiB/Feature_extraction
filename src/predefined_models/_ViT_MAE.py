# Third Party Library
import torch
from torch import nn

# First Party Library
from src import Tensor
from src.predefined_models._ViT_modules import (
    Block,
    PatchEmbedding,
    PosEmbedding,
)


class MAEencoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
        num_blocks: int = 12,
        heads: int = 8,
        mask_ratio: float = 0.75,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ) -> None:
        super(MAEencoder, self).__init__()
        # Patch embedding and Patch shuffle
        self.patch_emb = PatchEmbedding(
            input_channels, emb_dim, num_patch_row, image_size, mask_ratio
        )
        # Positional embedding
        self.pos_emb = PosEmbedding(emb_dim, num_patch_row, mask_ratio)
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[
                Block(emb_dim, heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        # Layer Normalization
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # indexes = (forward indexes, backward indexes)
        patches, indexes = self.patch_emb.forward(x)
        patches = self.pos_emb(patches)
        features = self.ln(self.transformer(patches))
        return features, indexes


if __name__ == "__main__":
    encoder = MAEencoder().to("cuda")
    x = torch.randn((1, 3, 32, 32), device="cuda")
    out = encoder.forward(x)

    # Third Party Library
    from torchinfo import summary

    summary(encoder, (1, 3, 32, 32))
