# Third Party Library
import torch
from torch import nn

# First Party Library
from src import Tensor
from src.predefined_models._ViT_modules import (
    Block,
    PatchEmbedding,
    PatchShuffle,
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
        hidden_dim: int = 384 * 4,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
    ) -> None:
        """Encoder of Masked AutoEncoder

        Parameters
        ----------
        input_channels : int, optional
            Number of input image channels, by default 3
        emb_dim : int, optional
            Embedding dimensions, by default 384
        num_patch_row : int, optional
            Number of patch's row, by default 2
        image_size : int, optional
            Image size. Input images must have the same height and width,
            by default 32
        num_blocks : int, optional
            Number of iterations of transformer blocks, by default 12
        heads : int, optional
            Number of heads for Multi-Heads Self-Attention, by default 8
        hidden_dim : int, optional
            Hidden dimensions of Feed Forward Network , by default 384*4
        mask_ratio : float, optional
            Ratio of masking patches, by default 0.75
        dropout : float, optional
            Dropout rate, by default 0.0
        """
        super(MAEencoder, self).__init__()

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # Patch embedding
        self.patch_emb = PatchEmbedding(
            input_channels, emb_dim, num_patch_row, image_size
        )
        # Positional embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patch_row**2 + 1, emb_dim)
        )
        # Patch shuffling
        self.shuffle = PatchShuffle(mask_ratio)
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[
                Block(emb_dim, heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        # Layer Normalization
        self.ln = nn.LayerNorm(emb_dim)

        # Parameter initialization
        self.initialize_weight()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """forward propagation

        Parameters
        ----------
        x : Tensor
            input image. shape=(B, C, H, W)
            B: batch size, C: number of channel, N: number of patches,
            H, W: image size

        Returns
        -------
        features: Tensor
            The encoder output, which has the shape (B, N, D).
            B: batch size, N: number of patches, D: embedding dimensions
        backward_indexes: Tensor
            The second element is indexes that restores the patches
            to their original order.
        """
        patches = self.patch_emb(x)
        patches = torch.cat(
            [self.cls_token.repeat(patches.shape[0], 1, 1), patches], dim=1
        )
        patches += self.pos_emb
        patches, _, backward_indexes = self.shuffle(patches)
        features = self.ln(self.transformer(patches))
        return features, backward_indexes

    def initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.cls_token)
        nn.init.kaiming_normal_(self.pos_emb)


class MAEdecoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
        num_blocks: int = 12,
        heads: int = 8,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ) -> None:
        """Decoder of Masked AutoEncoder

        Parameters
        ----------
        input_channels : int, optional
            Number of input image channels, by default 3
        emb_dim : int, optional
            Embedding dimensions, by default 384
        num_patch_row : int, optional
            Number of patch's row, by default 2
        image_size : int, optional
            Image size. Input images must have the same height and width,
            by default 32
        num_blocks : int, optional
            Number of iterations of transformer blocks, by default 12
        heads : int, optional
            Number of heads for Multi-Heads Self-Attention, by default 8
        hidden_dim : int, optional
            Hidden dimensions of Feed Forward Network , by default 384*4
        dropout : float, optional
            Dropout rate, by default 0.0
        """
        super(MAEdecoder, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Positional embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patch_row**2 + 1, emb_dim)
        )
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[
                Block(emb_dim, heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.reconstruction = nn.Linear(
            emb_dim, input_channels * (image_size // num_patch_row) ** 2
        )

        # Parameter initialization
        self.initialize_weight()

    def forward(
        self, features: Tensor, backward_indexes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward propagation

        Parameters
        ----------
        features: Tensor
            The encoder output, which has the shape (B, n+1, D).
            B: batch size, n: number of unmasked patches,
            D: embedding dimensions
        backward_indexes: Tensor
            The second element is indexes that restores the patches
            to their original order.

        Returns
        -------
        img: Tensor
            reconstructed image.
        mask: Tensor
        """
        num_patches = features.shape[1]
        # (B, n + 1, D) -> (B, N + 1, D)
        # N is the number of all patches.
        features = torch.cat(
            [
                features,
                self.mask_token.repeat(
                    features.shape[0],
                    # calculation the number of masked patches
                    backward_indexes.shape[1] - features.shape[1],
                    1,
                ),
            ],
            dim=1,
        )

        # Restoring the order of patches
        # (B, N + 1, D) -> (B, N + 1, D)
        features = PatchShuffle.take_indexes(features, backward_indexes)
        features += self.pos_emb

        # Decoding with transformer block
        # (B, N + 1, D) -> (B, N + 1, D)
        features = self.transformer(features)

        # Reject cls_token
        # (B, N + 1, D) -> (B, N, D)
        features = features[:, 1:, :]

        # Reconstruction from embedded vector
        # (B, N, D) -> (B, N, channels * image_size * image_size)
        patches = self.reconstruction(features)

        mask = torch.zeros_like(patches)
        mask[:, num_patches - 1 :] = 1
        mask = PatchShuffle.take_indexes(mask, backward_indexes[:, 1:] - 1)

        # Reshape to image
        # (B, N, channels * image_size * image_size) --
        # --> (B, channels, image_size, image_size)
        img = patches.view(
            -1, self.input_channels, self.image_size, self.image_size
        )
        mask = mask.view(
            -1, self.input_channels, self.image_size, self.image_size
        )

        return img, mask

    def initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.mask_token)
        nn.init.kaiming_normal_(self.pos_emb)


class MAEViT(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 192,
        num_patch_row: int = 2,
        image_size: int = 32,
        encoder_num_blocks: int = 12,
        decoder_num_blocks: int = 4,
        encoder_heads: int = 3,
        decoder_heads: int = 3,
        encoder_hidden_dim: int = 768,
        decoder_hidden_dim: int = 768,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
    ) -> None:
        """_summary_

        Parameters
        ----------
        input_channels : int, optional
            Number of input image channels, by default 3
        emb_dim : int, optional
            Embedding dimensions, by default 384
        num_patch_row : int, optional
            Number of patch's row, by default 2
        image_size : int, optional
            Image size. Input images must have the same height and width,
            by default 32
        num_blocks : int, optional
            Number of iterations of transformer blocks, by default 12
        heads : int, optional
            Number of heads for Multi-Heads Self-Attention, by default 8
        hidden_dim : int, optional
            Hidden dimensions of Feed Forward Network , by default 384*4
        mask_ratio : float, optional
            Ratio of masking patches, by default 0.75
        dropout : float, optional
            Dropout rate, by default 0.0
        """
        super(MAEViT, self).__init__()
        self.encoder = MAEencoder(
            input_channels,
            emb_dim,
            num_patch_row,
            image_size,
            encoder_num_blocks,
            encoder_heads,
            encoder_hidden_dim,
            mask_ratio,
            dropout,
        )
        self.decoder = MAEdecoder(
            input_channels,
            emb_dim,
            num_patch_row,
            image_size,
            decoder_num_blocks,
            decoder_heads,
            decoder_hidden_dim,
            dropout,
        )

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        feature, indexes = self.encoder(img)
        reconst_img, mask = self.decoder(feature, indexes)
        return reconst_img, mask

    def get_last_selfattention(self, x: Tensor) -> Tensor:
        patches = self.encoder.patch_emb(x)
        patches = torch.cat(
            [self.encoder.cls_token.repeat(patches.shape[0], 1, 1), patches],
            dim=1,
        )
        patches += self.encoder.pos_emb

        patches = torch.cat(
            [self.encoder.cls_token.repeat(patches.shape[0], 1, 1), patches],
            dim=1,
        )
        for i, block in enumerate(self.encoder.transformer):
            if i < len(self.encoder.transformer) - 1:
                patches = block(patches)
            else:
                attn: Tensor = block.get_attn(patches)
        return attn


if __name__ == "__main__":
    mae_config = {
        "input_channels": 3,
        "emb_dim": 192,
        "num_patch_row": 2,
        "image_size": 32,
        "encoder_num_blocks": 12,
        "decoder_num_blocks": 4,
        "encoder_heads": 3,
        "decoder_heads": 3,
        "encoder_hidden_dim": 768,
        "decoder_hidden_dim": 768,
        "mask_ratio": 0.75,
        "dropout": 0,
    }
    mae = MAEViT(**mae_config).to("cuda")
    x = torch.randn((1, 3, 32, 32), device="cuda")
    # Third Party Library
    from torchinfo import summary

    summary(mae, (1, 3, 32, 32))
