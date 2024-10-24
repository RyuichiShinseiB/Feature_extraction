# Third Party Library
import torch
import torch.nn.functional as nnf
from torch import nn

# First Party Library
from src.mytyping import Tensor


class MultiHeadsSelfAttention(nn.Module):
    def __init__(
        self, emb_dim: int = 384, heads: int = 3, dropout: float = 0.1
    ) -> None:
        """Multi-Heads Self-Attention (MHSA) Block

        Parameters
        ----------
        emb_dim : int, optional
            Length of embedded vector, by default 384
        heads : int, optional
            Number of heads, by default 3
        dropout : float, optional
            dropout rate, by default 0.1
        """

        super(MultiHeadsSelfAttention, self).__init__()
        self.heads = heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // heads
        self.sqrt_dh = self.head_dim**0.5

        #
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout)
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """forward propagation

        Parameters
        ----------
        z : Tensor
            Input to MHSA.
            shape = (B, N, D)
            B: Batch size, N: Number of tokens, D: Length of embedded vector

        Returns
        -------
        out: Tensor
            Output from MHSA
            shape = (B, N, D)
        """
        batch_size, num_patch, _ = z.size()

        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # (B, N, D) -> (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.view(batch_size, num_patch, self.heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, num_patch, self.heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_patch, self.heads, self.head_dim).transpose(
            1, 2
        )

        # (B, h, N, D//h) X (B, h, D//h, N) -> (B, h, N, N)
        dots = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_dh

        attn = nnf.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        # (B, h, N, N) X (B, h, N, D//h) -> (B, h, N, D//h)
        out = torch.matmul(attn, v)
        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)
        # (B, N, D) -> (B, N, D)
        out = self.w_o(out)
        return out, attn


class FFN(nn.Module):
    def __init__(
        self, emb_dim: int, hidden_dim: int, dropout: float = 0.1
    ) -> None:
        """Feed Forward Network

        Parameters
        ----------
        emb_dim : int
            Length of embedded vector
        hidden_dim : int
            Length of hidden layer
        dropout : float, optional
            Dropout rate, by default 0.1
        """
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward propagation

        Parameters
        ----------
        x : Tensor
            input from MHSA to FFN.
            Shape is (B, N, D)
            B: Batch size, N: Number of tokens, D: Length of embedded vector

        Returns
        -------
        out : Tensor
            output from FFN.
            Shape is (B, N, D)
        """
        out: Tensor = self.net(x)
        return out


class Block(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        heads: int = 8,
        hidden_dim: int = 384 * 4,
        dropout: float = 0.0,
    ) -> None:
        """basic Transformer forward block

        Parameters
        ----------
        emb_dim : int, optional
            Length of embedded vector, by default 384
        heads : int, optional
            Number of heads, by default 8
        hidden_dim : int, optional
            Dimension of hidden layer, by default 384*4
        dropout : float, optional
            Dropout rate, by default 0.0
        """
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.mhsa = MultiHeadsSelfAttention(emb_dim, heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = FFN(emb_dim, hidden_dim, dropout)

    def forward(self, z: Tensor) -> Tensor:
        """Forward propagation

        Parameters
        ----------
        z : Tensor
            Input to basic Transformer block. shape = (B, N, D)
            B: batch size, N: number of tokens, D: length of embedded vector

        Returns
        -------
        (out, attn) : tuple[Tensor, Tensor]
            output to next basic Transformer block or MLP head.
            shape = (B, N, D)
        """
        out, _ = self.mhsa(self.ln1(z))
        out += z
        out = self.ffn(self.ln2(out)) + out
        return torch.Tensor(out)

    def get_attn(self, z: Tensor) -> Tensor:
        """Get attention map from Multi-Heads Self-Attention

        Parameters
        ----------
        z : Tensor
            Patches. Shape is (B, N, D)

        Returns
        -------
        attn: Tensor
            Self attention from MHSA. Shape is (B, N, N)
        """
        _, attn = self.mhsa(self.ln1(z))
        return torch.Tensor(attn)


class PatchShuffle(nn.Module):
    def __init__(self, ratio: float) -> None:
        """Patches shuffling and some patches masking

        Parameters
        ----------
        ratio : float
            Ratio of patches to be masked.
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform shuffling and masking of patches

        Parameters
        ----------
        patches : Tensor
            Patches of image. Shape is (B, N, D)
            B: batch size, N: number of tokens, D: length of embedded vector

        Returns
        -------
        patches : Tensor
            Perform shuffling and masking of patches.
            Shape is (B, n, D).
            B: batch size, n: number of remaining unmasked tokens,
            D: length of embedded vector
            n = int(N * (1 - mask_ratio))
        forward_indexes : Tensor
            Indexes for shuffling the original patches.
            Shape is (B, N)
        backward_indexes : Tensor
            Indexes for restoring to the original patches.
            Shape is (B, N)
        """
        batch_size, num_patch, _ = patches.shape
        remain_patches = int(num_patch * (1 - self.ratio))

        indexes: list[tuple[Tensor, Tensor]] = [
            self.patch_random_indexes(num_patch) for _ in range(batch_size)
        ]
        forward_indexes = torch.vstack([idx[0] for idx in indexes]).to(
            dtype=torch.long, device=patches.device
        )
        backward_indexes = torch.vstack([idx[1] for idx in indexes]).to(
            dtype=torch.long, device=patches.device
        )

        patches = self.take_indexes(patches, forward_indexes)
        patches = patches[:, :remain_patches, :]
        return patches, forward_indexes, backward_indexes

    @staticmethod
    def patch_random_indexes(num_patch: int) -> tuple[Tensor, Tensor]:
        """Function to generate indexes for random sorting of patches

        Parameters
        ----------
        num_patch : int
            Number of patches N.

        Returns
        -------
        forward_indexes: Tensor
            Indexes of randomly arranged patches.
            The first element is 0, which is the index of the cls_token
            and the elements after it are the indexes of the patches.
        backward_indexes: Tensor
            Indexes for restoring the indexes.
            Same as forward_indexes for the first element.
        """
        _forward_indexes = torch.randperm(num_patch - 1)
        forward_indexes = torch.cat([torch.zeros(1), _forward_indexes + 1])
        backward_indexes = torch.argsort(forward_indexes)
        return forward_indexes, backward_indexes

    @staticmethod
    def take_indexes(sequences: Tensor, indexes: Tensor) -> Tensor:
        """Function to sort patches

        Parameters
        ----------
        sequences : Tensor
            Data splitted input image to patches.
            shape = (B, N, D)
        indexes : Tensor
            Indexes to sorting data

        Returns
        -------
        _type_
            _description_
        """

        return torch.gather(
            sequences,
            dim=1,
            index=indexes.unsqueeze(2).repeat(1, 1, sequences.shape[-1]),
        )


class PatchEmbeddingWithShuffle(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
        mask_ratio: float = 0.75,
    ) -> None:
        super(PatchEmbeddingWithShuffle, self).__init__()
        self.input_channels = input_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.num_patch_row**2
        self.patch_size = int(self.image_size // self.num_patch_row)

        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.shuffle = PatchShuffle(mask_ratio)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        z_0: Tensor = self.patch_emb_layer(x)

        # (B, D, H/P, W/P) -> (B, D, H*W/P^2) -> (B, H*W/P^2, D)
        z_0 = z_0.flatten(2).transpose(1, 2)

        z_0, indexes = self.shuffle(z_0)
        return z_0, indexes


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
    ) -> None:
        super(PatchEmbedding, self).__init__()
        self.input_channels = input_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.num_patch_row**2
        self.patch_size = int(self.image_size // self.num_patch_row)

        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        z_0: Tensor = self.patch_emb_layer(x)

        # (B, D, H/P, W/P) -> (B, D, H*W/P^2) -> (B, H*W/P^2, D)
        z_0 = z_0.flatten(2).transpose(1, 2)

        return z_0


class PosEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        mask_ratio: float = 0.0,
    ):
        super(PosEmbedding, self).__init__()
        remain_patches = int((num_patch_row**2) * (1 - mask_ratio))

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(
            torch.randn(1, remain_patches + 1, emb_dim)
        )

    def forward(self, z_0: Tensor) -> Tensor:
        # (B, H*W/P^2, D) -> (B, N, D)
        # N is H*W/P^2 + 1
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(z_0.size(0), 1, 1)), z_0], dim=1
        )
        # (B, N, D) -> (B, N, D)
        z_0 += self.pos_emb

        return z_0


if __name__ == "__main__":
    num_patch = 5
    f, b = PatchShuffle.patch_random_indexes(num_patch - 1)
    print("f=", f)
    print("b=", b)
