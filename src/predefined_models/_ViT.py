# Third Party Library
import torch
import torch.nn.functional as nnf
from torch import nn

# Local Library
from .. import Tensor


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
        print(out.shape)
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
            input to FFN

        Returns
        -------
        out : Tensor
            output of FFN
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

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
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
        out, attn = self.mhsa(self.ln1(z))
        out += z
        out = self.ffn(self.ln2(out)) + out
        return out, attn


class PatchPosEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
    ) -> None:
        super(PatchPosEmbedding, self).__init__()
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch + 1, self.emb_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P)
        z_0: Tensor = self.patch_emb_layer(x)

        # (B, D, H/P, W/P) -> (B, D, H*W/P^2) -> (B, H*W/P^2, D)
        z_0 = z_0.flatten(2).transpose(1, 2)

        # (B, H*W/P^2, D) -> (B, N, D)
        # N is H*W/P^2 + 1
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0), 1, 1)), z_0], dim=1
        )
        # (B, N, D) -> (B, N, D)
        z_0 += self.pos_emb

        return z_0
