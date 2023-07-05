# %% [markdown]
# # 第9回講義 宿題
#
# ## 課題
# 自己教師あり学習を用いて事前学習を行い，得られた表現をLinear probingで評価してみましょう．
# ネットワークの形などに制限はとくになく，今回のLessonで扱った内容以外の工夫も組み込んでもらって構いません．
#
# ## 目標精度
# なし
# - 自己教師あり学習の手法によっては計算リソースによって性能が大きく変わるため，目標精度は設定しておりません．
# - ただし以下の工夫を行うことで計算リソースが少なくとも，長い学習を分割して行うことができます．
#     - model，optimizer, schedulerを一定エポックで保存して，読み込むことで学習を再開することができます．
#     - 演習のようにschedulerを実装した場合は，保存は必要なく同じ引数でインスタンスを作成して\_\_call\_\_の際に与えるepochを学習の続きから与えれば動作します．
#     - 参考: https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# ## ルール
# - 予測ラベルは one_hot表現ではなく0~9のクラスラベル で表してください．
# - 自己教師あり学習では以下のセルで指定されている`x_train`以外の学習データは用いないでください．
# - Linear probingの際には`x_train`, `t_train`以外の学習データは用いないでください．
#
# ## 提出方法
# - 2つのファイルを提出していただきます．
#     - テストデータ (x_test) に対する予測ラベルをcsvファイル (ファイル名: submission_pred.csv) で提出してください．
#     - それに対応するpythonのコードをsubmission_code.pyとして提出してください (%%writefileコマンドなどを利用してください)．
#
# - コードの内容を変更した場合は，1と2の両方を提出し直してください．
#
# - なお採点は1で行い，2はコードの確認用として利用します．(成績優秀者はコード内容を公開させていただくかもしれません)
#
# ## 評価方法
#
# - 予測ラベルの`t_test`に対するAccuracyで評価します．
# - 即時採点しLeader Boardを更新します．
# - 締切時の点数を最終的な評価とします．

# %% [markdown]
# ### ドライブのマウント

# %%


# %% [markdown]
# ### データの読み込み
# - この部分は修正しないでください．

# Standard Library
# %%
import random

# Third Party Library
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm

# 学習データ
x_train = np.load("/workdir/tmp/data/x_train.npy")
t_train = np.load("/workdir/tmp/data/t_train.npy")

# テストデータ
x_test = np.load("/workdir/tmp/data/x_test.npy")


class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        data = x_train.astype("float32")
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.t_train = t_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(
            t_train[idx], dtype=torch.long
        )


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        data = x_test.astype("float32")
        self.x_test = []
        for i in range(data.shape[0]):
            self.x_test.append(Image.fromarray(np.uint8(data[i])))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.transform(self.x_test[idx])


trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)

# %% [markdown]
# ### データローダの準備

# %%
val_size = 3000
train_data, valid_data = torch.utils.data.random_split(
    trainval_data, [len(trainval_data) - val_size, val_size]
)

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)  # WRITE ME
valid_transform = transforms.Compose([transforms.ToTensor()])  # WRITE ME
test_transform = transforms.Compose([transforms.ToTensor()])  # WRITE ME

batch_size = 128

dataloader_train = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

# %% [markdown]
# ### 自己教師あり学習の実装
# - 初期の形式はMAEを利用することを想定していますが，他の自己教師あり学習を利用していただいて構いません．

# Standard Library
from typing import TypeAlias

# Third Party Library
# %%
import torch.nn.functional as nnf
from torch import nn, optim

Tensor: TypeAlias = torch.Tensor


def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_seed(seed=42)


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


# cosine scheduler
class CosineScheduler:
    def __init__(self, epochs, lr, warmup_length=5):
        """
        Arguments
        ---------
        epochs : int
            学習のエポック数．
        lr : float
            学習率．
        warmup_length : int
            warmupを適用するエポック数．
        """
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Arguments
        ---------
        epoch : int
            現在のエポック数．
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        if self.warmup:
            lr = lr * min(1.0, (epoch + 1) / self.warmup)

        return lr


def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ハイパーパラメータの設定
config = {
    "input_channels": 3,
    "emb_dim": 192,
    "num_patch_row": 16,
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


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MAEViT(**config).to(device)

epochs = 200
lr = 0.003
warmup_length = epochs // 10
batch_size = 512
step_count = 0
optimizer = optim.AdamW(
    model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05
)
scheduler = CosineScheduler(epochs, lr, warmup_length)

# %%
print(device)

# %% [markdown]
# ### 事前学習（自己教師あり学習）

# %%
for epoch in range(epochs):
    # スケジューラで学習率を更新する
    new_lr = scheduler(epoch)
    set_lr(new_lr, optimizer)

    total_train_loss = 0.0
    total_valid_loss = 0.0

    # モデルの訓練
    for x, _ in dataloader_train:
        step_count += 1
        model.train()
        x = x.to(device)

        rec_img, mask = model(x)
        train_loss = (
            torch.mean((rec_img - x) ** 2 * mask) / config["mask_ratio"]
        )
        train_loss.backward()

        if step_count % 8 == 0:  # 8イテレーションごとに更新することで，擬似的にバッチサイズを大きくしている
            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += train_loss.item()

    # モデルの評価
    with torch.no_grad():
        for x, _ in dataloader_valid:
            model.eval()

            with torch.no_grad():
                x = x.to(device)

                rec_img, mask = model(x)
                valid_loss = (
                    torch.mean((rec_img - x) ** 2 * mask)
                    / config["mask_ratio"]
                )

                total_valid_loss += valid_loss.item()

    print(
        f"Epoch[{epoch+1} / {epochs}] Train Loss: {total_train_loss/len(dataloader_train):.4f} Valid Loss: {total_valid_loss/len(dataloader_valid):.4f}"
    )

# モデルを保存しておく
torch.save(
    model.state_dict(), "/workdir/tmp/MYmodel/MAE_pretrain_params_v2.pth"
)

# %%
torch.save(
    model.state_dict(), "/workdir/tmp/MYmodel/MAE_pretrain_params_v2.pth"
)

# %%
del model, train_loss, valid_loss
torch.cuda.empty_cache()

# %% [markdown]
# ### Linear probing

# %%
val_size = 3000
train_data, valid_data = torch.utils.data.random_split(
    trainval_data, [len(trainval_data) - val_size, val_size]
)

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)  # WRITE ME
valid_transform = transforms.Compose([transforms.ToTensor()])  # WRITE ME
test_transform = transforms.Compose([transforms.ToTensor()])  # WRITE ME

batch_size = 128

dataloader_train = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

dataloader_valid = torch.utils.data.DataLoader(
    valid_data, batch_size=batch_size, shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

# %%
model.load_state_dict(
    torch.load("/workdir/tmp/MYmodel/MAE_pretrain_params_v2.pth")
)


# %%
class MLPClassifier(nn.Module):
    def __init__(
        self,
        hidden_dims: tuple[int, int] = (64, 32),
        num_classes: int = 10,
        dropout_ratio: float = 0.3,
    ) -> None:
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(config["emb_dim"], hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.layer3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.dropout1(x)
        x = nnf.leaky_relu(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = nnf.leaky_relu(x)
        x = self.layer3(x)
        x = nnf.softmax(x, dim=1)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = model.encoder  # WRITE ME
classifier_model = MLPClassifier().to(device)  # WRITE ME
epochs = 100
lr = 0.001
warmup_length = 10
batch_size = 64
optimizer = optim.AdamW(
    classifier_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05
)  # 分類器部分のみ学習
scheduler = CosineScheduler(epochs, lr, warmup_length)
criterion = nn.CrossEntropyLoss()

# %%
encoder.eval()
for epoch in range(epochs):
    new_lr = scheduler(epoch)
    set_lr(new_lr, optimizer)

    total_train_loss = 0.0
    total_train_acc = 0.0
    total_valid_loss = 0.0
    total_valid_acc = 0.0
    for x, t in dataloader_train:
        x, t = x.to(device), t.to(device)
        feature, _ = encoder(x)
        pred = classifier_model(feature[:, 0, :])

        train_loss = criterion(pred, t)
        train_acc = (torch.argmax(pred, dim=1) == t).float().mean().cpu()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()
        total_train_acc += train_acc

    with torch.no_grad():
        for x, t in dataloader_valid:
            x, t = x.to(device), t.to(device)
            feature, _ = encoder(x)
            pred = classifier_model(feature[:, 0, :])

            valid_loss = criterion(pred, t)
            valid_acc = (torch.argmax(pred, dim=1) == t).float().mean().cpu()

            total_valid_loss += valid_loss.item()
            total_valid_acc += valid_acc

    print(
        f"Epoch[{epoch+1} / {epochs}]",
        f"Train Loss: {total_train_loss/len(dataloader_train):.4f}",
        f"Train Acc.: {total_train_acc/len(dataloader_train):.4f}",
        f"Valid Loss: {total_valid_loss/len(dataloader_valid):.4f}",
        f"Valid Acc.: {total_valid_acc/len(dataloader_valid):.4f}",
    )

torch.save(
    classifier_model.state_dict(),
    "/workdir/tmp/MYmodel/MAE_classifier_params_v2.pth",
)

# %%
feature[:, 0, :].shape

# %%
classifier_model.eval()

t_pred = []
for x in dataloader_test:
    x = x.to(device)
    feature, _ = encoder(x)
    y = classifier_model(feature[:, 0, :])

    # モデルの出力を予測値のスカラーに変換
    pred = y.argmax(1).tolist()
    t_pred.extend(pred)

submission = pd.Series(t_pred, name="label")
submission.to_csv("submission_pred_v2.csv", header=True, index_label="id")

# %%
