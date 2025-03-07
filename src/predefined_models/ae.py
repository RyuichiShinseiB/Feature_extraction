from collections.abc import Callable

from torch import nn

from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import DownShape, SELayer, UpShape, add_actfunc


##### SimpleCAE #####
# ========================== Top For 16x16px size =========================== #
class SimpleConvEncoder16(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        """SimpleConvEncoder block of SimpleCAE16

        Parameters
        ----------
        input_channels : int, optional
            Number of input image channels, by default 1\n
        encoder_base_channels : int, optional
            Basic number of channels in the encoder block.
            With each layer, the number of channels in each layer increases
            by a factor of 2. , by default 64\n
        latent_dimensions : int, optional
            Number dimension of feature vector, by default 10\n
        actfunc : ActFuncName, optional
            ActFunc function at each layer.
            Please select in `relu`, `leakyrelu`, `selu`, `sigmoid`, `tanh`,
            `identity`, by default "relu"\n
        output_actfunc : ActFuncName, optional
            ActFunc function at output layer of decoder, by default "relu"\n
        device : Device, optional
            Device to place encoder., by default "cpu"\n
        """
        super().__init__()
        self.device = device
        # (B, C, 16, 16) -> (B, ebc, 8, 8)
        # ebc is encoder_base_channels
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_base_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(encoder_base_channels),
            add_actfunc(actfunc),
        )
        # (B, ebc, 8, 8) -> (B, ebc * 2, 4, 4)
        self.l2 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels,
                encoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        # (B, ebc * 2, 4, 4) -> (B, ebc * 4, 2, 2)
        self.l3 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        # (B, ebc * 4, 2, 2) -> (B, ld, 1, 1)
        # ld is latent_dimension
        self.l4 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 4,
                latent_dimensions,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(latent_dimensions),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        return x


######################################################
class SimpleConvDecoder16(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        """SimpleConvDecoder block of SimpleCAE16

        Parameters
        ----------
        input_channels : int, optional
            Number of input image channels, by default 1\n
        decoder_base_channels : int, optional
            Basic number of channels in the decoder block.
            With each layer, the number of channels in each layer decreases
            by a factor of 2. , by default 64\n
        latent_dimensions : int, optional
            Number dimension of feature vector, by default 10\n
        actfunc : ActFuncName, optional
            ActFunc function at each layer.
            Please select in `relu`, `leakyrelu`, `selu`, `sigmoid`, `tanh`,
            `identity`, by default "relu"\n
        output_actfunc : ActFuncName, optional
            ActFunc function at output layer of decoder, by default "relu"\n
        device : Device, optional
            Device to place decoder., by default "cpu"\n
        """
        super().__init__()
        self.device = device
        # (B, ld, 1, 1) -> (B, dbc * 4, 2, 2)
        # dbc is decoder_base_channels
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions,
                decoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        # (B, dbc * 4, 2, 2) -> (B, dbc * 2, 4, 4)
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        # (B, dbc * 2, 4, 4) -> (B, dbc, 8, 8)
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 2,
                decoder_base_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels),
            add_actfunc(actfunc),
        )
        # (B, dbc, 8, 8) -> (B, C, 16, 16)
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels,
                input_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


class SimpleCAE16(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "selu",
        decoder_output_actfunc: ActFuncName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        """Simply convolutional autoencoder for 16px x 16px images

        Parameters
        ----------
        input_channels : int
            Number of channels of images to be trained.\n
        latent_dimensions : int
            Number dimension of feature vector, by default 10\n
        encoder_base_channels : int
            Basic number of channels in the encoder block.
            With each layer, the number of channels in each layer increases
            by a factor of 2. , by default 64\n
        decoder_base_channels : int
            Basic number of channels in the decoder block.
            With each layer, the number of channels in each layer decreases by
            a factor of 2. , by default 64\n
        encoder_actfunc : ActFuncName, optional
            ActFunc function at the encoder block, by default "relu"\n
        decoder_actfunc : ActFuncName, optional
            ActFunc function at the decoder block, by default "relu"\n
        encoder_output_actfunc : ActFuncName, optional
            ActFunc function at output of the encoder block,
            by default "selu"\n
        decoder_output_actfunc : ActFuncName, optional
            ActFunc function at output of the decoder block,
            by default "sigmoid"\n
        device : Device, optional
            Device to place decoder., by default "cpu"\n
        """
        super(SimpleCAE16, self).__init__()
        self.device = device
        self.encoder: Callable[[Tensor], Tensor] = SimpleConvEncoder16(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder: Callable[[Tensor], Tensor] = SimpleConvDecoder16(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z


# ========================= Bottom For 16x16px size ========================= #


# ========================== Top For 32x32px size =========================== #
class SimpleConvEncoder32(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_base_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(encoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels,
                encoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 8,
                latent_dimensions,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(latent_dimensions),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        return x


######################################################
class SimpleConvDecoder32(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions,
                decoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 2,
                decoder_base_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels, input_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x


class SimpleCAE32(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "selu",
        decoder_output_actfunc: ActFuncName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super(SimpleCAE32, self).__init__()
        self.device = device
        self.encoder: Callable[[Tensor], Tensor] = SimpleConvEncoder32(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder: Callable[[Tensor], Tensor] = SimpleConvDecoder32(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z


# ========================= Bottom For 32x32px size ========================= #


# ========================== Top For 64x64px size =========================== #
class SimpleConvEncoder64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_base_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(encoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels,
                encoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 8,
                encoder_base_channels * 16,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 16),
            add_actfunc(actfunc),
        )
        self.l6 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 16,
                latent_dimensions,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(latent_dimensions),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)

        return h


######################################################
class SimpleConvDecoder64(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions,
                decoder_base_channels * 16,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 16),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 16,
                decoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 2,
                decoder_base_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels, input_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


class SimpleCAE64(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "relu",
        decoder_output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SimpleCAE64, self).__init__()
        self.device = device

        self.encoder = SimpleConvEncoder64(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )

        self.decoder = SimpleConvDecoder64(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z


# ========================= Bottom For 64x64px size ========================= #


# ========================= Top For 128x128px size ========================== #
class SimpleConvEncoder128(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_base_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(encoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels,
                encoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 8,
                encoder_base_channels * 16,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 16),
            add_actfunc(actfunc),
        )
        self.l6 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 16,
                encoder_base_channels * 32,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_base_channels * 32),
            add_actfunc(actfunc),
        )
        self.l7 = nn.Sequential(
            nn.Conv2d(
                encoder_base_channels * 32,
                latent_dimensions,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(latent_dimensions),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        return x


######################################################
class SimpleConvDecoder128(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions,
                decoder_base_channels * 32,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 32),
            add_actfunc(actfunc),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 32,
                decoder_base_channels * 16,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 16),
            add_actfunc(actfunc),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 16,
                decoder_base_channels * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 8),
            add_actfunc(actfunc),
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 4),
            add_actfunc(actfunc),
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels * 2),
            add_actfunc(actfunc),
        )
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels * 2,
                decoder_base_channels,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_base_channels),
            add_actfunc(actfunc),
        )
        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels, input_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            add_actfunc(output_actfunc),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


class SimpleCAE128(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "relu",
        decoder_output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SimpleCAE128, self).__init__()
        if decoder_actfunc is None:
            decoder_actfunc = encoder_actfunc
        if encoder_output_actfunc is None:
            encoder_output_actfunc = encoder_actfunc
        self.device = device
        self.encoder = SimpleConvEncoder128(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = SimpleConvDecoder128(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z


# ======================== Bottom For 128x128px size ======================== #


##### SECAE #####
# ========================== Top For 32x32px size =========================== #
class SEConvEncoder32(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        super(SEConvEncoder32, self).__init__()
        self.device = device

        self.l1 = nn.Sequential(
            DownShape(input_channels, encoder_base_channels, actfunc),
            SELayer(encoder_base_channels),
        )
        self.l2 = nn.Sequential(
            DownShape(
                encoder_base_channels, encoder_base_channels * 2, actfunc
            ),
            SELayer(encoder_base_channels * 2),
        )
        self.l3 = nn.Sequential(
            DownShape(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                actfunc,
            ),
            SELayer(encoder_base_channels * 4),
        )
        self.l4 = nn.Sequential(
            DownShape(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                actfunc,
            ),
            SELayer(encoder_base_channels * 8),
        )
        self.l5 = nn.Sequential(
            DownShape(
                encoder_base_channels * 8, latent_dimensions, output_actfunc
            ),
            SELayer(latent_dimensions),
        )

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)

        return h


########################################


class SEConvDecoder32(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SEConvDecoder32, self).__init__()
        self.device = device

        self.l1 = nn.Sequential(
            UpShape(latent_dimensions, decoder_base_channels * 8, actfunc),
            SELayer(decoder_base_channels * 8),
        )
        self.l2 = nn.Sequential(
            UpShape(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                actfunc,
            ),
            SELayer(decoder_base_channels * 4),
        )
        self.l3 = nn.Sequential(
            UpShape(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                actfunc,
            ),
            SELayer(decoder_base_channels * 2),
        )
        self.l4 = nn.Sequential(
            UpShape(decoder_base_channels * 2, decoder_base_channels, actfunc),
            SELayer(decoder_base_channels),
        )
        self.l5 = nn.Sequential(
            UpShape(decoder_base_channels, input_channels, output_actfunc),
            SELayer(input_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        return x


########################################


class SECAE32(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "sigmoid",
        decoder_output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SECAE32, self).__init__()
        self.device = device
        self.encoder = SEConvEncoder32(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = SEConvDecoder32(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z: Tensor = self.encoder(x)
        x_pred: Tensor = self.decoder(z)
        return x_pred, z


# ========================= Bottom For 32x32px size ========================= #


# ========================== Top For 64x64px size =========================== #
class SEConvEncoder64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        super(SEConvEncoder64, self).__init__()
        if output_actfunc is None:
            output_actfunc = actfunc

        self.device = device

        self.l1 = nn.Sequential(
            DownShape(input_channels, encoder_base_channels, actfunc),
            SELayer(encoder_base_channels),
        )
        self.l2 = nn.Sequential(
            DownShape(
                encoder_base_channels, encoder_base_channels * 2, actfunc
            ),
            SELayer(encoder_base_channels * 2),
        )
        self.l3 = nn.Sequential(
            DownShape(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                actfunc,
            ),
            SELayer(encoder_base_channels * 4),
        )
        self.l4 = nn.Sequential(
            DownShape(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                actfunc,
            ),
            SELayer(encoder_base_channels * 8),
        )
        self.l5 = nn.Sequential(
            DownShape(
                encoder_base_channels * 8,
                encoder_base_channels * 16,
                actfunc,
            ),
            SELayer(encoder_base_channels * 16),
        )
        self.l6 = nn.Sequential(
            DownShape(
                encoder_base_channels * 16,
                latent_dimensions,
                output_actfunc,
            ),
            SELayer(latent_dimensions),
        )

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)

        return h


########################################


class SEConvDecoder64(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SEConvDecoder64, self).__init__()
        self.device = device

        self.l1 = nn.Sequential(
            UpShape(latent_dimensions, decoder_base_channels * 16, actfunc),
            SELayer(decoder_base_channels * 16),
        )
        self.l2 = nn.Sequential(
            UpShape(
                decoder_base_channels * 16,
                decoder_base_channels * 8,
                actfunc,
            ),
            SELayer(decoder_base_channels * 8),
        )
        self.l3 = nn.Sequential(
            UpShape(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                actfunc,
            ),
            SELayer(decoder_base_channels * 4),
        )
        self.l4 = nn.Sequential(
            UpShape(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                actfunc,
            ),
            SELayer(decoder_base_channels * 2),
        )
        self.l5 = nn.Sequential(
            UpShape(decoder_base_channels * 2, decoder_base_channels, actfunc),
            SELayer(decoder_base_channels),
        )
        self.l6 = nn.Sequential(
            UpShape(decoder_base_channels, input_channels, output_actfunc),
            SELayer(input_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        return x


########################################


class SECAE64(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "sigmoid",
        decoder_output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SECAE64, self).__init__()
        self.device = device
        self.encoder = SEConvEncoder64(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = SEConvDecoder64(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z
