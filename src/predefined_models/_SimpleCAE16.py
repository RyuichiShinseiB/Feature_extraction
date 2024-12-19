# Standard Library
from typing import Callable

# Third Party Library
from torch import nn

# Local Library
from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import add_actfunc


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "relu",
        device: Device = "cpu",
    ) -> None:
        """Encoder block of SimpleCAE16

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
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        """Decoder block of SimpleCAE16

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
        self.encoder: Callable[[Tensor], Tensor] = Encoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder: Callable[[Tensor], Tensor] = Decoder(
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


if __name__ == "__main__":
    # Third Party Library
    from torchinfo import summary

    model = SimpleCAE16(
        1, 128, 64, 64, "selu", "selu", "selu", "sigmoid", device="cuda"
    )

    summary(model, (64, 1, 16, 16))
