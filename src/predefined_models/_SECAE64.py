# Third Party Library
from torch import nn

# Local Library
from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import DownShape, SELayer, UpShape


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
        super(Encoder, self).__init__()
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
        super(Decoder, self).__init__()
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
        self.encoder = Encoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = Decoder(
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
