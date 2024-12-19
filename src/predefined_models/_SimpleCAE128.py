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


if __name__ == "__main__":
    # Third Party Library
    from torchinfo import summary

    model = SimpleCAE128(1, 16, 16, 10, device="cuda")

    summary(model, (1, 1, 128, 128))
