# Third Party Library
from torch import nn

# Local Library
from .. import ActivationName, Device, Tensor
from ._modules import add_activation


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            nn.Conv2d(
                input_channels, encoder_base_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(encoder_base_channels),
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(output_activation),
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
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "tanh",
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
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
            add_activation(activation),
        )
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(
                decoder_base_channels, input_channels, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            add_activation(output_activation),
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
        encoder_activation: ActivationName = "relu",
        decoder_activation: ActivationName = "relu",
        encoder_output_activation: ActivationName = "relu",
        decoder_output_activation: ActivationName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super(SimpleCAE64, self).__init__()
        self.device = device

        self.encoder = Encoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_activation,
            encoder_output_activation,
            device,
        )

        self.decoder = Decoder(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_activation,
            decoder_output_activation,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z


if __name__ == "__main__":
    # Third Party Library
    from torchinfo import summary

    model = SimpleCAE64(1, 16, 16, 10, device="cuda")

    summary(model, (1, 1, 64, 64))
