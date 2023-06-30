# Third Party Library
import torch
from torch import nn

# Local Library
from .. import ActivationName, Device, Tensor
from ._modules import DownShape, SELayer, UpShape, add_activation


class VariationalSEEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimension: int = 10,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        if output_activation is None:
            output_activation = activation
        self.device = device
        self.l1 = nn.Sequential(
            DownShape(input_channels, encoder_base_channels, activation),
            SELayer(encoder_base_channels),
        )
        self.l2 = nn.Sequential(
            DownShape(
                encoder_base_channels, encoder_base_channels * 2, activation
            ),
            SELayer(encoder_base_channels * 2),
        )
        self.l3 = nn.Sequential(
            DownShape(
                encoder_base_channels * 2,
                encoder_base_channels * 4,
                activation,
            ),
            SELayer(encoder_base_channels * 4),
        )
        self.l4 = nn.Sequential(
            DownShape(
                encoder_base_channels * 4,
                encoder_base_channels * 8,
                activation,
            ),
            SELayer(encoder_base_channels * 8),
        )
        self.l5 = nn.Sequential(
            DownShape(
                encoder_base_channels * 8,
                encoder_base_channels * 16,
                activation,
            ),
            SELayer(encoder_base_channels * 16),
        )
        self.l6 = nn.Sequential(
            DownShape(
                encoder_base_channels * 16,
                encoder_base_channels * 32,
                activation,
            ),
            SELayer(encoder_base_channels * 32),
        )
        self.l_mean = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimension, bias=False
            ),
            add_activation(output_activation),
        )
        self.l_var = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimension, bias=False
            ),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h: Tensor = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        h = torch.flatten(h, start_dim=1)
        mean = self.l_mean(h)
        var = self.l_var(h)

        return mean, var


######################################################
class VariationalSEDecoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = nn.Sequential(
            UpShape(latent_dimension, decoder_base_channels * 16, activation),
            SELayer(decoder_base_channels * 16),
        )
        self.l2 = nn.Sequential(
            UpShape(
                decoder_base_channels * 16,
                decoder_base_channels * 8,
                activation,
            ),
            SELayer(decoder_base_channels * 8),
        )
        self.l3 = nn.Sequential(
            UpShape(
                decoder_base_channels * 8,
                decoder_base_channels * 4,
                activation,
            ),
            SELayer(decoder_base_channels * 4),
        )
        self.l4 = nn.Sequential(
            UpShape(
                decoder_base_channels * 4,
                decoder_base_channels * 2,
                activation,
            ),
            SELayer(decoder_base_channels * 2),
        )
        self.l5 = nn.Sequential(
            UpShape(
                decoder_base_channels * 2, decoder_base_channels, activation
            ),
            SELayer(decoder_base_channels),
        )
        self.l6 = UpShape(
            decoder_base_channels, input_channels, output_activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


######################################################
class VariationalDecoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.l1 = UpShape(
            latent_dimension, decoder_base_channels * 16, activation
        )
        self.l2 = UpShape(
            decoder_base_channels * 16, decoder_base_channels * 8, activation
        )
        self.l3 = UpShape(
            decoder_base_channels * 8, decoder_base_channels * 4, activation
        )
        self.l4 = UpShape(
            decoder_base_channels * 4, decoder_base_channels * 2, activation
        )
        self.l5 = UpShape(
            decoder_base_channels * 2, decoder_base_channels, activation
        )
        self.l6 = UpShape(
            decoder_base_channels, input_channels, output_activation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


###################################################################
class SECVAEsoftplus64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dimension: int = 10,
        encoder_base_channels: int = 64,
        decoder_base_channels: int = 64,
        encoder_activation: ActivationName = "relu",
        decoder_activation: ActivationName = "relu",
        encoder_output_activation: ActivationName = "sigmoid",
        decoder_output_activation: ActivationName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dimension = latent_dimension
        self.encoder = VariationalSEEncoder(
            input_channels,
            encoder_base_channels,
            latent_dimension,
            encoder_activation,
            encoder_output_activation,
            device,
        )
        self.decoder = VariationalSEDecoder(
            latent_dimension,
            decoder_base_channels,
            input_channels,
            decoder_activation,
            decoder_output_activation,
            device,
        )

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_pred = self.decoder(z)

        return x_pred, (z, mean, var)

    def reparameterization(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=self.device)
        return (mean + eps * torch.sqrt(var)).view(
            mean.shape[0], self.latent_dimension, 1, 1
        )
