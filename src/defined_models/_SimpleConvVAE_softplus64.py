# Third Party Library
import torch
from torch import nn

# Local Library
from ._modules import (
    ActivationName,
    Device,
    DownShape,
    Tensor,
    UpShape,
    add_activation,
)


class VariationalEncoder(nn.Module):
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

        self.l1 = DownShape(input_channels, encoder_base_channels, activation)
        self.l2 = DownShape(
            encoder_base_channels, encoder_base_channels * 2, activation
        )
        self.l3 = DownShape(
            encoder_base_channels * 2, encoder_base_channels * 4, activation
        )
        self.l4 = DownShape(
            encoder_base_channels * 4, encoder_base_channels * 8, activation
        )
        self.l5 = DownShape(
            encoder_base_channels * 8, encoder_base_channels * 16, activation
        )
        self.l6 = DownShape(
            encoder_base_channels * 16, encoder_base_channels * 32, activation
        )
        self.l_mean = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimensions, bias=False
            ),
            add_activation(output_activation),
        )
        self.l_var = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimensions, bias=False
            ),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.l1(x)
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
class VariationalDecoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        self.l1 = UpShape(
            latent_dimensions, decoder_base_channels * 16, activation
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
class SimpleCVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dimensions: int = 10,
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
        self.latent_dimensions = latent_dimensions
        self.encoder = VariationalEncoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_activation,
            encoder_output_activation,
            device,
        )
        self.decoder = VariationalDecoder(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_activation,
            decoder_output_activation,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_pred = self.decoder(z)

        return x_pred, z

    def reparameterization(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=self.device)
        return (mean + eps * torch.sqrt(var)).view(
            mean.shape[0], self.latent_dimensions, 1, 1
        )

    def lower_bound(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_pred = self.decoder(z)

        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(x_pred, start_dim=1)

        eps = 1e-3

        reconst_loss = -torch.mean(
            torch.sum(
                x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps),
                dim=1,
            )
        )
        latent_loss = -0.5 * torch.mean(
            torch.sum(1 + torch.log(var + eps) - mean**2 - var, dim=1).view(
                -1
            )
        )
        loss = reconst_loss + latent_loss

        return loss, x_pred
