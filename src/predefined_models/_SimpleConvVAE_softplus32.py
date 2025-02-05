# Third Party Library
import torch
from torch import nn

# Local Library
from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import DownShape, UpShape, add_actfunc


class VariationalEncoder(nn.Module):
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

        self.l1 = DownShape(input_channels, encoder_base_channels, actfunc)
        self.l2 = DownShape(
            encoder_base_channels, encoder_base_channels * 2, actfunc
        )
        self.l3 = DownShape(
            encoder_base_channels * 2, encoder_base_channels * 4, actfunc
        )
        self.l4 = DownShape(
            encoder_base_channels * 4, encoder_base_channels * 8, actfunc
        )
        self.l5 = DownShape(
            encoder_base_channels * 8, encoder_base_channels * 16, actfunc
        )
        self.l6 = nn.Sequential(
            nn.Linear(encoder_base_channels * 16, latent_dimensions),
            add_actfunc(actfunc),
        )

        self.l7_mean = nn.Sequential(
            nn.Linear(
                latent_dimensions,
                latent_dimensions,
            ),
            add_actfunc(output_actfunc),
        )
        self.l7_var = nn.Sequential(
            nn.Linear(
                latent_dimensions,
                latent_dimensions,
            ),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = torch.flatten(h, start_dim=1)
        h = self.l6(h)
        mean = self.l7_mean(h)
        var = self.l7_var(h)

        return mean, var


######################################################
class VariationalDecoder(nn.Module):
    def __init__(
        self,
        latent_dimensions: int = 10,
        decoder_base_channels: int = 64,
        input_channels: int = 1,
        actfunc: ActFuncName = "relu",
        output_actfunc: ActFuncName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        self.l1 = UpShape(
            latent_dimensions, decoder_base_channels * 16, actfunc
        )
        self.l2 = UpShape(
            decoder_base_channels * 16, decoder_base_channels * 8, actfunc
        )
        self.l3 = UpShape(
            decoder_base_channels * 8, decoder_base_channels * 4, actfunc
        )
        self.l4 = UpShape(
            decoder_base_channels * 4, decoder_base_channels * 2, actfunc
        )
        self.l5 = UpShape(
            decoder_base_channels * 2, input_channels, output_actfunc
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x


###################################################################
class SimpleCVAEsoftplus32(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dimensions: int = 10,
        encoder_base_channels: int = 64,
        decoder_base_channels: int = 64,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "sigmoid",
        decoder_output_actfunc: ActFuncName = "tanh",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dimensions = latent_dimensions
        self.encoder = VariationalEncoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VariationalDecoder(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_pred = self.decoder(z.view(-1, z.shape[1], 1, 1))

        return x_pred, (mean, var)

    def reparameterization(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=self.device)
        return mean + eps * var.sqrt()


if __name__ == "__main__":
    # Third Party Library
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCVAEsoftplus32(device=device)
    summary(model, (1, 1, 32, 32))
