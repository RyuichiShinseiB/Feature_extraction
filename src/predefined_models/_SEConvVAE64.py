# Third Party Library
import torch
from torch import nn

# Local Library
from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import DownShape, SELayer, UpShape, add_actfunc


class VariationalSEEncoder(nn.Module):
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
                encoder_base_channels * 32,
                actfunc,
            ),
            SELayer(encoder_base_channels * 32),
        )
        self.l_mean = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimensions, bias=False
            ),
            add_actfunc(output_actfunc),
        )
        self.l_logvar = nn.Linear(
            encoder_base_channels * 32, latent_dimensions
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        h = torch.flatten(h, start_dim=1)
        mean = self.l_mean(h)
        logvar = self.l_logvar(h)

        noise = torch.randn_like(logvar, device=self.device)
        z = mean + noise * torch.exp(0.5 * logvar)

        return z, mean, logvar


######################################################
class VariationalSEDecoder(nn.Module):
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
        self.l6 = UpShape(
            decoder_base_channels, input_channels, output_actfunc
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
            decoder_base_channels * 2, decoder_base_channels, actfunc
        )
        self.l6 = UpShape(
            decoder_base_channels, input_channels, output_actfunc
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
class SECVAE64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dimensions: int = 10,
        encoder_base_channels: int = 64,
        decoder_base_channels: int = 64,
        encoder_actfunc: ActFuncName = "relu",
        decoder_actfunc: ActFuncName = "relu",
        encoder_output_actfunc: ActFuncName = "sigmoid",
        decoder_output_actfunc: ActFuncName = "sigmoid",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dimensions = latent_dimensions
        self.encoder = VariationalSEEncoder(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VariationalSEDecoder(
            latent_dimensions,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
            device,
        )

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        z, mean, logvar = self.encoder(x)
        x_pred = self.decoder(z.view(x.shape[0], self.latent_dimensions, 1, 1))

        return x_pred, (z, mean, logvar)
