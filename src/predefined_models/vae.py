import torch
from torch import nn

from ..mytyping import ActFuncName, Device, Tensor
from ._CNN_modules import DownShape, SELayer, UpShape, add_actfunc


class VAEEncoder32(nn.Module):
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
class VAEDecoder32(nn.Module):
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
        self.encoder = VAEEncoder32(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VAEDecoder32(
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


###########################


class VAEEncoderSoftplus64(nn.Module):
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
        self.l6 = DownShape(
            encoder_base_channels * 16, encoder_base_channels * 32, actfunc
        )
        self.l_mean = nn.Sequential(
            nn.Linear(
                encoder_base_channels * 32, latent_dimensions, bias=False
            ),
            add_actfunc(output_actfunc),
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


class VAEEncoder64(nn.Module):
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
        self.l6 = DownShape(
            encoder_base_channels * 16, encoder_base_channels * 32, actfunc
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = self.l6(h)
        h = torch.flatten(h, start_dim=1)
        mean = self.l_mean(h)
        log_var = self.l_logvar(h)

        return mean, log_var


######################################################
class VAEDecoder64(nn.Module):
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
class SimpleCVAEsoftplus64(nn.Module):
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
        self.encoder = VAEEncoderSoftplus64(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VAEDecoder64(
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
        mean, var = self.encoder(x)
        z = self.reparameterization(mean, var)
        x_pred = self.decoder(z)

        return x_pred, (z, mean, var)

    def reparameterization(self, mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=self.device)
        return (mean + eps * torch.sqrt(var)).view(
            mean.shape[0], self.latent_dimensions, 1, 1
        )


class SimpleCVAE64(nn.Module):
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
        self.encoder = VAEEncoder64(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VAEDecoder64(
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
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_pred = self.decoder(z)

        return x_pred, (z, mean, log_var)

    def reparameterization(self, mean: Tensor, log_var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=self.device)
        return (mean + eps * log_var.mul(0.5).exp()).view(
            mean.shape[0], self.latent_dimensions, 1, 1
        )


##################


class VAESEEncoder64(nn.Module):
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


class VAESEEncoderSoftplus64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimension: int = 10,
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
                encoder_base_channels * 32, latent_dimension, bias=False
            ),
            add_actfunc(output_actfunc),
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
class VAESEDecoder64(nn.Module):
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
        self.encoder = VAESEEncoder64(
            input_channels,
            encoder_base_channels,
            latent_dimensions,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VAESEDecoder64(
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


class SECVAEsoftplus64(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dimension: int = 10,
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
        self.latent_dimension = latent_dimension
        self.encoder = VAESEEncoderSoftplus64(
            input_channels,
            encoder_base_channels,
            latent_dimension,
            encoder_actfunc,
            encoder_output_actfunc,
            device,
        )
        self.decoder = VAESEDecoder64(
            latent_dimension,
            decoder_base_channels,
            input_channels,
            decoder_actfunc,
            decoder_output_actfunc,
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
