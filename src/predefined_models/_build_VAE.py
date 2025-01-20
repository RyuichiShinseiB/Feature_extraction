import torch
from torch import nn

from ..mytyping import Model, Tensor
from . import _ResNetVAE as resmod


class VAEFrame(nn.Module):
    def __init__(
        self,
        encoder: Model,
        latent_mean: Model,
        latent_var: Model,
        decoder: Model,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_mean = latent_mean
        self.latent_var = latent_var
        self.decoder = decoder

        if isinstance(self.decoder, resmod.UpSamplingResNet):
            self._forward = self._resnetvae_forward
        elif isinstance(self.encoder, resmod.DownSamplingResNet):
            self._forward = self._resnet_encoder_forward
        else:
            self._forward = self._default_forward

    def _default_forward(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x = self.encoder(x)
        mean = self.latent_mean(x)
        var = self.latent_var(x)
        eps = torch.randn_like(mean)
        z: Tensor = mean + eps * var
        x = self.decoder(z)
        return x, (mean, var)

    def _resnetvae_forward(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x, size, indices = self.encoder(x)
        mean = self.latent_mean(x)
        var = self.latent_var(x)
        eps = torch.randn_like(mean)
        z: Tensor = mean + eps * var
        x = self.decoder(z, size, indices)
        return x, (mean, var)

    def _resnet_encoder_forward(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x, *_ = self.encoder(x)
        mean = self.latent_mean(x)
        var = self.latent_var(x)
        eps = torch.randn_like(mean)
        z: Tensor = mean + eps * var
        x = self.decoder(z)
        return x, (mean, var)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self._forward(x)

    @torch.no_grad()
    def evaluation(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Perform inference without computing the gradient.

        Parameters
        ----------
        x : Tensor
            Input data

        Returns
        -------
        prediction :Tensor
            Result for prediction
        (latent_mean, latent_var) : tuple[Tensor, Tensor]
            Parameters of latent variables
        """
        self.eval()
        return self._forward(x)

    @torch.no_grad()
    def generate_geometry(self, z: Tensor) -> Tensor:
        self.eval()
        z = self.decoder(z)
        return z
