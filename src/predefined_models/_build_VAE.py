import torch
from torch import nn

from ..configs.model_configs.autoencoder_configs.v2 import _VAEModelConfig
from ..configs.model_configs.base_configs import NetworkConfig
from ..mytyping import Tensor
from . import _ResNetVAE as resmod
from ._load_model import LoadModel


class VAEFrame(nn.Module):
    def __init__(
        self,
        encoder: NetworkConfig,
        latent_mean: NetworkConfig,
        latent_var: NetworkConfig,
        decoder: NetworkConfig,
    ) -> None:
        super().__init__()
        self.encoder = LoadModel.from_config(encoder)
        self.latent_mean = LoadModel.from_config(latent_mean)
        self.latent_var = LoadModel.from_config(latent_var)
        self.decoder = LoadModel.from_config(decoder)

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

    @classmethod
    def build_from_config(cls, cfg: _VAEModelConfig) -> "VAEFrame":
        if cfg.decoder.network_type == "UpSamplingResNet":
            if cfg.encoder.network_type != "DownSamplingResNet":
                raise ValueError(
                    "If `DownSamplingResNet is chosen as decoder, "
                    "encoder must be `DownSamplingResNet`"
                )

        return cls(cfg.encoder, cfg.latent_mean, cfg.latent_var, cfg.decoder)
