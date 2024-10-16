import torch.nn as nn

from ..mytyping import ActivationName, Device, Tensor
from ._CNN_modules import MyBasicBlock, ResNet, add_activation


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        encoder_base_channels: int = 64,
        latent_dimensions: int = 10,
        activation: ActivationName = "relu",
        mean_layer_activation: ActivationName = "relu",
        var_layer_activation: ActivationName = "relu",
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.resnet = ResNet(
            block=MyBasicBlock,
            layers=(3, 4, 6, 3),
            in_ch=input_channels,
            out_ch=latent_dimensions,
            inplanes=encoder_base_channels,
            activation=activation,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(activation),
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(mean_layer_activation),
        )

        self.var_layer = nn.Sequential(
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(activation),
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(var_layer_activation),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.resnet(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)

        return mean, var
