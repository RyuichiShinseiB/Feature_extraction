import torch.nn as nn

from ..mytyping import ActivationName, Device
from ._CNN_modules import MyBasicBlock, ResNet


class ResNetEncoder(nn.Module):
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
        self.resnet = ResNet(
            MyBasicBlock,
            layers=(3, 4, 6, 3),
            in_ch=input_channels,
            out_ch=latent_dimensions,
            activation=activation,
        )
        pass
