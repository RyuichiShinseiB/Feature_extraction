from collections.abc import Callable, Generator
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..mytyping import ActivationName, Device, ResNetBlockName, Tensor
from ._CNN_modules import (
    MyBasicBlock,
    MyBottleneck,
    SEBottleneck,
    add_activation,
    conv2d3x3,
)


def _make_resnet_layer(
    block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
    input_channels: int,
    output_channels: int,
    num_blocks: int,
    stride: int = 1,
    activation: ActivationName = "relu",
    how_sampling: Literal["down", "up"] = "down",
) -> nn.Sequential:
    layers = []
    layers.append(
        block(
            input_channels, output_channels, stride, activation, how_sampling
        )
    )
    for _ in range(1, num_blocks):
        layers.append(
            block(
                output_channels,
                output_channels,
                activation=activation,
                how_sampling=how_sampling,
            )
        )
    return nn.Sequential(*layers)


def _inplanes(
    plane: int, coeff: int, lim: int | None = None
) -> Generator[int, None, None]:
    i = 0
    while lim is None or i < lim:
        yield plane
        plane *= coeff
        i += 1


def _calc_layers_output_size(
    size: tuple[int, int] | int,
    kernel_size: tuple[tuple[int, int] | int, ...],
    padding: tuple[tuple[int, int] | int, ...],
    dilation: tuple[tuple[int, int] | int, ...],
    stride: tuple[tuple[int, int] | int, ...],
) -> tuple[int, int]:
    _size = torch.tensor(size)
    _kernel_size = torch.tensor(kernel_size)
    _padding = torch.tensor(padding)
    _dilation = torch.tensor(dilation)
    _stride = torch.tensor(stride)
    for k, p, d, s in zip(_kernel_size, _padding, _dilation, _stride):
        _tempsize = (_size + 2 * p - d * (k - 1) - 1) / s + 1
        _size = torch.floor(_tempsize)

    return (int(_size[0].item()), int(_size[1].item()))


def _set_resnet_block(
    block_name: ResNetBlockName | None,
) -> type[MyBasicBlock | MyBottleneck | SEBottleneck]:
    if block_name == "basicblock" or block_name is None:
        return MyBasicBlock
    elif block_name == "bottleneck":
        return MyBottleneck
    elif block_name == "sebottleneck":
        return SEBottleneck
    else:
        raise RuntimeError(f"Not implemente: {block_name}")


def _init_weights(module: nn.Module, zero_init_residual: bool) -> None:
    """Weight initialization

    In first half, conv and norm layers were initialized.

    In second half, Zero-initialize the last BN in each residual branch,
    so that the residual branch starts with zeros, and each residual block behaves like an identity.
    This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    Parameters
    ----------
    zero_init_residual : bool
        _description_
    module : nn.Module
        _description_
    """  # noqa: E501
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual:
        for m in module.modules():
            if isinstance(m, MyBottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, MyBasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)


class DownSamplingResNet(nn.Module):
    def __init__(
        self,
        block_name: ResNetBlockName | None,
        layers: tuple[int, int, int, int],
        input_channels: int = 1,
        output_channels: int = 1000,
        inplanes: int = 64,
        zero_init_residual: bool = False,
        activation: ActivationName = "relu",
        norm_layer: Callable[..., nn.Module] | None = None,
        resolution: int = 32,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        block = _set_resnet_block(block_name)
        self.inplaneses = (inplanes,) + tuple(
            _inplanes(inplanes * block.expansion, 2, 4)
        )
        self.num_donsampling = 6
        self.each_step_resolution: list[int] = [resolution]

        # calculate the resolution of the tensor after down sampling.
        for i in range(self.num_donsampling):
            prev_resolusion = self.each_step_resolution[-1]
            if i == 2:
                next_resolution = prev_resolusion
            else:
                next_resolution = (prev_resolusion + 1) // 2
            self.each_step_resolution.append(next_resolution)

        self.conv1 = nn.Conv2d(
            input_channels,
            self.inplaneses[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(inplanes)
        self.actfunc = add_activation(activation)

        # self.maxpool = nn.MaxPool2d(
        #     kernel_size=3, stride=2, padding=1, return_indices=True
        # )
        self.downsample = nn.Sequential(
            conv2d3x3(inplanes, inplanes, stride=2, how_sampling="down"),
            nn.BatchNorm2d(inplanes),
            add_activation(activation),
        )

        self.resblocks1 = _make_resnet_layer(
            block,
            self.inplaneses[0],
            self.inplaneses[1],
            layers[0],
            activation=activation,
        )

        self.resblocks2 = _make_resnet_layer(
            block,
            self.inplaneses[1],
            self.inplaneses[2],
            layers[1],
            stride=2,
            activation=activation,
        )

        self.resblocks3 = _make_resnet_layer(
            block,
            self.inplaneses[2],
            self.inplaneses[3],
            layers[2],
            stride=2,
            activation=activation,
        )

        self.resblocks4 = _make_resnet_layer(
            block,
            self.inplaneses[3],
            self.inplaneses[4],
            layers[3],
            stride=2,
            activation=activation,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplaneses[4], output_channels)

        _init_weights(self, zero_init_residual)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actfunc(x)
        # x, pool_indices = self.maxpool(x)
        x = self.downsample(x)

        x = self.resblocks1(x)
        x = self.resblocks2(x)
        x = self.resblocks3(x)
        x = self.resblocks4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.actfunc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """forwarding of DownSamplingResNet

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        feature: Tensor
            Feature map obtained from the input tensor
        """
        return self._forward_impl(x)


class UpSamplingResNet(nn.Module):
    def __init__(
        self,
        block_name: ResNetBlockName | None,
        layers: tuple[int, int, int, int],
        input_channels: int = 1000,
        output_channels: int = 1,
        inplanes: int = 64,
        zero_init_residual: bool = False,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "sigmoid",
        norm_layer: Callable[..., nn.Module] | None = None,
        *,
        resolution: int = 64,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.num_upsampling = 6
        self.each_step_resolution = [resolution]
        # calculate the resolution of the tensor after down sampling.
        for i in range(self.num_upsampling):
            prev_resolusion = self.each_step_resolution[-1]
            if i == 2:
                next_resolution = prev_resolusion
            else:
                next_resolution = (prev_resolusion + 1) // 2
            self.each_step_resolution.append(next_resolution)

        block = _set_resnet_block(block_name)

        self.inplaneses = (inplanes,) + tuple(
            _inplanes(inplanes * block.expansion, 2, 4)
        )

        self.output_actfunc = add_activation(output_activation)
        self.conv1_t = nn.ConvTranspose2d(
            self.inplaneses[0],
            output_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=1,
        )
        self.bn1_t = norm_layer(self.inplaneses[0])
        self.actfunc = add_activation(activation)
        self.downsampling_t = conv2d3x3(
            self.inplaneses[0], self.inplaneses[0], stride=2, how_sampling="up"
        )
        self.resblock1_t = _make_resnet_layer(
            block,
            self.inplaneses[1],
            self.inplaneses[0],
            layers[0],
            activation=activation,
            how_sampling="up",
        )
        self.resblock2_t = _make_resnet_layer(
            block,
            self.inplaneses[2],
            self.inplaneses[1],
            layers[1],
            stride=2,
            activation=activation,
            how_sampling="up",
        )
        self.resblock3_t = _make_resnet_layer(
            block,
            self.inplaneses[3],
            self.inplaneses[2],
            layers[2],
            stride=2,
            activation=activation,
            how_sampling="up",
        )
        self.resblock4_t = _make_resnet_layer(
            block,
            self.inplaneses[4],
            self.inplaneses[3],
            layers[3],
            stride=2,
            activation=activation,
            how_sampling="up",
        )

        self.avgpool_t = nn.UpsamplingNearest2d(self.each_step_resolution[-1])
        self.fc = nn.Linear(input_channels, self.inplaneses[4])

        _init_weights(self, zero_init_residual)

    def _forward_impl(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.fc(x)
        x = self.actfunc(x)
        x = self.avgpool_t(x.view(x.shape[0], -1, 1, 1))

        x = self.resblock4_t(x)
        if x.shape[-1] == self.each_step_resolution[5]:
            scale_factor = self.each_step_resolution[5] / x.shape[-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.resblock3_t(x)
        if x.shape[-1] == self.each_step_resolution[4]:
            scale_factor = self.each_step_resolution[4] / x.shape[-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.resblock2_t(x)
        if x.shape[-1] == self.each_step_resolution[3]:
            scale_factor = self.each_step_resolution[3] / x.shape[-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.resblock1_t(x)
        if x.shape[-1] == self.each_step_resolution[2]:
            scale_factor = self.each_step_resolution[2] / x.shape[-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.downsampling_t(x)
        if x.shape[-1] == self.each_step_resolution[1]:
            scale_factor = self.each_step_resolution[1] / x.shape[-1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.actfunc(x)
        x = self.bn1_t(x)
        x = self.conv1_t(x)
        x = self.output_actfunc(x)

        assert x.shape[-1] == self.each_step_resolution[0]

        return x

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self._forward_impl(x)


###############################################################################
class ResNetVAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dimensions: int,
        encoder_base_channels: int,
        decoder_base_channels: int,
        encoder_activation: ActivationName,
        decoder_activation: ActivationName,
        encoder_output_activation: ActivationName,
        decoder_output_activation: ActivationName,
        device: Device,
        input_resolution: int = 64,
        block_name: ResNetBlockName | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dimensions = latent_dimensions

        self.encoder = DownSamplingResNet(
            block_name=block_name,
            layers=(3, 4, 6, 3),
            input_channels=input_channels,
            output_channels=latent_dimensions,
            inplanes=encoder_base_channels,
            activation=encoder_activation,
            resolution=input_resolution,
        )

        self.dit_params = nn.Sequential(
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(encoder_activation),
            nn.Linear(latent_dimensions, 2 * latent_dimensions),
        )
        self.mean_layer_activation = add_activation(encoder_output_activation)
        self.var_layer_activation = add_activation("softplus")

        self.decoder = UpSamplingResNet(
            block_name=block_name,
            layers=(3, 4, 6, 3),
            input_channels=latent_dimensions,
            output_channels=input_channels,
            inplanes=decoder_base_channels,
            activation=decoder_activation,
            output_activation=decoder_output_activation,
            resolution=input_resolution,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x = self.encoder.forward(x)
        mean, var = self.estimate_distribution_params(x)

        z = self.reparametarization(mean, var)

        x = self.decoder(z)
        return x, (mean, var)

    @staticmethod
    def reparametarization(mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=mean.device)
        return mean + var.sqrt() * eps

    def estimate_distribution_params(
        self, feature_map: Tensor
    ) -> tuple[Tensor, Tensor]:
        mean, var = self.dit_params(feature_map).chunk(2, 1)
        mean = self.mean_layer_activation(mean)
        var = self.var_layer_activation(var)
        return mean, var

    @torch.no_grad()
    def estimate_distribution_mean(self, x: Tensor) -> Tensor:
        self.eval()
        mean, _ = self.estimate_distribution_params(x)
        return mean
