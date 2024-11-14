from collections.abc import Callable, Generator
from typing import Literal

import torch
from torch import nn

from ..mytyping import ActivationName, Device, ResNetBlockName, Tensor
from ._CNN_modules import (
    MyBasicBlock,
    MyBottleneck,
    SEBottleneck,
    add_activation,
)


def _make_resnet_layer(
    block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
    in_ch: int,
    out_ch: int,
    num_blocks: int,
    stride: int = 1,
    activation: ActivationName = "relu",
    how_sampling: Literal["down", "up"] = "down",
) -> nn.Sequential:
    layers = []
    layers.append(block(in_ch, out_ch, stride, activation, how_sampling))
    for _ in range(1, num_blocks):
        layers.append(
            block(
                out_ch,
                out_ch,
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


class DownSamplingResNet(nn.Module):
    def __init__(
        self,
        block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
        layers: tuple[int, int, int, int],
        in_ch: int = 1,
        out_ch: int = 1000,
        inplanes: int = 64,
        zero_init_residual: bool = False,
        activation: ActivationName = "relu",
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplaneses = tuple(
            [inplanes] + list(_inplanes(inplanes * block.expansion, 2, 4))
        )
        self.conv1 = nn.Conv2d(
            in_ch,
            self.inplaneses[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(inplanes)
        self.actfunc = add_activation(activation)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=True
        )
        self.layer1 = _make_resnet_layer(
            block,
            self.inplaneses[0],
            self.inplaneses[1],
            layers[0],
            activation=activation,
        )
        self.layer2 = _make_resnet_layer(
            block,
            self.inplaneses[1],
            self.inplaneses[2],
            layers[1],
            stride=2,
            activation=activation,
        )
        self.layer3 = _make_resnet_layer(
            block,
            self.inplaneses[2],
            self.inplaneses[3],
            layers[2],
            stride=2,
            activation=activation,
        )
        self.layer4 = _make_resnet_layer(
            block,
            self.inplaneses[3],
            self.inplaneses[4],
            layers[3],
            stride=2,
            activation=activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplaneses[4], out_ch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.  # noqa: E501
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MyBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MyBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(
        self, x: Tensor
    ) -> tuple[Tensor, tuple[int, ...], Tensor]:
        x = self.conv1(x)
        output_size_conv1 = x.shape
        x = self.bn1(x)
        x = self.actfunc(x)
        x, pool_indices = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, output_size_conv1, pool_indices

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, ...], Tensor]:
        return self._forward_impl(x)


class UpSamplingResNet(nn.Module):
    def __init__(
        self,
        block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
        layers: tuple[int, int, int, int],
        in_ch: int = 1000,
        out_ch: int = 1,
        inplanes: int = 64,
        zero_init_residual: bool = False,
        activation: ActivationName = "relu",
        output_activation: ActivationName = "sigmoid",
        norm_layer: Callable[..., nn.Module] | None = None,
        *,
        reconstructed_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplaneses = tuple(
            [inplanes] + list(_inplanes(inplanes * block.expansion, 2, 4))
        )
        self.reconstructed_size = reconstructed_size

        self.output_actfunc = add_activation(output_activation)
        self.conv1_t = nn.ConvTranspose2d(
            self.inplaneses[0],
            out_ch,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=1,
        )
        self.bn1_t = norm_layer(self.inplaneses[0])
        self.actfunc = add_activation(activation)
        self.maxpool_t = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_t = _make_resnet_layer(
            block,
            self.inplaneses[1],
            self.inplaneses[0],
            layers[0],
            activation=activation,
            how_sampling="up",
        )
        self.layer2_t = _make_resnet_layer(
            block,
            self.inplaneses[2],
            self.inplaneses[1],
            layers[1],
            stride=2,
            activation=activation,
            how_sampling="up",
        )
        self.layer3_t = _make_resnet_layer(
            block,
            self.inplaneses[3],
            self.inplaneses[2],
            layers[2],
            stride=2,
            activation=activation,
            how_sampling="up",
        )
        self.layer4_t = _make_resnet_layer(
            block,
            self.inplaneses[4],
            self.inplaneses[3],
            layers[3],
            stride=2,
            activation=activation,
            how_sampling="up",
        )

        before_avgpool_size = _calc_layers_output_size(
            reconstructed_size,
            # (conv1, maxpool, layer1, layer2, layer3, layer4)
            kernel_size=(7, 3, 3, 3, 3, 3),
            padding=(3, 1, 1, 1, 1, 1),
            dilation=(1, 1, 1, 1, 1, 1),
            stride=(2, 2, 1, 2, 2, 2),
        )
        self.avgpool_t = nn.UpsamplingNearest2d(before_avgpool_size)
        self.fc = nn.Linear(in_ch, self.inplaneses[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.  # noqa: E501
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MyBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, MyBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(
        self,
        x: Tensor,
        input_size_conv1_t: tuple[int, ...],
        pool_indices: Tensor,
    ) -> Tensor:
        x = self.fc(x)
        x = self.actfunc(x)
        x = self.avgpool_t(x.view(x.shape[0], -1, 1, 1))

        x = self.layer4_t(x)
        x = self.layer3_t(x)
        x = self.layer2_t(x)
        x = self.layer1_t(x)

        x = self.maxpool_t(x, pool_indices, input_size_conv1_t)
        x = self.actfunc(x)
        x = self.bn1_t(x)
        x = self.conv1_t(x)
        x = self.output_actfunc(x)

        return x

    def forward(
        self,
        x: Tensor,
        input_size_conv1_t: tuple[int, ...],
        pool_indices: Tensor,
    ) -> Tensor:
        return self._forward_impl(x, input_size_conv1_t, pool_indices)


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
        input_size: tuple[int, int],
        device: Device,
        block_name: ResNetBlockName | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.latent_dimensions = latent_dimensions

        self.encoder = DownSamplingResNet(
            block=_set_resnet_block(block_name),
            layers=(3, 4, 6, 3),
            in_ch=input_channels,
            out_ch=latent_dimensions,
            inplanes=encoder_base_channels,
            activation=encoder_activation,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(encoder_activation),
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(encoder_output_activation),
        )
        self.var_layer = nn.Sequential(
            nn.Linear(latent_dimensions, latent_dimensions),
            add_activation(encoder_activation),
            nn.Linear(latent_dimensions, latent_dimensions),
            nn.Softplus(),
        )

        self.decoder = UpSamplingResNet(
            block=_set_resnet_block(block_name),
            layers=(3, 4, 6, 3),
            in_ch=latent_dimensions,
            out_ch=input_channels,
            inplanes=decoder_base_channels,
            activation=decoder_activation,
            output_activation=decoder_output_activation,
            reconstructed_size=input_size,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        x, output_size_conv1, indices = self.encoder.forward(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        z = self.reparametarization(mean, var)

        x = self.decoder(z, output_size_conv1, indices)
        return x, (mean, var)

    @staticmethod
    def reparametarization(mean: Tensor, var: Tensor) -> Tensor:
        eps = torch.randn(mean.shape, device=mean.device)
        return mean + var.sqrt() * eps

    def estimate_distribution_params(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x, _, _ = self.encoder.forward(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        return mean, var
