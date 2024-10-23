# Standard Library
from collections.abc import Callable, Generator
from typing import Literal, Union

# Third Party Library
import torch
from torch import nn

from src.mytyping import ActivationName, Tensor


def conv2d3x3(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    how_sampling: Literal["down", "up"] = "down",
) -> nn.Conv2d | nn.ConvTranspose2d:
    """Convolution layer with kernel size of 3x3

    Parameters
    ----------
    in_ch : int
        Number of channels in the input image
    out_ch : int
        Number of channels in the output image
    stride : int, optional
        Stride of the convolution. Default: 1
    how_sampling : Literal["down", "up"], optional
        Either down-sampling or up-sampling, by default "down"

    Returns
    -------
    nn.Conv2d | nn.ConvTranspose2d
        _description_
    """
    if how_sampling == "down":
        return nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
    elif how_sampling == "up":
        return nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            output_padding=0 if stride < 2 else 1,
        )


def conv2d1x1(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    how_sampling: Literal["down", "up"] = "down",
) -> nn.Conv2d | nn.ConvTranspose2d:
    """Convolution layer with kernel size of 1x1

    Parameters
    ----------
    in_ch : int
        Number of channels in the input image
    out_ch : int
        Number of channels in the output image
    stride : int, optional
        Stride of the convolution. Default: 1
    how_sampling : bool, optional
        Whether to use ConvTranspose or not. Default: False

    Returns
    -------
    nn.Conv2d | nn.ConvTranspose2d
        2D convolution layer with a kernel size of 1 x 1
    """
    if how_sampling == "down":
        return nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=stride, bias=False
        )
    elif how_sampling == "up":
        return nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=stride,
            bias=False,
            output_padding=0 if stride < 2 else 1,
        )


def add_activation(
    activation: Literal[
        "relu", "selu", "leakyrelu", "sigmoid", "tanh", "identity"
    ] = "relu",
) -> Union[nn.ReLU, nn.SELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Identity]:
    """Add the specified activation function

    Parameters
    ----------
    activation : str, optional
        Name of the activation function you wish to specify, by default "relu"
        Supported activation functions:
            ReLU: "relu"
            SELU: "selu"
            LeakyReLU: "leakyrelu"
            Sigmoid: "sigmoid"
            Tanh: "tanh"
            Identity: "identity"

    Returns
    -------
    Union[nn.ReLU, nn.SELU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.Identity]
        Layer of the specified activation function

    Raises
    ------
    RuntimeError
        Caused when the function name does not match the supported activation
        function name.
    """
    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "selu":
        return nn.SELU(True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(0.02, True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "identity":
        return nn.Identity()
    else:
        raise RuntimeError(
            f'There is no activation function such as "{activation}"'
        )


class MyBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: ActivationName = "relu",
        how_sampling: Literal["down", "up"] = "down",
        *,
        expansion: int | None = None,
    ) -> None:
        super().__init__()
        if expansion is not None:
            self.expansion = expansion
        self.conv1 = conv2d3x3(in_ch, out_ch, stride, how_sampling)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv2d3x3(out_ch, out_ch, how_sampling=how_sampling)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch, stride, how_sampling),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Sequential()
        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.conv1(x)
        h = self.bn1(h)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.bn2(h)

        h += self.shortcut(x)

        h = self.activation(h)
        return h


class MyBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: ActivationName = "relu",
        how_sampling: Literal["down", "up"] = "down",
        *,
        expansion: int | None = None,
    ) -> None:
        super().__init__()
        if expansion is not None:
            self.expansion = expansion
        if how_sampling == "down":
            mid_ch = out_ch // self.expansion
        elif how_sampling == "up":
            mid_ch = out_ch * self.expansion
        self.conv1 = conv2d1x1(in_ch, mid_ch, how_sampling=how_sampling)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = conv2d3x3(mid_ch, mid_ch, stride, how_sampling)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = conv2d1x1(mid_ch, out_ch, how_sampling=how_sampling)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.activation = add_activation(activation)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch, stride, how_sampling),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.conv1(x)
        h = self.bn1(h)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.activation(h)

        h = self.conv3(h)
        h = self.bn3(h)

        h += self.shortcut(x)

        h = self.activation(h)

        return h


class DownShape(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, activation: ActivationName = "relu"
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.conv1(x)
        h = self.bn1(h)
        h = self.activation(h)

        return h


class UpShape(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, activation: ActivationName = "leakyrelu"
    ) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.conv1(x)
        h = self.bn1(h)
        h = self.activation(h)

        return h


class SELayer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        reduction: int = 16,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_ch, in_ch // reduction)
        self.actfunc = add_activation(activation)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        bs, c, _, _ = x.size()
        out: Tensor = self.gap(x)
        out = self.fc1(out.view(bs, -1))
        out = self.actfunc(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out.view(bs, c, 1, 1).expand_as(x)


class SEBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: ActivationName = "relu",
        how_sampling: Literal["down", "up"] = "down",
        *,
        expansion: int = 4,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.expansion = expansion
        if how_sampling == "down":
            mid_ch = out_ch // self.expansion
        elif how_sampling == "up":
            mid_ch = out_ch * self.expansion
        self.residual = nn.Sequential(
            conv2d1x1(in_ch, mid_ch, how_sampling=how_sampling),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d3x3(mid_ch, mid_ch, stride, how_sampling),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d1x1(mid_ch, out_ch, how_sampling=how_sampling),
            nn.BatchNorm2d(out_ch),
        )

        self.se_layer = SELayer(out_ch, reduction, activation)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch, stride, how_sampling),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.residual(x)
        h = self.shortcut(x) + self.se_layer(h)
        h = self.activation(h)
        return h


def _inplanes(
    plane: int, coeff: int, expansion: bool, lim: int | None = None
) -> Generator[int, None, None]:
    i = 0
    while lim is None or i < lim:
        yield plane
        if expansion:
            plane *= coeff
        else:
            plane //= coeff
        i += 1


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

        self.inplaneses = tuple(_inplanes(inplanes, 2, True, 5))
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
        norm_layer: Callable[..., nn.Module] | None = None,
        *,
        reconstructed_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplaneses = tuple(_inplanes(inplanes, 2, True, 5))
        self.reconstructed_size = reconstructed_size

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
        print(x.shape)

        x = self.layer4_t(x)
        print(x.shape)
        x = self.layer3_t(x)
        print(x.shape)
        x = self.layer2_t(x)
        print(x.shape)
        x = self.layer1_t(x)
        print(x.shape)

        x = self.maxpool_t(x, pool_indices, input_size_conv1_t)
        print(x.shape)
        x = self.actfunc(x)
        x = self.bn1_t(x)
        x = self.conv1_t(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.actfunc(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(
        self,
        x: Tensor,
        input_size_conv1_t: tuple[int, ...],
        pool_indices: Tensor,
    ) -> Tensor:
        return self._forward_impl(x, input_size_conv1_t, pool_indices)

    @staticmethod
    def _inplanes(plane: int, expansion: int) -> Generator[int, None, None]:
        while True:
            yield plane
            plane *= expansion


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
