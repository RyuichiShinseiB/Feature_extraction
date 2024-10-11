# Standard Library
from collections.abc import Callable
from typing import Literal, Union

# Third Party Library
import torch
from torch import nn

from src.mytyping import ActivationName, Tensor


def conv2d3x3(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    next_feature_size: Literal["down", "up"] = "down",
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
    next_feature_size : bool, optional
        Whether to use ConvTranspose or not. Default: False

    Returns
    -------
    nn.Conv2d | nn.ConvTranspose2d
        2D convolution layer with a kernel size of 3 x 3
    """
    if next_feature_size == "down":
        return nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
    elif next_feature_size == "up":
        return nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )


def conv2d1x1(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    next_feature_size: Literal["down", "up"] = "down",
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
    next_feature_size : bool, optional
        Whether to use ConvTranspose or not. Default: False

    Returns
    -------
    nn.Conv2d | nn.ConvTranspose2d
        2D convolution layer with a kernel size of 1 x 1
    """
    if next_feature_size == "down":
        return nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=stride, bias=False
        )
    elif next_feature_size == "up":
        return nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=1, stride=stride, bias=False
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
        next_feature_size: Literal["down", "up"] = "down",
        *,
        expansion: int | None = None,
    ) -> None:
        super().__init__()
        if expansion is not None:
            self.expansion = expansion
        mid_ch = in_ch * self.expansion
        self.conv1 = conv2d3x3(in_ch, mid_ch, stride, next_feature_size)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = conv2d3x3(mid_ch, out_ch, stride, next_feature_size)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch, stride, next_feature_size),
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
        print(f"x size is \n{x.shape}")
        print(f"h size is \n{h.shape}")

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
        next_feature_size: Literal["down", "up"] = "down",
        *,
        expansion: int | None = None,
    ) -> None:
        super().__init__()
        if expansion is not None:
            self.expansion = expansion
        mid_ch = in_ch * self.expansion
        self.conv1 = conv2d1x1(in_ch, mid_ch, stride, next_feature_size)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = conv2d3x3(mid_ch, mid_ch, stride, next_feature_size)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = conv2d1x1(mid_ch, out_ch, stride, next_feature_size)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.activation = add_activation(activation)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch, stride, next_feature_size),
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
        next_feature_size: Literal["down", "up"] = "down",
        *,
        expansion: int = 4,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.expansion = expansion
        mid_ch = in_ch * self.expansion
        self.residual = nn.Sequential(
            conv2d1x1(in_ch, mid_ch, stride, next_feature_size),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d3x3(mid_ch, mid_ch, stride, next_feature_size),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d1x1(mid_ch, out_ch, stride, next_feature_size),
            nn.BatchNorm2d(out_ch),
        )

        self.se_layer = SELayer(out_ch, reduction, activation)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, in_ch * expansion, stride),
                nn.BatchNorm2d(in_ch * expansion),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.residual(x)
        h = self.shortcut(x) + self.se_layer(h)
        h = self.activation(h)
        return h


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
        layers: tuple[int, int, int, int],
        in_ch: int = 1,
        out_ch: int = 1000,
        zero_init_residual: bool = False,
        activation: ActivationName = "relu",
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            in_ch,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.actfunc = add_activation(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], activation=activation
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            activation=activation,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            activation=activation,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            activation=activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_ch)

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

    def _make_layer(
        self,
        block: type[MyBasicBlock | MyBottleneck | SEBottleneck],
        planes: int,
        num_blocks: int,
        stride: int = 1,
        activation: ActivationName = "relu",
        next_feature_size: Literal["down", "up"] = "down",
    ) -> nn.Sequential:
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, activation, next_feature_size)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    activation,
                    next_feature_size,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actfunc(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
