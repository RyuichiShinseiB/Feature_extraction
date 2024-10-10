# Standard Library
from typing import Literal, Union

# Third Party Library
import torch
import torch.nn.functional as nn_func
from torch import nn

from src.mytyping import ActivationName, Tensor


def conv2d3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """Convolution layer with kernel size of 3x3

    Parameters
    ----------
    in_ch : int
        Number of channels in the input image
    out_ch : int
        Number of channels in the output image
    stride : int, optional
        Stride of the convolution. Default: 1

    Returns
    -------
    nn.Conv2d
        Layer of 2D Convolution
    """
    return nn.Conv2d(
        in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv2d1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """Convolution layer with kernel size of 1x1

    Parameters
    ----------
    in_ch : int
        Number of channels in the input image
    out_ch : int
        Number of channels in the output image
    stride : int, optional
        Stride of the convolution. Default: 1

    Returns
    -------
    nn.Conv2d
        Layer of 2D Convolution
    """
    return nn.Conv2d(
        in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False
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


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = conv2d3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv2d3x3(out_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, out_ch * self.expansion, stride=stride),
                nn.BatchNorm2d(out_ch * self.expansion),
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


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = conv2d1x1(in_ch, mid_ch)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = conv2d3x3(mid_ch, mid_ch, stride)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = conv2d1x1(mid_ch, in_ch * self.expansion)
        self.bn3 = nn.BatchNorm2d(in_ch * self.expansion)

        self.activation = add_activation(activation)

        if in_ch != in_ch * self.expansion:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, in_ch * self.expansion, stride),
                nn.BatchNorm2d(in_ch * self.expansion),
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
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            add_activation(activation),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, c, h, w = x.size()
        y: Tensor = nn_func.avg_pool2d(x, (h, w))
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        expansion: int = 4,
        reduction: int = 16,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.Residual = nn.Sequential(
            conv2d1x1(in_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d3x3(mid_ch, mid_ch, stride),
            nn.BatchNorm2d(mid_ch),
            add_activation(activation),
            conv2d1x1(mid_ch, in_ch * expansion),
            nn.BatchNorm2d(in_ch * expansion),
        )

        self.se_layer = SELayer(in_ch * expansion, reduction, activation)

        if in_ch != in_ch * expansion:
            self.shortcut = nn.Sequential(
                conv2d1x1(in_ch, in_ch * expansion, stride),
                nn.BatchNorm2d(in_ch * expansion),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = add_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.Residual(x)
        h = self.shortcut(x) + self.se_layer(h)
        h = self.activation(h)
        return h
