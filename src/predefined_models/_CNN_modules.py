# Standard Library
from typing import Literal, Union

# Third Party Library
from torch import nn

from ..mytyping import ActivationName, Tensor


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
        Either down-sampling or up-sampling, by default "down"

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
    activation: ActivationName = "relu",
) -> Union[
    nn.ReLU,
    nn.SELU,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Identity,
    nn.Softplus,
    nn.SiLU,
]:
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
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "softmax":
        return nn.Softmax(1)
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(
            f'There is no activation function such as "{activation}"'
        )


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
