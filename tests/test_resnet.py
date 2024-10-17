import torch

from src.predefined_models._CNN_modules import (
    MyBasicBlock,
    MyBottleneck,
    ResNet,
    SEBottleneck,
)


def test_mybasicblock() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet(
        MyBasicBlock,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)

    t = torch.rand(10, 3, 32, 32, device=device)

    out = resnet(t)
    assert out.shape == (10, 100)


def test_mybottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet(
        MyBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)
    print(resnet)

    t = torch.rand(10, 3, 32, 32, device=device)

    out = resnet(t)
    assert out.shape == (10, 100)


def test_sebottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ResNet(
        SEBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)
    print(resnet)

    t = torch.rand(10, 3, 32, 32, device=device)

    out = resnet(t)
    assert out.shape == (10, 100)
