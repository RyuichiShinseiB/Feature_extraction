import torch

from src.predefined_models._CNN_modules import (
    MyBasicBlock,
    MyBottleneck,
    ResNet,
    ResNet_,
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

    out, indices = resnet(t)
    print(f"{indices.shape=}")
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

    t = torch.rand(10, 3, 32, 32, device=device)

    out, indices = resnet(t)
    print(f"{indices.shape=}")
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

    t = torch.rand(10, 3, 32, 32, device=device)

    out, indices = resnet(t)
    assert out.shape == (10, 100)


def test_upsample_mybasicblock() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.rand(10, 100, device=device)
    indices = torch.randint(0, 10, (10, 64, 8, 8), device=device)
    resnet = ResNet_(
        MyBasicBlock,
        layers=(3, 4, 6, 3),
        in_ch=100,
        out_ch=1,
        activation="relu",
        reconstructed_size=(32, 32),
    ).to(device)
    print(resnet)

    out = resnet(t, indices)
    assert out.shape == (10, 1, 32, 32)
