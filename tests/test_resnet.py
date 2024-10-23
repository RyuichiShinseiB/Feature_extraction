import torch

from src.predefined_models._CNN_modules import (
    DownSamplingResNet,
    MyBasicBlock,
    MyBottleneck,
    SEBottleneck,
    UpSamplingResNet,
    _calc_layers_output_size,
)


def test_mybasicblock() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (32, 32)
    resnet = DownSamplingResNet(
        MyBasicBlock,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)

    t = torch.rand(10, 3, *input_size, device=device)

    output_size_conv1_calc = _calc_layers_output_size(
        input_size, (7,), (3,), (1,), (2,)
    )
    indices_size_calc = _calc_layers_output_size(
        output_size_conv1_calc, (3,), (1,), (1,), (2,)
    )

    out, output_size_conv1_act, indices = resnet(t)

    assert output_size_conv1_act == (10, 64, *output_size_conv1_calc)
    assert indices.shape == (10, 64, *indices_size_calc)
    assert out.shape == (10, 100)


def test_mybottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (32, 32)
    resnet = DownSamplingResNet(
        MyBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)

    t = torch.rand(10, 3, *input_size, device=device)

    output_size_conv1_calc = _calc_layers_output_size(
        input_size, (7,), (3,), (1,), (2,)
    )
    indices_size_calc = _calc_layers_output_size(
        output_size_conv1_calc, (3,), (1,), (1,), (2,)
    )

    out, output_size_conv1_act, indices = resnet(t)

    assert output_size_conv1_act == (10, 64, *output_size_conv1_calc)
    assert indices.shape == (10, 64, *indices_size_calc)
    assert out.shape == (10, 100)


def test_sebottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (32, 32)
    resnet = DownSamplingResNet(
        SEBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)

    t = torch.rand(10, 3, *input_size, device=device)

    output_size_conv1_calc = _calc_layers_output_size(
        input_size, (7,), (3,), (1,), (2,)
    )
    indices_size_calc = _calc_layers_output_size(
        output_size_conv1_calc, (3,), (1,), (1,), (2,)
    )

    out, output_size_conv1_act, indices = resnet(t)

    assert output_size_conv1_act == (10, 64, *output_size_conv1_calc)
    assert indices.shape == (10, 64, *indices_size_calc)
    assert out.shape == (10, 100)


def test_upsample_mybasicblock() -> None:
    upsampled_size = (32, 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.rand(10, 100, device=device)
    output_size_conv1_calc = _calc_layers_output_size(
        upsampled_size, (7,), (3,), (1,), (2,)
    )
    indices_size = _calc_layers_output_size(
        output_size_conv1_calc, (3,), (1,), (1,), (2,)
    )
    indices = torch.randint(0, 10, (10, 64, *indices_size), device=device)

    resnet = UpSamplingResNet(
        MyBasicBlock,
        layers=(3, 4, 6, 3),
        in_ch=100,
        out_ch=1,
        activation="relu",
        reconstructed_size=(32, 32),
    ).to(device)
    print(resnet)

    out = resnet.forward(t, output_size_conv1_calc, indices)
    assert out.shape == (10, 1, *upsampled_size)
