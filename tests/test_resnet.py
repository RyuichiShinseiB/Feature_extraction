import hydra
import torch
from omegaconf import DictConfig

from src.configs.model_configs import TrainAutoencoderConfig
from src.predefined_models import model_define
from src.predefined_models._CNN_modules import (
    MyBasicBlock,
    MyBottleneck,
    SEBottleneck,
)
from src.predefined_models._ResNetVAE import (
    DownSamplingResNet,
    UpSamplingResNet,
    _calc_layers_output_size,
)


def _load_config() -> DictConfig:
    with hydra.initialize(
        "../src/configs/train_conf/test_configs/", version_base=None
    ):
        cfg = hydra.compose(config_name="autoencoder")
    return cfg


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


def test_upsample_mybottleneck() -> None:
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
        MyBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=100,
        out_ch=1,
        activation="relu",
        reconstructed_size=(32, 32),
    ).to(device)
    # print(resnet)
    print(f"{output_size_conv1_calc=}")
    print(f"{indices.shape=}")

    out = resnet.forward(t, output_size_conv1_calc, indices)
    assert out.shape == (10, 1, *upsampled_size)


def test_upsample_sebottleneck() -> None:
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
        SEBottleneck,
        layers=(3, 4, 6, 3),
        in_ch=100,
        out_ch=1,
        activation="relu",
        reconstructed_size=(32, 32),
    ).to(device)
    print(resnet)

    print(f"{output_size_conv1_calc=}")
    out = resnet.forward(t, output_size_conv1_calc, indices)
    assert out.shape == (10, 1, *upsampled_size)


def test_up_down_sampling_with_bottleneck() -> None:
    block = MyBottleneck
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (32, 32)

    downsampling = DownSamplingResNet(
        block,
        layers=(3, 4, 6, 3),
        in_ch=3,
        out_ch=100,
        activation="relu",
    ).to(device)

    upsampling = UpSamplingResNet(
        block,
        layers=(3, 4, 6, 3),
        in_ch=100,
        out_ch=3,
        activation="relu",
        reconstructed_size=input_size,
    ).to(device)

    t = torch.rand(10, 3, *input_size, device=device)

    vec, size, indices = downsampling.forward(t)
    print(f"{vec.shape=}")
    print(f"{size=}")
    print(f"{indices.shape=}")
    mat = upsampling(vec, size, indices)

    assert mat.shape == t.shape


def test_load_config() -> None:
    _load_config()


def test_load_resnetvae() -> None:
    # cfg = _load_config()
    cfg = TrainAutoencoderConfig.from_dictconfig(_load_config())
    print(cfg.model)

    model = model_define(cfg.model)
    print(model)
