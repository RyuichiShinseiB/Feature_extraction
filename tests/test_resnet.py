import hydra
import torch
from omegaconf import DictConfig

from src.configs.model_configs import TrainAutoencoderConfig
from src.predefined_models._load_model import model_define
from src.predefined_models._ResNetVAE import (
    DownSamplingResNet,
    UpSamplingResNet,
)


def _load_config() -> DictConfig:
    with hydra.initialize("../tests/test_configs/", version_base=None):
        cfg = hydra.compose(config_name="ResNetVAE")
    return cfg


def test_downsample_mybasicblock() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = 32
    resnet = DownSamplingResNet(
        "basicblock",
        layers=(3, 4, 6, 3),
        input_channels=3,
        output_channels=100,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    t = torch.rand(10, 3, resolution, resolution, device=device)

    out = resnet.forward(t)

    assert out.shape == (10, 100)


def test_downsample_mybottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = 32
    resnet = DownSamplingResNet(
        "bottleneck",
        layers=(3, 4, 6, 3),
        input_channels=3,
        output_channels=100,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    t = torch.rand(10, 3, resolution, resolution, device=device)

    out = resnet(t)

    assert out.shape == (10, 100)


def test_downsample_sebottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = 32
    resnet = DownSamplingResNet(
        "sebottleneck",
        layers=(3, 4, 6, 3),
        input_channels=3,
        output_channels=100,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    t = torch.rand(10, 3, resolution, resolution, device=device)

    out = resnet(t)

    assert out.shape == (10, 100)


def test_upsample_mybasicblock() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.rand(10, 100, device=device)
    resolution = 32
    dim = 100
    ch = 3

    resnet = UpSamplingResNet(
        "basicblock",
        layers=(3, 4, 6, 3),
        input_channels=dim,
        output_channels=ch,
        actfunc="relu",
        resolution=resolution,
    ).to(device)
    print(resnet)

    out = resnet.forward(t)
    assert out.shape == (10, ch, resolution, resolution)


def test_upsample_mybottleneck() -> None:
    resolution = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.rand(10, 100, device=device)
    dim = 100
    ch = 3

    resnet = UpSamplingResNet(
        "bottleneck",
        layers=(3, 4, 6, 3),
        input_channels=dim,
        output_channels=ch,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    out = resnet.forward(t)
    assert out.shape == (10, ch, resolution, resolution)


def test_upsample_sebottleneck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.rand(10, 100, device=device)
    resolution = 32
    dim = 100
    ch = 3

    resnet = UpSamplingResNet(
        "bottleneck",
        layers=(3, 4, 6, 3),
        input_channels=dim,
        output_channels=ch,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    out = resnet.forward(t)
    assert out.shape == (10, ch, resolution, resolution)


def test_up_down_sampling_with_bottleneck() -> None:
    block = "bottleneck"
    resolution = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channel = 3
    latent_dimension = 100

    downsampling = DownSamplingResNet(
        block,  # type: ignore
        layers=(3, 4, 6, 3),
        input_channels=input_channel,
        output_channels=latent_dimension,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    upsampling = UpSamplingResNet(
        block,  # type: ignore
        layers=(3, 4, 6, 3),
        input_channels=latent_dimension,
        output_channels=input_channel,
        actfunc="relu",
        resolution=resolution,
    ).to(device)

    t = torch.rand(10, 3, resolution, resolution, device=device)

    vec = downsampling.forward(t)
    mat = upsampling.forward(vec)

    assert mat.shape == t.shape


def test_load_config() -> None:
    _load_config()


def test_load_resnetvae() -> None:
    # cfg = _load_config()
    cfg = TrainAutoencoderConfig.from_dictconfig(_load_config())
    print(cfg.model)

    model = model_define(cfg.model)
    print(model)


def test_training_resnetvae() -> None:
    cfg = TrainAutoencoderConfig.from_dictconfig(_load_config())

    ch = cfg.model.hyper_parameters.input_channels
    resolution = cfg.model.hyper_parameters.input_resolution

    assert isinstance(ch, int)
    assert isinstance(resolution, int)

    model = model_define(cfg.model)
    print(model)
    dummy = torch.randn(1, ch, resolution, resolution)
    y = model(dummy)[0]
    assert y.shape == dummy.shape
    # torchinfo.summary(model, (1, ch, resolution, resolution))
