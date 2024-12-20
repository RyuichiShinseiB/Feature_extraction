from pathlib import Path
from typing import Any

from hydra import compose, initialize

from src.configs.model_configs import TrainClassificationModel
from src.configs.model_configs.base_configs import TrainDatasetConfig
from src.predefined_models import LoadModel


def test_train_classification_model() -> None:
    with initialize("../src/configs/train_conf/", version_base=None):
        _cfg = compose("classification")

    cfg = TrainClassificationModel.from_dictconfig(_cfg)

    assert cfg.model.feature.network_type == "DownSamplingResNet"
    assert cfg.model.classifier.network_type == "MLP"
    assert isinstance(cfg.dataset.path, Path)
    assert (
        cfg.model.feature.hyper_parameters.output_channels
        == cfg.model.classifier.hyper_parameters.input_dimension
    )


def test_cvt_cfg_to_dict() -> None:
    with initialize("../src/configs/train_conf/", version_base=None):
        _cfg = compose("classification")

    cfg = TrainClassificationModel.from_dictconfig(_cfg).to_dict()

    def _assert_recursive_dict(d: dict[str, None | dict | Any]) -> None:
        for v in d.values():
            if v is None:
                assert True
            if isinstance(v, dict):
                _assert_recursive_dict(v)

    _assert_recursive_dict(cfg)


def test_load_cfg_to_ResNet() -> None:
    with initialize("../src/configs/train_conf/", version_base=None):
        _cfg = compose("classification")
    cfg = TrainClassificationModel.from_dictconfig(_cfg)
    _model = LoadModel.load_model(
        cfg.model.feature.network_type,
        cfg.model.feature.hyper_parameters,
    )


def test_load_cfg_to_MLP() -> None:
    with initialize(
        "../src/configs/train_conf/classifilation", version_base=None
    ):
        _cfg = compose("ResNet-highlow")
    cfg = TrainClassificationModel.from_dictconfig(_cfg)
    _model = LoadModel.load_model(
        cfg.model.classifier.network_type,
        cfg.model.classifier.hyper_parameters,
    )


def test_datasetconfig_for_training() -> None:
    with initialize(
        "../src/configs/train_conf/classifilation", version_base=None
    ):
        _cfg = compose("ResNet-highlow")

    cfg = TrainDatasetConfig.from_dictconfig(_cfg.dataset)

    print(cfg)
