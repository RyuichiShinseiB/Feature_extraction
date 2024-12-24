from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize

from src.configs.model_configs import TrainClassificationModel
from src.configs.model_configs.base_configs import (
    NetworkConfig,
    TrainDatasetConfig,
    _TwoStageModelConfig,
)
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


def test_two_stage_model_config() -> None:
    _cfg = {
        "name": "Test",
        "encoder": {
            "network_type": "MLP",
            "pretrained_path": "./",
            "hyper_parameters": {
                "input_dimension": 100,
                "middle_dimensions": [100, 50, 25],
                "output_dimension": 100,
            },
        },
        "classifier": {
            "network_type": "MLP",
            "pretrained_path": "./",
            "hyper_parameters": {
                "input_dimension": 100,
                "middle_dimensions": [100, 50, 25],
                "output_dimension": 100,
            },
        },
    }
    cfg = _TwoStageModelConfig.from_dict(_cfg)

    assert isinstance(cfg.feature, NetworkConfig)
    assert isinstance(cfg.classifier, NetworkConfig)
    assert cfg.encoder is None
    assert cfg.decoder is None
    assert cfg.feature is cfg.first_stage
    assert cfg.classifier is cfg.seconde_stage

    print(cfg)


def test_two_stage_model_config_raise_error() -> None:
    _cfg = {
        "name": "Test",
        "decoder": {
            "network_type": "MLP",
            "pretrained_path": "./",
            "hyper_parameters": {
                "input_dimension": 100,
                "middle_dimensions": [100, 50, 25],
                "output_dimension": 100,
            },
        },
        "classifier": {
            "network_type": "MLP",
            "pretrained_path": "./",
            "hyper_parameters": {
                "input_dimension": 100,
                "middle_dimensions": [100, 50, 25],
                "output_dimension": 100,
            },
        },
    }
    with pytest.raises(ValueError):
        _TwoStageModelConfig.from_dict(_cfg)
