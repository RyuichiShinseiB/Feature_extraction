import hydra
from omegaconf import DictConfig

from src.configs.model_configs import TrainClassificationModel
from src.predefined_models import LoadModel


def _load_config() -> DictConfig:
    with hydra.initialize("../tests/test_configs/", version_base=None):
        cfg = hydra.compose(config_name="classification")
    return cfg


def test_resnet_classifier() -> None:
    cfg = TrainClassificationModel.from_dictconfig(_load_config())

    _feature = LoadModel.from_config(cfg.model.feature)
    print(_feature)
