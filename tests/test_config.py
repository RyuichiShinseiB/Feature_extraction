from pathlib import Path

from hydra import compose, initialize

from src.configs.model_configs import TrainClassificationModel


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
