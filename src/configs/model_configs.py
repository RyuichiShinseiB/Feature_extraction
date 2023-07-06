# Standard Library
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Type, TypeVar

# Third Party Library
import hydra
from omegaconf import DictConfig, OmegaConf

# First Party Library
from src import ActivationName, ModelName, TransformsNameValue

DataclassConfig = TypeVar("DataclassConfig")


@dataclass
class HyperParameterConfig:
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActivationName = "relu"
    decoder_activation: ActivationName = "relu"
    encoder_output_activation: ActivationName = "relu"
    decoder_output_activation: ActivationName = "sigmoid"


@dataclass
class ModelConfig:
    name: ModelName = "SimpleCAE64"
    hyper_parameters: HyperParameterConfig = HyperParameterConfig()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    reconst_loss: Literal["bce", "mse"] = "bce"
    latent_loss: Literal["softplus", "general"] | None = None
    num_save_reconst_image: int = 5
    early_stopping: bool = False
    trained_save_path: str = "./models"


@dataclass
class DatasetConfig:
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    transform: TransformsNameValue = field(default_factory=dict)


@dataclass
class MyConfig:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: DatasetConfig = DatasetConfig()
    hydra: Any | None = None


def dictconfig2dataclass(
    cfg: DictConfig, dataclass_cfg_cls: Type[DataclassConfig]
) -> DataclassConfig:
    dictconfig = OmegaConf.to_container(cfg, resolve=True)
    config = dataclass_cfg_cls(**dictconfig)
    return config


@hydra.main(version_base=None, config_path="train_conf", config_name="configs")
def main(cfg: DictConfig) -> None:
    print(f"{type(cfg)=}")
    print(OmegaConf.to_yaml(cfg))
    dataclass_cfg = dictconfig2dataclass(cfg, MyConfig)
    print(type(dataclass_cfg))
    print(asdict(dataclass_cfg))


if __name__ == "__main__":
    main()
