# Standard Library
from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, Type, TypeAlias, TypeVar, get_type_hints

# Third Party Library
import hydra
from omegaconf import DictConfig, OmegaConf

# First Party Library
from src import ActivationName, ModelName, TransformsNameValue

Dataclass = TypeVar("Dataclass")


@dataclass
class RecursiveDataclass:
    pass

    @classmethod
    def from_dict(cls, src: dict) -> "RecursiveDataclass":
        kwargs = dict()
        field_dict: dict[str, Field] = {fld.name: fld for fld in fields(cls)}
        field_type_dict: dict[str, type] = get_type_hints(cls)
        for src_key, src_value in src.items():
            assert src_key in field_dict, "Invalid Data Structure"
            fld = field_dict[src_key]
            field_type = field_type_dict[fld.name]
            if issubclass(field_type, RecursiveDataclass):
                kwargs[src_key] = field_type.from_dict(src_value)
            else:
                kwargs[src_key] = src_value
        return cls(**kwargs)


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
class TrainHyperParameterConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    reconst_loss: Literal["bce", "mse"] = "bce"
    latent_loss: Literal["softplus", "general"] | None = None
    num_save_reconst_image: int = 5
    early_stopping: bool = False
    trained_save_path: str = "./models"


@dataclass
class TrainDatasetConfig:
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    transform: TransformsNameValue = field(default_factory=dict)


@dataclass
class ExtractDatasetConfig:
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    train_path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    check_path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    transform: TransformsNameValue = field(default_factory=dict)


@dataclass
class TrainConfig:
    model: ModelConfig = ModelConfig()
    train_hyperparameter: TrainHyperParameterConfig = (
        TrainHyperParameterConfig()
    )
    dataset: TrainDatasetConfig = TrainDatasetConfig()


@dataclass
class ExtractConfig:
    model: ModelConfig = ModelConfig()
    train_hyperparameter: TrainHyperParameterConfig = (
        TrainHyperParameterConfig()
    )
    dataset: ExtractDatasetConfig = ExtractDatasetConfig()
    feature_save_path: str = "${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"


@dataclass
class MyConfig:
    model: ModelConfig = ModelConfig()
    train: TrainHyperParameterConfig = TrainHyperParameterConfig()
    train_dataset: TrainDatasetConfig = TrainDatasetConfig()


def dict2dataclass(cls: Type[Dataclass], src: dict) -> Dataclass:
    kwargs = dict()
    field_dict: dict[str, Field] = {fld.name: fld for fld in fields(cls)}
    field_type_dict: dict[str, type] = get_type_hints(cls)
    for src_key, src_value in src.items():
        assert src_key in field_dict, "Invalid Data Structure"
        fld = field_dict[src_key]
        field_type = field_type_dict[fld.name]
        if is_dataclass(field_type):
            kwargs[src_key] = dict2dataclass(field_type, src_value)
        else:
            kwargs[src_key] = src_value
    return cls(**kwargs)


def dictconfig2dataclass(
    cfg: DictConfig, dataclass_cfg_cls: Type[Dataclass]
) -> Dataclass:
    dictconfig = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(dictconfig, dict):
        config = dict2dataclass(dataclass_cfg_cls, dictconfig)
    else:
        raise ValueError(f"cfg is not dictconfig.")
    return config


@hydra.main(version_base=None, config_path="train_conf", config_name="configs")
def main(cfg: DictConfig) -> None:
    print(f"{type(cfg)=}")
    print(OmegaConf.to_container(cfg))
    dataclass_cfg = dictconfig2dataclass(cfg, MyConfig)
    print(type(dataclass_cfg))
    print(asdict(dataclass_cfg))


if __name__ == "__main__":
    # Standard Library
    from pprint import pprint

    d = {
        "model": {
            "name": "SimpleCAE32",
            "hyper_parameters": {
                "input_channels": 1,
                "latent_dimensions": 128,
                "encoder_base_channels": 64,
                "decoder_base_channels": 64,
                "encoder_activation": "leakyrelu",
                "decoder_activation": "leakyrelu",
                "encoder_output_activation": "leakyrelu",
                "decoder_output_activation": "relu",
            },
        },
        "train_hyperparameter": {
            "lr": 0.001,
            "epochs": 100,
            "batch_size": 128,
            "reconst_loss": "mse",
            "latent_loss": None,
            "num_save_reconst_image": 5,
            "early_stopping": False,
            "trained_save_path": "${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        },
        "dataset": {
            "image_target": "CNTForest",
            "path": "./data/processed/CNTForest/cnt_sem_32x32/10k",
            "transform": {
                "Grayscale": 1,
                "RandomVerticalFlip": 0.5,
                "RandomHorizontalFlip": 0.5,
                "ToTensor": 0,
            },
        },
    }

    d2 = {
        "model": {
            "name": "SimpleCAE32",
            "hyper_parameters": {
                "input_channels": 1,
                "latent_dimensions": 128,
                "encoder_base_channels": 64,
                "decoder_base_channels": 64,
                "encoder_activation": "selu",
                "decoder_activation": "selu",
                "encoder_output_activation": "selu",
                "decoder_output_activation": "sigmoid",
            },
        },
        "train_hyperparameter": {
            "lr": 0.001,
            "epochs": 100,
            "batch_size": 128,
            "reconst_loss": "mse",
            "latent_loss": None,
            "num_save_reconst_image": 5,
            "early_stopping": False,
            "trained_save_path": "${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        },
        "dataset": {
            "image_target": "CNTForest",
            "train_path": "data/processed/CNTForest/cnt_sem_32x32/10k",
            "check_path": "data/processed/check/CNTForest/cnt_sem_for_check_32x32/10k",
            "transform": {
                "Grayscale": 1,
                "RandomVerticalFlip": 0.5,
                "RandomHorizontalFlip": 0.5,
                "ToTensor": 0,
            },
        },
    }

    a = dict2dataclass(ExtractConfig, d2)
    pprint(a)
