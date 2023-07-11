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


# Hyper parameters dataclasses
@dataclass
class AutoencoderHyperParameter:
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActivationName = "relu"
    decoder_activation: ActivationName = "relu"
    encoder_output_activation: ActivationName = "relu"
    decoder_output_activation: ActivationName = "sigmoid"


@dataclass
class MAEViTHyperParameter:
    input_channels: int = 3
    emb_dim: int = 192
    num_patch_row: int = 2
    image_size: int = 32
    encoder_num_blocks: int = 12
    decoder_num_blocks: int = 4
    encoder_heads: int = 3
    decoder_heads: int = 3
    encoder_hidden_dim: int = 768
    decoder_hidden_dim: int = 768
    mask_ratio: float = 0.75
    dropout: float = 0.0


# Model dataclasses
@dataclass
class AutoencoderModelConfig:
    name: ModelName = "SimpleCAE64"
    hyper_parameters: AutoencoderHyperParameter = AutoencoderHyperParameter()


@dataclass
class MAEViTModelConfig:
    name: str = "MAEViT"
    hyper_parameters: MAEViTHyperParameter = MAEViTHyperParameter()


# Training hyper parameters dataclasses
@dataclass
class TrainHyperParameter:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    reconst_loss: Literal["bce", "mse", "None"] = "bce"
    latent_loss: Literal["softplus", "general"] | None = None
    num_save_reconst_image: int = 5
    early_stopping: bool = False


# Basic Training configuration dataclasses
@dataclass
class TrainConfig:
    trained_save_path: str = "./models"
    train_hyperparameter: TrainHyperParameter = TrainHyperParameter()


# Dataset dataclasses
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


# Actually Use dataclasses
@dataclass
class TrainAutoencoderConfig:
    model: AutoencoderModelConfig = AutoencoderModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: TrainDatasetConfig = TrainDatasetConfig()


@dataclass
class TrainMAEViTConfig:
    model: MAEViTModelConfig = MAEViTModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: TrainDatasetConfig = TrainDatasetConfig()


@dataclass
class ExtractConfig:
    model: AutoencoderModelConfig = AutoencoderModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: ExtractDatasetConfig = ExtractDatasetConfig()
    feature_save_path: str = "${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"


def dict2dataclass(cls: Type[Dataclass], src: dict) -> Dataclass:
    kwargs = dict()
    field_dict: dict[str, Field] = {fld.name: fld for fld in fields(cls)}
    field_type_dict: dict[str, type] = get_type_hints(cls)
    for src_key, src_value in src.items():
        assert src_key in field_dict, f"Invalid Data Structure: {src_key}"
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


@hydra.main(version_base=None, config_path="train_conf", config_name="MAEViT")
def main(cfg: DictConfig) -> None:
    print(f"{type(cfg)=}")
    print(OmegaConf.to_container(cfg))
    dataclass_cfg = dictconfig2dataclass(cfg, TrainMAEViTConfig)
    print(type(dataclass_cfg))
    print(asdict(dataclass_cfg))


if __name__ == "__main__":
    main()
