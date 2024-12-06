# Standard Library
from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Type, TypeGuard, TypeVar, get_type_hints

# Third Party Library
import hydra
from omegaconf import DictConfig, OmegaConf

# First Party Library
from src.mytyping import (
    ActivationName,
    ModelName,
    ResNetBlockName,
    TransformsNameValue,
)


def _is_recursivedataclass(obj: type) -> TypeGuard["RecursiveDataclass"]:
    return is_dataclass(obj) and issubclass(obj, RecursiveDataclass)


RecursiveDcT = TypeVar("RecursiveDcT", bound="RecursiveDataclass")

DataClassT = TypeVar("DataClassT", bound="RecursiveDataclass")


@dataclass
class RecursiveDataclass:
    pass

    @classmethod
    def from_dict(cls: Type[RecursiveDcT], src: dict) -> RecursiveDcT:
        kwargs: dict[str, "RecursiveDataclass"] = {}
        field_dict: dict[str, Field] = {
            field.name: field for field in fields(cls)
        }
        field_type_dict: dict[str, type] = get_type_hints(cls)
        for src_key, src_value in src.items():
            assert (
                src_key in field_dict
            ), f"Invalid Data Structure: {src_key} in {cls}"
            fld = field_dict[src_key]
            field_type = field_type_dict[fld.name]
            if _is_recursivedataclass(field_type):
                kwargs[src_key] = field_type.from_dict(src_value)
            else:
                kwargs[src_key] = src_value
        return cls(**kwargs)

    @classmethod
    def from_dictconfig(
        cls: Type[RecursiveDcT], cfg: DictConfig
    ) -> RecursiveDcT:
        src = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(src, dict):
            return cls.from_dict(src)
        else:
            raise TypeError(
                "Expected a config format like dict,"
                "but a config format that converts to"
                f"{type(src)} was entered."
            )

    def to_dict(self, ignore_none: bool = True) -> dict[str, Any]:
        if ignore_none:
            return asdict(
                self,
                dict_factory=lambda x: {k: v for k, v in x if v is not None},
            )
        else:
            return asdict(self)


# Hyper parameters dataclasses
@dataclass
class AutoencoderHyperParameter(RecursiveDataclass):
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActivationName = "relu"
    decoder_activation: ActivationName = "relu"
    encoder_output_activation: ActivationName = "relu"
    decoder_output_activation: ActivationName = "sigmoid"
    # For ResNetVAE
    input_size: tuple[int, int] | None = None
    block_name: ResNetBlockName | None = None


@dataclass
class MAEViTHyperParameter(RecursiveDataclass):
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
class AutoencoderModelConfig(RecursiveDataclass):
    name: ModelName = "SimpleCAE64"
    hyper_parameters: AutoencoderHyperParameter = AutoencoderHyperParameter()


@dataclass
class MAEViTModelConfig(RecursiveDataclass):
    name: ModelName = "MAEViT"
    hyper_parameters: MAEViTHyperParameter = MAEViTHyperParameter()


# Training hyper parameters dataclasses
@dataclass
class TrainHyperParameter(RecursiveDataclass):
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    reconst_loss: Literal["bce", "mse", "ce", "None"] = "bce"
    latent_loss: Literal["softplus", "general"] | None = None
    num_save_reconst_image: int = 5
    early_stopping: bool = False


# Basic Training configuration dataclasses
@dataclass
class TrainConfig(RecursiveDataclass):
    trained_save_path: str = "./models"
    train_hyperparameter: TrainHyperParameter = TrainHyperParameter()


# Dataset dataclasses
@dataclass
class TrainDatasetConfig(RecursiveDataclass):
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: Path = Path("../../data/processed/CNTForest/cnt_sem_64x64/10k")
    transform: TransformsNameValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)


@dataclass
class ExtractDatasetConfig(RecursiveDataclass):
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    train_path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    check_path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    transform: TransformsNameValue = field(default_factory=dict)


# Actually Use dataclasses
@dataclass
class TrainAutoencoderConfig(RecursiveDataclass):
    model: AutoencoderModelConfig = AutoencoderModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: TrainDatasetConfig = TrainDatasetConfig()


@dataclass
class TrainMAEViTConfig(RecursiveDataclass):
    model: MAEViTModelConfig = MAEViTModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: TrainDatasetConfig = TrainDatasetConfig()


@dataclass
class ExtractConfig(RecursiveDataclass):
    model: AutoencoderModelConfig = AutoencoderModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: ExtractDatasetConfig = ExtractDatasetConfig()
    feature_save_path: str = "${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"


def dict2dataclass(cls: Type[DataClassT], src: dict) -> DataClassT:
    kwargs = {}
    field_dict: dict[str, Field] = {fld.name: fld for fld in fields(cls)}
    field_type_dict: dict[str, type] = get_type_hints(cls)
    for src_key, src_value in src.items():
        assert src_key in field_dict, f"Invalid Data Structure: {src_key}"
        fld = field_dict[src_key]
        field_type = field_type_dict[fld.name]
        if is_dataclass(field_type):
            kwargs[src_key] = dict2dataclass(field_type, src_value)  # type: ignore
        else:
            kwargs[src_key] = src_value
    return cls(**kwargs)


def dictconfig2dataclass(
    cfg: DictConfig, dataclass_cfg_cls: Type[DataClassT]
) -> DataClassT:
    dictconfig = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(dictconfig, dict):
        config = dict2dataclass(dataclass_cfg_cls, dictconfig)
    else:
        raise ValueError("cfg is not dictconfig.")
    return config


@hydra.main(version_base=None, config_path="train_conf", config_name="MAEViT")
def main(cfg: DictConfig) -> None:
    print("In main: ", Path.cwd())
    print(f"{type(cfg)=}")
    pprint(cfg.train)
    pprint(OmegaConf.to_container(cfg), sort_dicts=False)
    pprint(TrainMAEViTConfig.from_dictconfig(cfg))


if __name__ == "__main__":
    main()
