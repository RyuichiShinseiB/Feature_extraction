from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Sequence,
    Type,
    TypeGuard,
    TypeVar,
    get_type_hints,
)

from omegaconf import DictConfig, OmegaConf

from ...mytyping import (
    ActFuncName,
    ModelName,
    ResNetBlockName,
    TransformsNameValue,
)

RecursiveDcT = TypeVar("RecursiveDcT", bound="RecursiveDataclass")

DataClassT = TypeVar("DataClassT", bound="RecursiveDataclass")


def _is_recursivedataclass(obj: type) -> TypeGuard["RecursiveDataclass"]:
    return is_dataclass(obj) and issubclass(obj, RecursiveDataclass)


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


########## Network configurations ##########
@dataclass
class NetworkHyperParams(RecursiveDataclass):
    """General network hyper parameters

    ## MLP hyper parameters
    - input_dimension: int | None
    - middle_dimensions: Sequence[int] | None
    - output_dimension: int | None
    - actfunc: ActFuncName | Non
    - output_actfunc: ActFuncName | None = None

    ## CNN hyper parameters
    - input_channels: int | None
    - middle_channels: int | None
    - output_channels: int | None

    ## VAE hyper parameters
    - latent_dimensions: int | Non
    - encoder_base_channels: int | None
    - decoder_base_channels: int | None
    - encoder_actfunc: ActFuncName | None
    - decoder_actfunc: ActFuncName | None
    - encoder_output_actfunc: ActFuncName | None
    - decoder_output_actfunc: ActFuncName | None

    ## ResNet hyper parameters
    - inplanes: int | None
    - block_name: ResNetBlockName | None
    - layers: tuple[int, int, int, int] | None
    - input_size: Sequence[int] | None
    - input_resolusino: ActFuncName | None


    Raises
    ------
    ValueError
        In case of set `layers`,
    """

    # for mlp model
    input_dimension: int | None = None
    middle_dimensions: Sequence[int] | None = None
    output_dimension: int | None = None
    actfunc: ActFuncName | None = None
    dropout_rate: float | None = None

    # for cnn model
    input_channels: int | None = None
    middle_channels: int | None = None
    output_channels: int | None = None

    # for autoencoder model
    latent_dimensions: int | None = None
    encoder_base_channels: int | None = None
    decoder_base_channels: int | None = None
    encoder_actfunc: ActFuncName | None = None
    decoder_actfunc: ActFuncName | None = None
    encoder_output_actfunc: ActFuncName | None = None
    decoder_output_actfunc: ActFuncName | None = None

    # for resnet
    inplanes: int | None = None
    block_name: ResNetBlockName | None = None
    layers: tuple[int, int, int, int] | None = None
    output_actfunc: ActFuncName | None = None
    input_resolution: int | None = None

    def __post_init__(self) -> None:
        if self.layers is not None and len(self.layers) != 4:
            raise ValueError(
                "ResNet has four types of skipp-connections, "
                "which are set using the `layers` field.\n"
                f"However, There are {len(self.layers)} elements in `layers`"
            )


@dataclass
class NetworkConfig(RecursiveDataclass):
    network_type: ModelName = "MLP"
    pretrained_path: Path | None = None
    hyper_parameters: NetworkHyperParams = NetworkHyperParams()

    def __post_init__(self) -> None:
        if self.pretrained_path is None:
            return
        if not isinstance(self.pretrained_path, Path):
            self.pretrained_path = Path(self.pretrained_path)


########## Training configurations ##########
@dataclass
class TrainHyperParameter(RecursiveDataclass):
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    reconst_loss: Literal["bce", "mse", "ce", "None"] = "bce"
    latent_loss: Literal["softplus", "general"] | None = None
    weight_decay: float = 0.0
    num_save_reconst_image: int = 5
    early_stopping: int | None = None


@dataclass
class TrainConfig(RecursiveDataclass):
    trained_save_path: str = "./models"
    train_hyperparameter: TrainHyperParameter = TrainHyperParameter()


########## Dataset configurations ###########
@dataclass
class TrainDatasetConfig(RecursiveDataclass):
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: Path = Path("../../data/processed/CNTForest/cnt_sem_64x64/10k")
    cls_conditions: dict[int, list[str]] | None = None
    transform: TransformsNameValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)


@dataclass
class ExtractDatasetConfig(RecursiveDataclass):
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    train_path: Path = Path("../../data/processed/CNTForest/cnt_sem_64x64/10k")
    check_path: Path = Path("../../data/processed/CNTForest/cnt_sem_64x64/10k")
    cls_conditions: dict[int, list[str]] | None = None
    transform: TransformsNameValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.train_path, Path):
            self.train_path = Path(self.train_path)
        if not isinstance(self.check_path, Path):
            self.check_path = Path(self.check_path)
