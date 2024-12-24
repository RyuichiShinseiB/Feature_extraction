from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ....mytyping import (
    ActFuncName,
    ModelName,
    ResNetBlockName,
    TransformsNameValue,
)
from ..base_configs import (
    ExtractDatasetConfig,
    NetworkHyperParams,
    RecursiveDataclass,
    TrainConfig,
)


# Hyper parameters dataclasses
@dataclass
class AutoencoderHyperParameter(RecursiveDataclass):
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActFuncName = "relu"
    decoder_activation: ActFuncName = "relu"
    encoder_output_activation: ActFuncName = "relu"
    decoder_output_activation: ActFuncName = "sigmoid"
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
    hyper_parameters: NetworkHyperParams = NetworkHyperParams()


@dataclass
class MAEViTModelConfig(RecursiveDataclass):
    name: ModelName = "MAEViT"
    hyper_parameters: MAEViTHyperParameter = MAEViTHyperParameter()


# Dataset dataclasses
@dataclass
class TrainDatasetConfig(RecursiveDataclass):
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: Path = Path("../../data/processed/CNTForest/cnt_sem_64x64/10k")
    transform: TransformsNameValue = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)


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
