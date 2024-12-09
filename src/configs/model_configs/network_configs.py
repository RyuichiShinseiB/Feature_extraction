from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ...mytyping import ActivationName, ModelName, ResNetBlockName
from .autoencoder_configs import (
    RecursiveDataclass,
    TrainConfig,
    TrainDatasetConfig,
)


@dataclass
class NetworkHyperParams(RecursiveDataclass):
    # for mlp model
    input_dimension: int | None = None
    middle_dimensions: Sequence[int] | None = None
    output_dimension: int | None = None
    activation: ActivationName | None = None

    # for cnn model
    input_channels: int | None = None
    middle_channels: int | None = None
    output_channels: int | None = None

    # for autoencoder model
    latent_dimensions: int | None = None
    encoder_base_channels: int | None = None
    decoder_base_channels: int | None = None
    encoder_activation: ActivationName | None = None
    decoder_activation: ActivationName | None = None
    encoder_output_activation: ActivationName | None = None
    decoder_output_activation: ActivationName | None = None

    # for resnet
    inplanes: int | None = None
    block_name: ResNetBlockName | None = None
    layers: tuple[int, int, int, int] | None = None
    input_size: Sequence[int] | None = None
    output_activation: ActivationName | None = None

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


@dataclass
class ClassificationModelConfig(RecursiveDataclass):
    name: str = "NeuralNetwork"
    feature: NetworkConfig = NetworkConfig()
    classifier: NetworkConfig = NetworkConfig()


@dataclass
class TrainClassificationModel(RecursiveDataclass):
    model: ClassificationModelConfig = ClassificationModelConfig()
    train: TrainConfig = TrainConfig()
    dataset: TrainDatasetConfig = TrainDatasetConfig()
