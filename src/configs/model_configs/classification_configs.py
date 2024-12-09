from dataclasses import dataclass

from .base_configs import (
    NetworkConfig,
    RecursiveDataclass,
    TrainConfig,
    TrainDatasetConfig,
)


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
