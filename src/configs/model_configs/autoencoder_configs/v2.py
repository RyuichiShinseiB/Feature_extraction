from dataclasses import dataclass

from ..base_configs import (
    NetworkConfig,
    RecursiveDataclass,
    TrainConfig,
    TrainDatasetConfig,
)


@dataclass
class _ModelConfig(RecursiveDataclass):
    name: str
    encoder: NetworkConfig
    decoder: NetworkConfig


@dataclass
class TrainAutoencoderConfigV2(RecursiveDataclass):
    model: _ModelConfig
    train: TrainConfig
    dataset: TrainDatasetConfig
