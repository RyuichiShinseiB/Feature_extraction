from dataclasses import dataclass

from ..base_configs import (
    NetworkConfig,
    RecursiveDataclass,
    TrainConfig,
    TrainDatasetConfig,
)


@dataclass
class _AutoencoderModelConfig(RecursiveDataclass):
    name: str
    encoder: NetworkConfig
    decoder: NetworkConfig


@dataclass
class TrainAutoencoderConfigV2(RecursiveDataclass):
    model: _AutoencoderModelConfig
    train: TrainConfig
    dataset: TrainDatasetConfig


@dataclass
class _VAEModelConfig(RecursiveDataclass):
    name: str
    encoder: NetworkConfig
    latent_mean: NetworkConfig
    latent_var: NetworkConfig
    decoder: NetworkConfig


@dataclass
class TrainVAEConfig(RecursiveDataclass):
    model: _VAEModelConfig
    train: TrainConfig
    dataset: TrainDatasetConfig
