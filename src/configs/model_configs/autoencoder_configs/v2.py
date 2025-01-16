from dataclasses import dataclass
from typing import overload

from torch.utils.data import DataLoader

from ....mytyping import Device
from ....predefined_models._build_VAE import VAEFrame
from ....utilities import find_project_root, get_dataloader
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

    @property
    def verbose_interval(self) -> int:
        return (
            self.train.train_hyperparameter.epochs
            // self.train.train_hyperparameter.num_save_reconst_image
        )

    @overload
    def create_dataloader(
        self, split_ratio: None, seed: int = 42
    ) -> DataLoader:
        ...

    @overload
    def create_dataloader(
        self, split_ratio: tuple[float, float], seed: int = 42
    ) -> tuple[DataLoader, DataLoader]:
        ...

    def create_dataloader(
        self,
        split_ratio: tuple[float, float] | None = (0.8, 0.2),
        seed: int = 42,
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        root = find_project_root()
        return get_dataloader(
            dataset_path=root / self.dataset.path,
            dataset_transform=self.dataset.transform,
            split_ratio=split_ratio,
            batch_size=self.train.train_hyperparameter.batch_size,
            generator_seed=seed,
            cls_conditions=self.dataset.cls_conditions,
        )

    def create_vae_model(self, device: Device = "cpu") -> VAEFrame:
        model = VAEFrame.build_from_config(self.model).to(device)
        return model
