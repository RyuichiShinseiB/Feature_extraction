from dataclasses import dataclass
from typing import overload

import torch
from torch.utils.data import DataLoader

from ...mytyping import Device, Model
from ...predefined_models._load_model import LoadModel
from ...utilities import find_project_root, get_dataloader
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

    def create_model(self) -> tuple[Model, Model]:
        first_stage = LoadModel.load_model_from_config(self.model.classifier)
        second_stage = LoadModel.load_model_from_config(self.model.feature)
        return first_stage, second_stage

    def create_sequential_model(
        self, device: Device = "cpu"
    ) -> torch.nn.Sequential:
        stages = self.create_model()
        model = torch.nn.Sequential(stages[0], stages[1]).to(device)
        return model
