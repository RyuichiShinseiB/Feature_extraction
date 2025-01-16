from dataclasses import dataclass

from torch.utils.data import DataLoader

from ...mytyping import Model
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

    def create_dataset(self, seed: int = 42) -> DataLoader:
        root = find_project_root()
        return get_dataloader(
            dataset_path=root / self.dataset.path,
            dataset_transform=self.dataset.transform,
            batch_size=self.train.train_hyperparameter.batch_size,
            generator_seed=seed,
            cls_conditions=self.dataset.cls_conditions,
        )

    def create_model(self) -> tuple[Model, Model]:
        first_stage = LoadModel.load_model_from_config(self.model.classifier)
        second_stage = LoadModel.load_model_from_config(self.model.feature)
        return first_stage, second_stage
