from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Sized, TypeVar

import torch
from torch.utils.data import DataLoader

from ...mytyping import Device
from ...predefined_models._load_model import LoadModel
from ...utilities import find_project_root, get_dataloader
from .base_configs import (
    ExtractDatasetConfig,
    RecursiveDataclass,
    TrainConfig,
    _TwoStageModelConfig,
)

SizedT = TypeVar("SizedT", bound=Sized)


@dataclass
class ExtractConfig(RecursiveDataclass):
    model: _TwoStageModelConfig = field(default_factory=_TwoStageModelConfig)
    train: TrainConfig = TrainConfig()
    dataset: ExtractDatasetConfig = ExtractDatasetConfig()
    feature_save_path: Path = field(default_factory=Path)

    def create_datasets(self, seed: int = 42) -> tuple[DataLoader, DataLoader]:
        root = find_project_root()
        train_dataloader = get_dataloader(
            dataset_path=root / self.dataset.train_path,
            dataset_transform=self.dataset.transform,
            batch_size=self.train.train_hyperparameter.batch_size,
            shuffle=False,
            generator_seed=seed,
            extraction=True,
            cls_conditions=self.dataset.cls_conditions,
        )
        check_dataloader = get_dataloader(
            dataset_path=root / self.dataset.check_path,
            dataset_transform=self.dataset.transform,
            batch_size=self.train.train_hyperparameter.batch_size,
            shuffle=False,
            generator_seed=seed,
            extraction=True,
            cls_conditions=self.dataset.cls_conditions,
        )
        return train_dataloader, check_dataloader

    @staticmethod
    def _get_longest_value(items: Sequence[SizedT]) -> SizedT:
        a = {v: len(v) for v in items}
        aa = max(a.items(), key=lambda kv: kv[1])
        val = aa[0]
        # aa = sorted(a.items(), key=lambda kv: kv[1])
        return val

    def create_model(self, device: Device = "cpu") -> torch.nn.Sequential:
        root = find_project_root()
        pth_dir = root / "models" / self.train.trained_save_path
        base_pth_name = "model_parameters.pth"
        pths = [
            pth.name
            for pth in pth_dir.glob("*.pth")
            if base_pth_name in pth.name
        ]
        pth = self._get_longest_value(pths)
        pth_path = pth_dir / pth

        first_stage = LoadModel.load_model(
            self.model.first_stage.network_type,
            self.model.first_stage.hyper_parameters,
        )
        second_stage = LoadModel.load_model(
            self.model.second_stage.network_type,
            self.model.second_stage.hyper_parameters,
        )

        two_stage_model = torch.nn.Sequential(first_stage, second_stage).to(
            device
        )
        two_stage_model.load_state_dict(torch.load(pth_path))
        return two_stage_model
