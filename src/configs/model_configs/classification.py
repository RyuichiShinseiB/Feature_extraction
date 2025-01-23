from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

import polars as pl
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

    def __post_init__(self) -> None:
        self.non_check_data_fullpath = self.get_not_check_data()
        pass

    @property
    def verbose_interval(self) -> int:
        return (
            self.train.train_hyperparameter.epochs
            // self.train.train_hyperparameter.num_save_reconst_image
        )

    @overload
    def create_dataloader(
        self,
        split_ratio: None,
        seed: int = 42,
        ignore_check_data: bool = False,
    ) -> DataLoader:
        ...

    @overload
    def create_dataloader(
        self,
        split_ratio: tuple[float, float],
        seed: int = 42,
        ignore_check_data: bool = False,
    ) -> tuple[DataLoader, DataLoader]:
        ...

    def create_dataloader(
        self,
        split_ratio: tuple[float, float] | None = (0.8, 0.2),
        seed: int = 42,
        ignore_check_data: bool = False,
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        root = find_project_root()
        is_valid_file = (
            self.callback_is_not_check_data() if ignore_check_data else None
        )
        return get_dataloader(
            dataset_path=root / self.dataset.path,
            dataset_transform=self.dataset.transform,
            split_ratio=split_ratio,
            batch_size=self.train.train_hyperparameter.batch_size,
            generator_seed=seed,
            is_valid_file=is_valid_file,
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

    # def callback_is_not_check_data(self) -> Callable[[str], bool]:
    def get_not_check_data(self) -> set[str]:
        dataset_path = find_project_root() / self.dataset.path
        data_paths = list(dataset_path.glob("*/*.png"))
        filenames = [f.name for f in data_paths]
        dirnames = [f.parent.name for f in data_paths]

        path_df = pl.DataFrame({"dirname": dirnames, "filename": filenames})
        pattern = r"[a-zA-Z](\d+)"
        non_check_data_path = (
            path_df.with_columns(
                (
                    # 画像ファイルの名前の後半に書かれているクロップ位置を抽出
                    pl.col("filename")
                    .str.extract_all(pattern)
                    .list.eval(
                        pl.element()
                        .str.replace_all(r"[a-zA-Z]", "")
                        .cast(pl.Int32)
                    )
                    .list.to_struct(fields=["crop_loc_x", "crop_loc_y"])
                    .alias("crop_location")
                )
            )
            .unnest("crop_location")
            .filter(
                # チェック用のデータは画像サイズごとにクロップされているので
                # 一致するクロップ位置の画像は除外
                (pl.col("crop_loc_x") % self.dataset.image_size != 0)
                | (pl.col("crop_loc_y") % self.dataset.image_size != 0)
            )
            .select(
                pl.concat_str(
                    [pl.col("dirname").cast(pl.Utf8), pl.col("filename")],
                    separator="/",
                ).alias("filepath")
            )
            .to_series()
        )

        return {str(dataset_path / p) for p in non_check_data_path}

    def callback_is_not_check_data(self) -> Callable[[str], bool]:
        return lambda s: s in self.non_check_data_fullpath
