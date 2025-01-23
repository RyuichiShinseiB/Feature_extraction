from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Sized, TypeVar

import polars as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ...mytyping import Device
from ...predefined_models._load_model import LoadModel
from ...utilities import find_project_root, get_dataloader
from .base_configs import (
    ExtractDatasetConfig,
    RecursiveDataclass,
    TrainConfig,
    TrainDatasetConfig,
    _TwoStageModelConfig,
)

SizedT = TypeVar("SizedT", bound=Sized)


@dataclass
class ExtractConfig(RecursiveDataclass):
    model: _TwoStageModelConfig = field(default_factory=_TwoStageModelConfig)
    train: TrainConfig = TrainConfig()
    dataset: ExtractDatasetConfig = ExtractDatasetConfig()
    feature_save_path: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        self.non_check_data_fullpath = self.get_not_check_data()

        project_root = find_project_root()
        trained_model_cfg_dir = (
            project_root
            / "models"
            / self.train.trained_save_path
            / ".hydra"
            / "config.yaml"
        )
        _cfg = OmegaConf.load(trained_model_cfg_dir)
        self.model_at_train = _TwoStageModelConfig.from_dictconfig(_cfg.model)
        self.dataset_at_train = TrainDatasetConfig.from_dictconfig(
            _cfg.dataset
        )

    def create_dataloader(
        self, seed: int = 42, ignore_check_data: bool = False
    ) -> tuple[DataLoader, DataLoader]:
        root = find_project_root()
        is_valid_file = (
            self.callback_is_not_check_data() if ignore_check_data else None
        )
        train_dataloader = get_dataloader(
            dataset_path=root / self.dataset.train_path,
            dataset_transform=self.dataset.transform,
            batch_size=self.train.train_hyperparameter.batch_size,
            shuffle=False,
            generator_seed=seed,
            extraction=True,
            cls_conditions=self.dataset.cls_conditions,
            is_valid_file=is_valid_file,
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
        base_pth_name = "model_parameters"
        pths = [
            pth.name
            for pth in pth_dir.glob("*.pth")
            if base_pth_name in pth.name
        ]
        print(pths)
        pth = self._get_longest_value(pths)
        pth_path = pth_dir / pth

        print(f"Load trained parameters: {pth_path}")
        first_stage = LoadModel.load_model(
            self.model_at_train.first_stage.network_type,
            self.model_at_train.first_stage.hyper_parameters,
        )
        second_stage = LoadModel.load_model(
            self.model_at_train.second_stage.network_type,
            self.model_at_train.second_stage.hyper_parameters,
        )

        two_stage_model = torch.nn.Sequential(first_stage, second_stage).to(
            device
        )
        two_stage_model.load_state_dict(torch.load(pth_path))
        return two_stage_model

    def get_not_check_data(self) -> set[str]:
        dataset_path = find_project_root() / self.dataset.train_path
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
