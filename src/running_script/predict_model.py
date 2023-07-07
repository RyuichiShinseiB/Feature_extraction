# Standard Library
import os
from pathlib import Path

# Third Party Library
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchinfo import summary

# First Party Library
from src import Tensor
from src.configs.model_configs import ExtractConfig, dictconfig2dataclass
from src.predefined_models import model_define
from src.utilities import extract_features, get_dataloader


@hydra.main(
    version_base=None,
    config_path="../configs/predict_conf",
    config_name="configs",
)
def main(_cfg: DictConfig) -> None:
    cfg = dictconfig2dataclass(_cfg, ExtractConfig)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 抽出した特徴量の保存先
    base_save_path = Path(cfg.train_hyperparameter.trained_save_path)
    model_save_path = "./models" / base_save_path / "model_parameters.pth"
    feature_storing_path = "./reports/features" / Path(cfg.feature_save_path)
    # 事前学習済みのモデルのロード
    # Load pretrained model
    model = model_define(cfg.model, device=device).to(device)
    model.load_state_dict(torch.load(model_save_path))
    summary(model, (1, 1, 32, 32))

    # データローダーを設定
    # extraction=Trueにすることで、データだけでなくデータのファイル名とディレクトリ名も取得
    dataloader = get_dataloader(
        cfg.dataset.train_path,
        cfg.dataset.transform,
        cfg.train_hyperparameter.batch_size,
        generator_seed=None,
        extraction=True,
    )
    features, dirnames, filenames = extract_features(model, dataloader, device)
    df = (
        pl.DataFrame(
            features,
        )
        .select(
            [
                pl.all(),
                pl.lit(pl.Series("dirname", dirnames)),
                pl.lit(pl.Series("filename", filenames)),
            ]
        )
        .sort(pl.col("filename"))
    )
    df.write_csv(feature_storing_path / "features_train_data.csv")

    dataloader = get_dataloader(
        cfg.dataset.check_path,
        cfg.dataset.transform,
        cfg.train_hyperparameter.batch_size,
        generator_seed=None,
        extraction=True,
    )
    features, dirnames, filenames = extract_features(model, dataloader, device)
    df = (
        pl.DataFrame(
            features,
        )
        .select(
            [
                pl.all(),
                pl.lit(pl.Series("dirname", dirnames)),
                pl.lit(pl.Series("filename", filenames)),
            ]
        )
        .sort(pl.col("filename"))
    )
    df.write_csv(feature_storing_path / "features_check_data.csv")


if __name__ == "__main__":
    main()
