# Standard Library
import os
from pathlib import Path

# Third Party Library
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from torchinfo import summary

# First Party Library
from src import Tensor
from src.configs.model_configs import MyConfig, dictconfig2dataclass
from src.predefined_models import model_define
from src.utilities import get_dataloader


@hydra.main(
    version_base=None,
    config_path="../configs/predict_conf",
    config_name="configs",
)
def main(_cfg: DictConfig) -> None:
    cfg = dictconfig2dataclass(_cfg, MyConfig)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 抽出した特徴量の保存先
    feature_storing_path = "./reports/features" / Path(
        cfg.train.trained_save_path
    )

    # 事前学習済みのモデルのロード
    # Load pretrained model
    model = model_define(cfg.model, device=device).to(device)
    # model.load_state_dict(torch.load(cfg.pretrained_path))
    summary(model, (1, 1, 32, 32))

    # データローダーを設定
    # extraction=Trueにすることで、データだけでなくデータのファイル名とディレクトリ名も取得
    dataloader = get_dataloader(
        cfg.dataset.path,
        cfg.dataset.transform,
        cfg.train.batch_size,
        generator_seed=None,
        extraction=True,
    )

    features_list: list[Tensor] = []
    dirnames_list: list[str] = []
    filenames_list: list[str] = []
    model.eval()
    with torch.no_grad():
        for x, _, dirnames, filenames in dataloader:
            _, features = model(x.to(device))
            features_list.extend(
                torch.flatten(features, start_dim=1).detach().cpu().tolist()
            )
            dirnames_list.extend(dirnames)
            filenames_list.extend(filenames)
    features_array = np.array(features_list)
    df = (
        pl.DataFrame(
            features_array,
        )
        .select(
            [
                pl.all(),
                pl.lit(pl.Series("dirname", dirnames_list)),
                pl.lit(pl.Series("filename", filenames_list)),
            ]
        )
        .sort(pl.col("filename"))
    )

    print(df)

    df.write_csv(feature_storing_path / "features.csv")


if __name__ == "__main__":
    main()
