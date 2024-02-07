# Standard Library
from pathlib import Path

# Third Party Library
import hydra
import polars as pl
import torch
from omegaconf import DictConfig

# First Party Library
from src.configs.model_configs import ExtractConfig
from src.predefined_models import model_define
from src.utilities import extract_features, get_dataloader


@hydra.main(
    version_base=None,
    config_path="../configs/predict_conf",
    config_name="autoencoder",
)
def main(_cfg: DictConfig) -> None:
    cfg = ExtractConfig.from_dictconfig(_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 抽出した特徴量の保存先
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = "./models" / base_save_path / "model_parameters.pth"
    feature_storing_path = "./reports/features" / base_save_path
    print(f"Path to store features {feature_storing_path}")

    # 事前学習済みのモデルのロード
    # Load pretrained model
    model = model_define(cfg.model, device=device).to(device)
    model.load_state_dict(torch.load(model_save_path))

    # データローダーを設定
    # extraction=Trueにすることで、特徴量、ディレクトリ名、ファイル名を取得
    # 訓練用データでの特徴量抽出
    dataloader = get_dataloader(
        dataset_path=cfg.dataset.train_path,
        dataset_transform=cfg.dataset.transform,
        batch_size=cfg.train.train_hyperparameter.batch_size,
        shuffle=False,
        generator_seed=None,
        extraction=True,
    )
    print("Extraction from training data...")
    features, dirnames, filenames = extract_features(model, dataloader, device)
    df = pl.DataFrame(features).select(
        [
            pl.all(),
            pl.lit(pl.Series("dirname", dirnames)),
            pl.lit(pl.Series("filename", filenames)),
        ]
    )
    df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "features_train_data.csv"
    )
    del dataloader, df

    # 確認用データでの特徴量抽出
    dataloader = get_dataloader(
        dataset_path=cfg.dataset.check_path,
        dataset_transform=cfg.dataset.transform,
        batch_size=cfg.train.train_hyperparameter.batch_size,
        shuffle=False,
        generator_seed=None,
        extraction=True,
    )
    print("Extraction from checking data...")
    features, dirnames, filenames = extract_features(model, dataloader, device)
    df = pl.DataFrame(features).select(
        [
            pl.all(),
            pl.lit(pl.Series("dirname", dirnames)),
            pl.lit(pl.Series("filename", filenames)),
        ]
    )
    df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "features_check_data.csv"
    )


if __name__ == "__main__":
    main()
