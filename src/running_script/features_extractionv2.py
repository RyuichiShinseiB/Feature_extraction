from pathlib import Path

import hydra
import polars as pl
import torch
from omegaconf import DictConfig

from src.configs.model_configs import ExtractConfig
from src.utilities import feat_ext


@hydra.main(
    version_base=None,
    config_path="../configs/eval_conf/classification",
    config_name="ResNet-highlow-testdata",
)
def main(_cfg: DictConfig) -> None:
    cfg = ExtractConfig.from_dictconfig(_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 抽出した特徴量の保存先
    base_save_path = Path(cfg.train.trained_save_path)
    # model_save_path = "./models" / base_save_path / "model_parameters_ea.pth"
    feature_storing_path = "./reports/features" / base_save_path
    if not feature_storing_path.exists():
        feature_storing_path.mkdir(parents=True)
    print(f"Path to store features: {feature_storing_path}")

    # 事前学習済みのモデルのロード
    # Load pretrained model
    model = cfg.create_model(device)

    # データローダーを設定
    # extraction=Trueにすることで、特徴量、ディレクトリ名、ファイル名を取得
    # 訓練用データでの特徴量抽出
    train_dataloader, check_dataloader = cfg.create_dataloader()
    print("Extraction from training data...")
    train_df = feat_ext.get_feature_table(model, train_dataloader, device)
    train_df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "_features_train_data.csv"
    )

    # 確認用データでの特徴量抽出
    print("Extraction from checking data...")
    check_df = feat_ext.get_feature_table(model, check_dataloader, device)
    check_df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "_features_check_data.csv"
    )


if __name__ == "__main__":
    main()
