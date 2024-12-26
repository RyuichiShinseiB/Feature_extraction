from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.configs.model_configs import ExtractConfig
from src.mytyping import Device, Tensor
from src.predefined_models import LoadModel
from src.utilities import ForExtractFolder, get_dataloader


@torch.no_grad()
def _extract_features(
    model: torch.nn.Sequential,
    dataloader: DataLoader[ForExtractFolder],
    device: Device,
) -> tuple[np.ndarray, list[int], list[int], list[str], list[str]]:
    features_list: list[Tensor] = []
    pred_list: list[int] = []
    target_list: list[int] = []
    dirnames_list: list[str] = []
    filenames_list: list[str] = []

    model.eval()
    for x, target, _, dirnames, filenames in tqdm.tqdm(dataloader):
        features = model[0](x.to(device))
        y: Tensor = model[1](features)
        if isinstance(features, tuple):
            if len(features) == 2:
                features = features[0]
            elif len(features) == 3:
                features = features[1]
        features_list.append(
            torch.flatten(features, start_dim=1).detach().cpu()
        )
        if y.shape[1] == 1:
            pred = torch.where(y > 0.5, 1, 0).detach().cpu().flatten().tolist()
        else:
            pred = torch.argmax(y).detach().cpu().flatten().tolist()
        target_list.extend(target)
        pred_list.extend(pred)
        dirnames_list.extend(dirnames)
        filenames_list.extend(filenames)
    features_array = torch.concat(features_list, dim=0).numpy()

    return (
        features_array,
        target_list,
        pred_list,
        dirnames_list,
        filenames_list,
    )


def _sort_out_extracted_data(
    features: np.ndarray,
    targets: list[int],
    preds: list[int],
    dirnames: list[str],
    filenames: list[str],
) -> pl.DataFrame:
    df = pl.DataFrame(features).with_columns(
        [
            pl.lit(pl.Series("target", targets, dtype=pl.Int32)),
            pl.lit(pl.Series("prediction", preds, dtype=pl.Int32)),
            pl.lit(pl.Series("dirname", dirnames, dtype=pl.Utf8)),
            pl.lit(pl.Series("filename", filenames, dtype=pl.Utf8)),
        ]
    )
    return df


def _get_feature_table(
    model: torch.nn.Sequential,
    dataloader: DataLoader[ForExtractFolder],
    device: Device,
) -> pl.DataFrame:
    features, targets, preds, dirnames, filenames = _extract_features(
        model, dataloader, device
    )
    df = _sort_out_extracted_data(
        features, targets, preds, dirnames, filenames
    )
    return df


@hydra.main(
    version_base=None,
    config_path="../configs/eval_conf/classification",
    config_name="ResNet-highlow",
)
def main(_cfg: DictConfig) -> None:
    cfg = ExtractConfig.from_dictconfig(_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 抽出した特徴量の保存先
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = "./models" / base_save_path / "model_parameters_ea.pth"
    feature_storing_path = "./reports/features" / base_save_path
    if not feature_storing_path.exists():
        feature_storing_path.mkdir(parents=True)
    print(f"Path to store features: {feature_storing_path}")

    # 事前学習済みのモデルのロード
    # Load pretrained model
    first_stage = LoadModel.from_config(cfg.model.first_stage)
    second_stage = LoadModel.from_config(cfg.model.second_stage)
    model = torch.nn.Sequential(first_stage, second_stage).to(device)
    model.load_state_dict(torch.load(model_save_path))

    # データローダーを設定
    # extraction=Trueにすることで、特徴量、ディレクトリ名、ファイル名を取得
    # 訓練用データでの特徴量抽出
    train_dataloader = get_dataloader(
        dataset_path=cfg.dataset.train_path,
        dataset_transform=cfg.dataset.transform,
        batch_size=cfg.train.train_hyperparameter.batch_size,
        shuffle=False,
        generator_seed=None,
        extraction=True,
        cls_conditions=cfg.dataset.cls_conditions,
    )
    print("Extraction from training data...")
    train_df = _get_feature_table(model, train_dataloader, device)
    train_df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "features_train_data.csv"
    )

    # 確認用データでの特徴量抽出
    check_dataloader = get_dataloader(
        dataset_path=cfg.dataset.check_path,
        dataset_transform=cfg.dataset.transform,
        batch_size=cfg.train.train_hyperparameter.batch_size,
        shuffle=False,
        generator_seed=None,
        extraction=True,
        cls_conditions=cfg.dataset.cls_conditions,
    )
    print("Extraction from checking data...")
    check_df = _get_feature_table(model, check_dataloader, device)
    check_df.select(pl.all().sort_by("filename")).write_csv(
        feature_storing_path / "features_check_data.csv"
    )


if __name__ == "__main__":
    main()
