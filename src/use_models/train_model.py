# Standard Library
from typing import Any, Optional

# Third Party Library
import hydra
import torch
from torch import nn, optim

# First Party Library
from src import Tensor
from src.configs.model_configs import MyConfig
from src.predefined_models import model_define
from src.utilities import EarlyStopping, calc_loss, get_dataloader, weight_init


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: MyConfig) -> None:
    # CPUで計算するかGPUで計算するかを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    model = model_define(cfg.model_cfg, device=device)
    # 重みの初期化
    model.apply(weight_init)

    # Early stoppingを使うかどうか
    # 使用する場合、過学習を防ぐためにepochの途中で終了する
    early_stopping: Optional[EarlyStopping]
    if cfg.train_cfg.early_stopping:
        early_stopping = EarlyStopping()
    else:
        early_stopping = None

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    dataloader = get_dataloader(
        cfg.dataset_cfg.path,
        cfg.dataset_cfg.transform,
        cfg.train_cfg.batch_size,
        shuffle=True,
        split_ratio=(0.8, 0.2),
        generator_seed=42,
    )

    # 損失関数の設定
    criterion: nn.BCELoss | nn.MSELoss | Any
    if cfg.train_cfg.loss == "bce":
        criterion = nn.BCELoss()
    elif cfg.train_cfg.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise RuntimeError("Please select another loss function")

    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), cfg.train_cfg.lr)

    reconst_images: list[Tensor] = []
    train_losses: list[float] = []
    valid_losses: list[float] = []

    for epoch in range(cfg.train_cfg.epochs):
        train_loss = 0.0
        valid_loss = 0.0

        for _i_train, (x, _) in enumerate(dataloader, 0):
            # モデルの訓練
            model.train()

            # データをGPUに移動
            x = x.to(device)
            # 損失の計算
            loss, _ = calc_loss(x, criterion, model)

            # 重みの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 損失をリストに保存
            train_losses.append(loss.cpu().item())
            train_loss += loss.cpu().item()

        for _i_valid, (x, _) in enumerate(dataloader, 0):
            model.eval()
            x = x.to(device)
            loss, x_pred = calc_loss(x, criterion, model)

            # 損失をリストに保存
            valid_losses.append(loss.cpu().item())
            valid_loss += loss.cpu().item()

        print(
            "Epoch: {}/{}\t|Train loss: {}\t|Valid loss: {}".format(
                epoch + 1,
                cfg.train_cfg.epochs,
                train_loss / (_i_train + 1),
                valid_loss / (_i_valid + 1),
            )
        )
        if (
            epoch
            % (cfg.train_cfg.epochs // cfg.train_cfg.num_save_reconst_image)
            == 0
        ):
            reconst_images.append(x_pred)

        if cfg.train_cfg.early_stopping:
            early_stopping(train_loss, model, cfg.train_cfg.trained_save_path)
