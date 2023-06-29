# Standard Library
import os
from pathlib import Path
from typing import Any

# Third Party Library
import hydra
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn, optim

# First Party Library
from src import Tensor
from src.configs.model_configs import MyConfig
from src.predefined_models import model_define
from src.utilities import (
    EarlyStopping,
    calc_loss,
    display_cfg,
    get_dataloader,
    weight_init,
)


@hydra.main(
    version_base=None, config_path="../configs/conf", config_name="configs"
)
def main(cfg: MyConfig) -> None:
    # Display Configuration
    display_cfg(cfg)

    # 訓練済みモデル、訓練途中の再構成画像のパス
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = Path("./models") / base_save_path
    figure_save_path = Path("./reports/figures") / base_save_path

    # 再構成画像を保存する間隔
    save_interval = cfg.train.epochs // cfg.train.num_save_reconst_image

    # CPUで計算するかGPUで計算するかを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    model = model_define(cfg.model, device=device).to(device)
    # 重みの初期化
    model.apply(weight_init)

    # Early stoppingの宣言
    # 過学習を防ぐためにepochの途中で終了する
    early_stopping = EarlyStopping()

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    train_dataloader, val_dataloader = get_dataloader(
        cfg.dataset.path,
        cfg.dataset.transform,
        cfg.train.batch_size,
        shuffle=True,
        split_ratio=(0.8, 0.2),
        generator_seed=42,
    )

    # 損失関数の設定
    criterion: nn.BCELoss | nn.MSELoss | Any
    if cfg.train.loss == "bce":
        criterion = nn.BCELoss()
    elif cfg.train.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise RuntimeError("Please select another loss function")

    # オプティマイザの設定
    optimizer = optim.Adam(model.parameters(), cfg.train.lr)

    reconst_images: list[Tensor] = []
    train_losses: list[float] = []
    valid_losses: list[float] = []

    test_image = next(iter(val_dataloader))[0][:64].to(device)

    # Training start
    for epoch in range(cfg.train.epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # 訓練
        for _i_train, (x, _) in enumerate(train_dataloader, 0):
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

        # 検証
        for _i_valid, (x, _) in enumerate(val_dataloader, 0):
            model.eval()
            x = x.to(device)
            loss, _ = calc_loss(x, criterion, model)

            # 損失をリストに保存
            valid_losses.append(loss.cpu().item())
            valid_loss += loss.cpu().item()

        print(
            "Epoch: {}/{}\t|Train loss: {:.5f}\t|Valid loss: {:.5f}".format(
                epoch + 1,
                cfg.train.epochs,
                train_loss / (_i_train + 1),
                valid_loss / (_i_valid + 1),
            )
        )

        # 再構成できているかを確認する画像の保存
        if epoch % save_interval == 0:
            model.eval()
            _, test_output = calc_loss(test_image, criterion, model)
            reconst_images.append(
                vutils.make_grid(test_output, normalize=True)
            )

        if cfg.train.early_stopping:
            early_stopping(
                train_loss, model, save_path=model_save_path / "model.pth"
            )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(train_losses)), train_losses, label="train loss")
    ax.set_xlabel("iterations")
    ax.set_ylabel(f"{cfg.train.loss.upper()} loss")
    ax.legend()
    if not figure_save_path.exists():
        os.makedirs(figure_save_path)
    fig.savefig(figure_save_path / "loss.jpg")

    for i, reconst_image in enumerate(reconst_images):
        vutils.save_image(
            reconst_image,
            fp=figure_save_path / f"reconst_images{i*save_interval:03}.png",
        )

    torch.save(model.state_dict(), model_save_path / "model_parameters.pth")


if __name__ == "__main__":
    main()
