# Standard Library
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
from src.utilities import EarlyStopping, calc_loss, get_dataloader, weight_init


@hydra.main(
    version_base=None, config_path="../configs/conf", config_name="configs"
)
def main(cfg: MyConfig) -> None:
    # 訓練済みモデル、訓練途中の再構成画像のパス
    print("Ran")
    print(f"type(cfg) == {type(cfg)}")

    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = "./model" / base_save_path
    figure_save_path = "./reports/figure" / base_save_path
    print("Ran")
    return
    # 再構成画像を保存する間隔
    save_interval = cfg.train.epochs // cfg.train.num_save_reconst_image

    # CPUで計算するかGPUで計算するかを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    model = model_define(cfg.model, device=device)
    # 重みの初期化
    model.apply(weight_init)

    # Early stoppingの宣言
    # 過学習を防ぐためにepochの途中で終了する
    early_stopping = EarlyStopping()

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    dataloader = get_dataloader(
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

    for epoch in range(cfg.train.epochs):
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
                cfg.train.epochs,
                train_loss / (_i_train + 1),
                valid_loss / (_i_valid + 1),
            )
        )
        if epoch % save_interval == 0:
            reconst_images.append(x_pred)

        if cfg.train.early_stopping:
            early_stopping(
                train_loss, model, save_path=model_save_path / "model.pth"
            )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(len(train_losses), train_losses, label="train loss")
    ax.set_xlabel("iterations")
    ax.set_ylabel(f"{cfg.train.loss.upper()} loss")
    ax.legend()
    fig.savefig(figure_save_path / "loss.jpg")

    for i, reconst_image in enumerate(reconst_images):
        vutils.save_image(
            vutils.make_grid(reconst_image, normalize=True),
            fp=figure_save_path / f"reconst_images{i*save_interval:03}.png",
        )

    torch.save(model.state_dict(), model_save_path / "model.pth")


if __name__ == "__main__":
    main()
