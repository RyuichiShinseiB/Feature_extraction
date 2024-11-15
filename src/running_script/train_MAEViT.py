# Standard Library
import os
from pathlib import Path

# Third Party Library
import hydra
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import optim

from src.configs.model_configs import TrainMAEViTConfig, dictconfig2dataclass
from src.loss_function import LossFunction, calc_loss

# First Party Library
from src.mytyping import Tensor
from src.predefined_models import model_define
from src.utilities import (
    EarlyStopping,
    display_cfg,
    get_dataloader,
    weight_init,
)


@hydra.main(
    version_base=None,
    config_path="../configs/train_conf",
    config_name="MAEViT",
)
def main(_cfg: DictConfig) -> None:
    # Display Configuration
    display_cfg(_cfg)
    cfg = dictconfig2dataclass(_cfg, TrainMAEViTConfig)

    # 訓練済みモデル、訓練途中の再構成画像の保存先
    # Paths to store trained models and reconstructed images in training
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = Path("./models") / base_save_path
    figure_save_path = Path("./reports/figures") / base_save_path

    # 再構成画像を保存する間隔
    # Storage interval of reconstructed images
    save_interval = (
        cfg.train.train_hyperparameter.epochs
        // cfg.train.train_hyperparameter.num_save_reconst_image
    )

    # GPUが使える環境であればCPUでなくGPUを使うようにする設定
    # Use GPU instead of CPU if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    # Model selection and hyperparameter configuration from configs.yaml
    model = model_define(cfg.model, device=device).to(device)

    # 重みの初期化
    # Weight initialization
    model.apply(weight_init)

    # Early stoppingの宣言
    # 過学習を防ぐためにepochの途中で終了する
    # Declaration of Early stopping
    early_stopping = EarlyStopping()

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、  # noqa: E501
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    train_dataloader, val_dataloader = get_dataloader(
        cfg.dataset.path,
        cfg.dataset.transform,
        cfg.train.train_hyperparameter.batch_size,
        shuffle=True,
        split_ratio=(0.8, 0.2),
        generator_seed=42,
    )  # type: ignore

    # 損失関数の設定
    criterion = LossFunction(
        cfg.train.train_hyperparameter.reconst_loss,
        cfg.train.train_hyperparameter.latent_loss,
    )

    # オプティマイザの設定
    optimizer = optim.Adam(
        model.parameters(), cfg.train.train_hyperparameter.lr
    )

    reconst_images: list[Tensor] = []
    train_losses: list[float] = []
    valid_losses: list[float] = []

    test_image = next(iter(val_dataloader))[0][:64].to(device)

    # Training start
    for epoch in range(cfg.train.train_hyperparameter.epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # 訓練
        for _i_train, (x, _) in enumerate(train_dataloader, 0):
            # モデルの訓練
            model.train()

            # データをGPUに移動
            x = x.to(device)
            # 損失の計算
            loss, _ = calc_loss(
                input_data=x,
                reconst_loss=criterion.reconst_loss,
                latent_loss=criterion.latent_loss,
                model=model,
            )

            # 重みの更新
            optimizer.zero_grad()
            loss["reconst"].backward()
            optimizer.step()

            # 損失をリストに保存
            train_losses.append(loss["reconst"].cpu().item())
            train_loss += loss["reconst"].cpu().item()

        # 検証
        for _i_valid, (x, _) in enumerate(val_dataloader, 0):
            model.eval()
            x = x.to(device)
            loss, _ = calc_loss(
                input_data=x,
                reconst_loss=criterion.reconst_loss,
                latent_loss=criterion.latent_loss,
                model=model,
            )

            # 損失をリストに保存
            valid_losses.append(loss["reconst"].cpu().item())
            valid_loss += loss["reconst"].cpu().item()

        print(
            "Epoch: {}/{}\t|Train loss: {:.5f}\t|Valid loss: {:.5f}".format(
                epoch + 1,
                cfg.train.train_hyperparameter.epochs,
                train_loss / (_i_train + 1),
                valid_loss / (_i_valid + 1),
            )
        )

        # 再構成できているかを確認する画像の保存
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            model.eval()
            test_output, _ = model(test_image)
            reconst_images.append(
                vutils.make_grid(test_output, normalize=True)
            )

        if cfg.train.train_hyperparameter.early_stopping:
            early_stopping(
                train_loss, model, save_path=model_save_path / "model.pth"
            )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(train_losses)), train_losses, label="train loss")
    ax.set_xlabel("iterations")
    ax.set_ylabel(
        f"{cfg.train.train_hyperparameter.reconst_loss.upper()} loss"
    )
    ax.legend()
    if not figure_save_path.exists():
        os.makedirs(figure_save_path)
    fig.savefig(figure_save_path / "loss.jpg")

    for i, reconst_image in enumerate(reconst_images):
        vutils.save_image(
            reconst_image,
            fp=figure_save_path / f"reconst_images{i*save_interval:03}.png",
        )

    vutils.save_image(
        vutils.make_grid(test_image, normalize=True),
        fp=figure_save_path / "test_images.png",
    )

    torch.save(model.state_dict(), model_save_path / "model_parameters.pth")


if __name__ == "__main__":
    main()
