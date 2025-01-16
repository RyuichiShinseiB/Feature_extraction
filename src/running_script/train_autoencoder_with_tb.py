# Standard Library
import math
from pathlib import Path

# Third Party Library
import hydra
import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.configs.model_configs import TrainAutoencoderConfig
from src.loss_function import LossFunction
from src.predefined_models._load_model import model_define
from src.utilities import (
    EarlyStopping,
    display_cfg,
    get_dataloader,
    weight_init,
)


def _mean(vals: list[float]) -> float:
    if vals == []:
        return float("nan")
    return sum(vals) / len(vals)


def _write_loss_progress(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    train_reconst: float,
    train_kldiv: float,
    valid_loss: float,
    valid_reconst: float,
    valid_kldiv: float,
) -> SummaryWriter:
    ## 訓練時
    writer.add_scalar("Loss/Training/Total", train_loss, epoch)

    writer.add_scalar("Loss/Training/Reconst", train_reconst, epoch)

    if not math.isnan(train_kldiv):
        writer.add_scalar("Loss/Training/KL-Div.", train_kldiv, epoch)

    ## 検証時
    writer.add_scalar("Loss/Valid/Total", valid_loss, epoch)

    writer.add_scalar("Loss/Valid/Reconst", valid_reconst, epoch)

    if not math.isnan(valid_kldiv):
        writer.add_scalar("Loss/Valid/KL-Div.", valid_kldiv, epoch)

    return writer


@hydra.main(
    version_base=None,
    config_path="../configs/train_conf",
    config_name="ResNetVAE",
    # config_name="SimpleCVAE",
)
def main(_cfg: DictConfig) -> None:
    # Display Configuration
    display_cfg(_cfg)
    cfg = TrainAutoencoderConfig.from_dictconfig(_cfg)

    # 訓練済みモデル、訓練途中の再構成画像の保存先
    # Paths to store trained models and reconstructed images in training
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = Path("./models") / base_save_path
    # figure_save_path = Path("./reports/figures") / base_save_path

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
    early_stopping = EarlyStopping(patience=50)

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、  # noqa: E501
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    train_dataloader, val_dataloader = get_dataloader(
        cfg.dataset.path,
        cfg.dataset.transform,
        split_ratio=(0.8, 0.2),
        batch_size=cfg.train.train_hyperparameter.batch_size,
        shuffle=True,
        generator_seed=42,
    )

    # length_of_dataloader = len(train_dataloader)
    # latent_interval = length_of_dataloader // 10

    # 損失関数の設定
    criterion = LossFunction(
        cfg.train.train_hyperparameter.reconst_loss,
        cfg.train.train_hyperparameter.latent_loss,
    )

    # オプティマイザの設定
    optimizer = optim.Adam(
        model.parameters(), cfg.train.train_hyperparameter.lr
    )

    # TensorBoard による学習経過の追跡
    # Managing learning progress with TensorBoard.
    writer = SummaryWriter(model_save_path / "run-tb")
    # reconst_images: list[Tensor] = []

    test_image: torch.Tensor = next(iter(val_dataloader))[0][:64].to(device)

    writer.add_image(
        "Test_images",
        vutils.make_grid(test_image, normalize=True),
    )

    try:
        # Training start
        for epoch in range(cfg.train.train_hyperparameter.epochs):
            train_losses: list[float] = []
            train_reconst_errors: list[float] = []
            train_kldiv_errors: list[float] = []

            valid_losses: list[float] = []
            valid_reconst_errors: list[float] = []
            valid_kldiv_errors: list[float] = []
            valid_latent_means: list[torch.Tensor] = []
            valid_classes: list[torch.Tensor] = []
            valid_reconst_images: list[torch.Tensor] = []

            # 訓練
            for i, (x, _) in enumerate(train_dataloader, 1):
                # モデルの訓練
                model.train()

                # データをGPUに移動
                x = x.to(device)
                reconst, latent_params = model(x)
                # 損失の計算
                errors = criterion.forward(reconst, x, latent_params)
                loss = errors[0] + criterion.calc_weight(errors[0]) * errors[1]
                # if i % latent_interval == 0:
                #     loss += errors.get("kldiv", 0)

                # 重みの更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 損失をリストに保存
                train_losses.append(loss.cpu().item())
                train_reconst_errors.append(errors[0].cpu().item())
                if errors[1] is not None:
                    train_kldiv_errors.append(errors[1].cpu().item())

            # 検証
            with torch.no_grad():
                for x, classes in val_dataloader:
                    model.eval()
                    x = x.to(device)
                    reconst, latent_params = model(x)
                    errors = criterion.forward(reconst, x, latent_params)
                    loss = (
                        errors[0]
                        + criterion.calc_weight(errors[0]) * errors[1]
                    )

                    # 損失をリストに保存
                    valid_losses.append(loss.cpu().item())
                    valid_reconst_errors.append(errors[0].cpu().item())
                    if errors[1] is not None:
                        valid_kldiv_errors.append(errors[1].cpu().item())
                    valid_latent_means.append(latent_params[0])
                    valid_classes.append(classes)
                    valid_reconst_images.append(reconst)

            # 1エポックでの損失の平均
            mean_train_loss = _mean(train_losses)
            mean_train_reconst_error = _mean(train_reconst_errors)
            mean_train_kldiv_error = _mean(train_kldiv_errors)
            mean_valid_loss = _mean(valid_losses)
            mean_valid_reconst_error = _mean(valid_reconst_errors)
            mean_valid_kldiv_error = _mean(valid_kldiv_errors)

            # 損失の経過を記録
            _write_loss_progress(
                writer,
                epoch,
                mean_train_loss,
                mean_train_reconst_error,
                mean_train_kldiv_error,
                mean_valid_loss,
                mean_valid_reconst_error,
                mean_valid_kldiv_error,
            )

            print(
                "Epoch: {}/{} |Train loss: {:.3f} |Valid loss: {:.3f}".format(
                    epoch + 1,
                    cfg.train.train_hyperparameter.epochs,
                    mean_train_loss,
                    mean_valid_loss,
                )
            )

            # 再構成できているかを確認する画像の保存
            with torch.no_grad():
                if (epoch + 1) % save_interval == 0 or epoch == 0:
                    model.eval()
                    test_output, _ = model(test_image)
                    writer.add_image(
                        "Reconstructed_images",
                        vutils.make_grid(
                            test_output,
                            normalize=True,
                        ),
                        epoch + 1,
                    )

                    writer.add_embedding(
                        torch.concat(valid_latent_means),
                        metadata=torch.concat(valid_classes),
                        global_step=epoch,
                        tag="DistributionOnLatentSpace",
                    )
                    writer.flush()

            if cfg.train.train_hyperparameter.early_stopping:
                early_stopping(
                    mean_valid_loss,
                    model,
                    save_path=model_save_path
                    / "early_stopped_model_parameters.pth",
                )
                if early_stopping.early_stop:
                    break
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt(f"Training was stopped: {e}") from e
    finally:
        writer.close()
        torch.save(
            model.state_dict(), model_save_path / "model_parameters.pth"
        )

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(range(len(train_losses)), train_losses, label="train loss")
    # ax.set_xlabel("iterations")
    # ax.set_ylabel(
    #     f"{cfg.train.train_hyperparameter.reconst_loss.upper()} loss"
    # )
    # ax.legend()
    # if not figure_save_path.exists():
    #     os.makedirs(figure_save_path)
    # fig.tight_layout()
    # fig.savefig(figure_save_path / "loss.jpg")

    # for i, reconst_image in enumerate(reconst_images):
    #     vutils.save_image(
    #         reconst_image,
    #         fp=figure_save_path / f"reconst_images{i*save_interval:03}.png",
    #     )

    # vutils.save_image(
    #     vutils.make_grid(test_image, normalize=True),
    #     fp=figure_save_path / "test_images.png",
    # )


if __name__ == "__main__":
    main()
