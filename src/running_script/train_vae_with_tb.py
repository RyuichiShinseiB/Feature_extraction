# Standard Library
import math
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import hydra
import torch
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.configs.model_configs import TrainVAEConfig
from src.loss_function import LossFunction
from src.mytyping import Device, Tensor
from src.predefined_models._build_VAE import VAEFrame
from src.utilities import (
    EarlyStopping,
    display_cfg,
    weight_init,
)


@dataclass
class TrainResults:
    originals: list[Tensor] = []
    reconsts: list[Tensor] = []
    latent_features: list[Tensor] = []
    targets: list[Tensor] = []
    reconst_errs: list[float] = []
    kldivs: list[float] = []
    losses: list[float] = []

    def appends(
        self,
        original: Tensor,
        reconst: Tensor,
        targets: Tensor,
        latent_params: tuple[Tensor, Tensor],
        errors: tuple[Tensor, Tensor],
        loss: Tensor,
    ) -> None:
        self.originals.append(original.detach().cpu())
        self.reconsts.append(reconst.detach().cpu())
        self.targets.append(targets.detach().cpu())
        self.latent_features.append(latent_params[0].detach().cpu())
        self.reconst_errs.append(errors[0].detach().cpu().item())
        self.kldivs.append(errors[1].detach().cpu().item())
        self.losses.append(loss.detach().cpu().item())

    def calc_loss_and_error_mean(self) -> tuple[float, float, float]:
        mean_loss = _mean(self.losses)
        mean_reconst_err = _mean(self.reconst_errs)
        mean_kldiv = _mean(self.kldivs)
        return mean_loss, mean_reconst_err, mean_kldiv


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


def train_one_step(
    model: VAEFrame,
    x: Tensor,
    criterion: LossFunction,
    optimizer: optim.Optimizer,
    device: Device,
) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
    # データをGPUに移動
    x = x.to(device)
    reconst, latent_params = model.forward(x)
    # 損失の計算
    errors = criterion.forward(reconst, x, latent_params)
    loss = errors[0] + criterion.calc_weight(errors[0]) * errors[1]

    # 重みの更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return reconst, latent_params, errors, loss


def pass_through_one_epoch(
    model: VAEFrame,
    dataloader: DataLoader,
    criterion: LossFunction,
    optimizer: optim.Optimizer,
    device: Device,
) -> TrainResults:
    results = TrainResults()
    x: Tensor
    t: Tensor
    for x, t in dataloader:
        reconst, lparas, errs, loss = train_one_step(
            model, x, criterion, optimizer, device
        )
        results.appends(x, reconst, t, lparas, errs, loss)
    return results


def train_model(
    model: VAEFrame,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    test_data: Tensor,
    criterion: LossFunction,
    optimizer: optim.Optimizer,
    device: Device,
    total_epochs: int,
    writer: SummaryWriter,
    early_stopping: EarlyStopping | None,
    save_interval: int,
) -> None:
    for epoch in range(1, total_epochs + 1):
        model.train()
        t_result = pass_through_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        model.eval()
        with torch.no_grad():
            v_result = pass_through_one_epoch(
                model, valid_dataloader, criterion, optimizer, device
            )

        train_means = t_result.calc_loss_and_error_mean()
        valid_means = v_result.calc_loss_and_error_mean()
        _write_loss_progress(
            writer,
            epoch,
            train_means[0],
            train_means[1],
            train_means[2],
            valid_means[0],
            valid_means[1],
            valid_means[2],
        )
        print(
            "Epoch: {}/{} |Train loss: {:.3f} |Valid loss: {:.3f}".format(
                epoch + 1, total_epochs, train_means[0], valid_means[0]
            )
        )
        with torch.no_grad():
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                model.eval()
                test_output, _ = model(test_data)
                writer.add_image(
                    "Reconstructed_images",
                    vutils.make_grid(
                        test_output,
                        normalize=True,
                    ),
                    epoch + 1,
                )

                writer.add_embedding(
                    torch.concat(v_result.latent_features),
                    metadata=torch.concat(v_result.targets),
                    global_step=epoch,
                    tag="DistributionOnLatentSpace",
                )
                writer.flush()

        if early_stopping is not None:
            early_stopping(valid_means[0], model)
            if early_stopping.early_stop:
                break


@hydra.main(
    version_base=None,
    config_path="../configs/train_conf",
    config_name="ResNetVAE_new",
    # config_name="SimpleCVAE",
)
def main(_cfg: DictConfig) -> None:
    # Display Configuration
    display_cfg(_cfg)
    cfg = TrainVAEConfig.from_dictconfig(_cfg)

    # 訓練済みモデル、訓練途中の再構成画像の保存先
    # Paths to store trained models and reconstructed images in training
    base_save_path = Path(cfg.train.trained_save_path)
    model_save_path = Path("./models") / base_save_path
    # figure_save_path = Path("./reports/figures") / base_save_path

    # 再構成画像を保存する間隔
    # Storage interval of reconstructed images
    save_interval = cfg.verbose_interval

    # GPUが使える環境であればCPUでなくGPUを使うようにする設定
    # Use GPU instead of CPU if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    # Model selection and hyperparameter configuration from configs.yaml
    model = cfg.create_vae_model(device)

    # 重みの初期化
    # Weight initialization
    model.apply(weight_init)

    # Early stoppingの宣言
    # 過学習を防ぐためにepochの途中で終了する
    # Declaration of Early stopping
    if cfg.train.train_hyperparameter.early_stopping is not None:
        early_stopping = EarlyStopping(
            patience=cfg.train.train_hyperparameter.early_stopping,
            save_path=model_save_path / "early_stopped_model_parameters.pth",
        )
    else:
        early_stopping = None

    # データローダーを設定
    # split_ratioを設定していると（何かしら代入していると）、データセットを分割し、  # noqa: E501
    # 訓練用と検証用のデータローダーを作製する。
    # generator_seedは、データセットを分割するときのseed値
    train_dataloader, val_dataloader = cfg.create_dataloader((0.8, 0.2))

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

    test_image: Tensor = next(iter(val_dataloader))[0][:64].to(device)

    writer.add_image(
        "Test_images",
        vutils.make_grid(test_image, normalize=True),
    )

    try:
        # Training start
        train_model(
            model,
            train_dataloader,
            val_dataloader,
            test_image,
            criterion,
            optimizer,
            device,
            cfg.train.train_hyperparameter.epochs,
            writer,
            early_stopping,
            save_interval,
        )
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt(f"Training was stopped: {e}") from e
    finally:
        writer.close()
        torch.save(
            model.state_dict(), model_save_path / "model_parameters.pth"
        )
    print("Completed to train")


if __name__ == "__main__":
    main()
