# Standard Library
from pathlib import Path
from typing import TypedDict

# Third Party Library
import hydra
import torch
import torch.nn.functional as F
import torch.utils.data
from omegaconf import DictConfig
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import binary_accuracy, multiclass_accuracy

from src.configs.model_configs import TrainClassificationModel
from src.loss_function import LossFunction
from src.mytyping import Device
from src.predefined_models import LoadModel
from src.utilities import (
    EarlyStopping,
    display_cfg,
    get_dataloader,
)


class TrainReturnDict(TypedDict):
    loss: list[float]
    feature_map: list[torch.Tensor]
    target: list[torch.Tensor]
    predicted: list[torch.Tensor]


def _mean(vals: list[float]) -> float:
    if vals == []:
        return float("nan")
    return sum(vals) / len(vals)


def _write_loss_progress(
    writer: SummaryWriter,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    train_accuracy: float,
    valid_accuracy: float,
) -> SummaryWriter:
    # 訓練時
    writer.add_scalar("Loss/Training/Total", train_loss, epoch)
    writer.add_scalar("Accuracy/Training", train_accuracy, epoch)

    # 検証時
    writer.add_scalar("Loss/Valid/Total", valid_loss, epoch)
    writer.add_scalar("Accuracy/Valid", valid_accuracy, epoch)

    return writer


def _pass_through_one_epoch(
    feature: torch.nn.Module,
    classifier: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: LossFunction,
    device: Device,
    num_classes: int = 1,
    optimizer: optim.Optimizer | None = None,
) -> TrainReturnDict:
    recorder: TrainReturnDict = {
        "loss": [],
        "feature_map": [],
        "target": [],
        "predicted": [],
    }
    for x, classes in dataloader:
        x = x.to(device)
        feature_map = feature(x)
        if isinstance(feature_map, (tuple, list)):
            feature_map = feature_map[0]
        pred: torch.Tensor = classifier(feature_map)
        if num_classes != 1:
            t = F.one_hot(classes, num_classes).to(
                device=device, dtype=torch.float32
            )
        else:
            t = classes.to(device=device, dtype=torch.float32)
        loss = criterion.forward(pred, t)[0]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        recorder["loss"].append(loss.detach().cpu().item())
        recorder["feature_map"].append(feature_map.detach().cpu())
        recorder["target"].append(classes.detach().cpu())
        recorder["predicted"].append(pred.detach().cpu())

    return recorder


def _train_model(
    model: torch.nn.Sequential,
    dataloaders: tuple[
        torch.utils.data.DataLoader, torch.utils.data.DataLoader
    ],
    criterion: LossFunction,
    optimizer: optim.Optimizer,
    device: Device,
    max_epoch: int,
    writer: SummaryWriter,
    early_stopping: EarlyStopping | None,
    num_classes: int = 1,
    progress_interval: int = 10,
) -> None:
    for epoch in range(1, max_epoch + 1):
        # training
        model.train()
        train_recorder = _pass_through_one_epoch(
            model[0],
            model[1],
            dataloaders[0],
            criterion,
            device,
            num_classes,
            optimizer,
        )

        # validation
        model.eval()
        with torch.no_grad():
            valid_recorder = _pass_through_one_epoch(
                model[0],
                model[1],
                dataloaders[1],
                criterion,
                device,
                num_classes,
            )

        mean_train_loss = _mean(train_recorder["loss"])
        mean_valid_loss = _mean(valid_recorder["loss"])

        if num_classes == 1:
            train_acc = binary_accuracy(
                torch.concat(train_recorder["predicted"]),
                torch.concat(train_recorder["target"]),
            ).item()

            valid_acc = binary_accuracy(
                torch.concat(valid_recorder["predicted"]),
                torch.concat(valid_recorder["target"]),
            ).item()
        else:
            train_acc = multiclass_accuracy(
                torch.concat(train_recorder["predicted"]),
                torch.concat(train_recorder["target"]),
                average="micro",
                num_classes=num_classes,
            ).item()

            valid_acc = multiclass_accuracy(
                torch.concat(valid_recorder["predicted"]),
                torch.concat(valid_recorder["target"]),
                average="micro",
                num_classes=num_classes,
            ).item()

        _write_loss_progress(
            writer,
            epoch,
            mean_train_loss,
            mean_valid_loss,
            train_acc,
            valid_acc,
        )
        print(
            "Epoch: {}/{} |Train loss: {:.3f} |Valid loss: {:.3f}".format(
                epoch,
                max_epoch,
                mean_train_loss,
                mean_valid_loss,
            )
        )
        if epoch % progress_interval == 0 or epoch == 1:
            writer.add_embedding(
                torch.concat(valid_recorder["feature_map"]),
                metadata=torch.concat(valid_recorder["target"]),
                global_step=epoch,
                tag="FeatureMap",
            )
            writer.flush()

        if early_stopping is not None:
            early_stopping(mean_valid_loss, model)
            if early_stopping.early_stop:
                break


@hydra.main(
    version_base=None,
    config_path="../configs/train_conf/classifilation",
    config_name="ResNet",
)
def main(_cfg: DictConfig) -> None:
    # Display Configuration
    display_cfg(_cfg)
    cfg = TrainClassificationModel.from_dictconfig(_cfg)

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

    if save_interval == 0:
        save_interval = 1

    # GPUが使える環境であればCPUでなくGPUを使うようにする設定
    # Use GPU instead of CPU if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの選択とハイパラの設定
    # Model selection and hyperparameter configuration from configs.yaml
    feature = LoadModel.load_model(
        cfg.model.feature.network_type,
        cfg.model.feature.hyper_parameters,
    )
    classifier = LoadModel.load_model(
        cfg.model.classifier.network_type,
        cfg.model.classifier.hyper_parameters,
    )
    model = torch.nn.Sequential(feature, classifier).to(device)

    # Early stoppingの宣言
    # 過学習を防ぐためにepochの途中で終了する
    # Declaration of Early stopping
    if cfg.train.train_hyperparameter.early_stopping is not None:
        early_stopping = EarlyStopping(
            patience=cfg.train.train_hyperparameter.early_stopping,
            save_path=model_save_path / "model_parameters_ea.pth",
        )
    else:
        early_stopping = None

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

    # クラスの数
    model_hparams = cfg.model.classifier.hyper_parameters
    if model_hparams.output_dimension is not None:
        num_classes = model_hparams.output_dimension
    elif model_hparams.output_channels is not None:
        num_classes = model_hparams.output_channels
    else:
        num_classes = 1

    # オプティマイザの設定
    optimizer = optim.Adam(
        model.parameters(),
        cfg.train.train_hyperparameter.lr,
        weight_decay=cfg.train.train_hyperparameter.weight_decay,
    )

    # TensorBoard による学習経過の追跡
    # Managing learning progress with TensorBoard.
    writer = SummaryWriter(model_save_path / "run-tb")

    try:
        # Training start
        _train_model(
            model,
            (train_dataloader, val_dataloader),
            criterion,
            optimizer,
            device,
            cfg.train.train_hyperparameter.epochs,
            writer,
            early_stopping,
            num_classes,
            save_interval,
        )
        # for epoch in range(1, cfg.train.train_hyperparameter.epochs + 1):
        #     train_losses: list[float] = []

        #     valid_losses: list[float] = []
        #     valid_classes: list[torch.Tensor] = []
        #     valid_feature_maps: list[torch.Tensor] = []

        #     # 訓練
        #     model.train()
        #     for x, classes in train_dataloader:
        #         # データをGPUに移動
        #         x = x.to(device)
        #         feature_map = feature(x)
        #         if isinstance(feature_map, (tuple, list)):
        #             feature_map = feature_map[0]
        #         pred = classifier(feature_map)
        #         if num_classes != 1:
        #             classes = F.one_hot(classes, num_classes).to(
        #                 device=device, dtype=torch.float32
        #             )

        #         # 損失の計算
        #         loss = criterion.forward(pred, classes)[0]

        #         # 重みの更新
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         # 損失をリストに保存
        #         train_losses.append(loss.cpu().item())

        #     # 検証
        #     with torch.no_grad():
        #         model.eval()
        #         for x, classes in val_dataloader:
        #             x = x.to(device)
        #             feature_map = feature(x)
        #             if isinstance(feature_map, (tuple, list)):
        #                 feature_map = feature_map[0]
        #             if num_classes != 1:
        #                 classes = F.one_hot(classes, num_classes).to(
        #                     device=device, dtype=torch.float32
        #                 )
        #             pred = classifier(feature_map)
        #             loss = criterion.forward(pred, classes)[0]

        #             # 損失をリストに保存
        #             valid_losses.append(loss.cpu().item())
        #             valid_classes.append(classes)
        #             valid_feature_maps.append(feature_map)

        #     # 1エポックでの損失の平均
        #     mean_train_loss = _mean(train_losses)
        #     mean_valid_loss = _mean(valid_losses)

        #     # 損失の経過を記録
        #     _write_loss_progress(
        #         writer,
        #         epoch,
        #         mean_train_loss,
        #         mean_valid_loss,
        #     )

        #     print(
        #         "Epoch: {}/{} |Train loss: {:.3f} |Valid loss: {:.3f}".format(
        #             epoch,
        #             cfg.train.train_hyperparameter.epochs,
        #             mean_train_loss,
        #             mean_valid_loss,
        #         )
        #     )

        #     # `feature` により抽出された特徴量のマッピング
        #     if epoch % save_interval == 0 or epoch == 1:
        #         writer.add_embedding(
        #             torch.concat(valid_feature_maps),
        #             metadata=torch.concat(valid_classes),
        #             global_step=epoch,
        #             tag="FeatureMap",
        #         )
        #         writer.flush()

        #     if cfg.train.train_hyperparameter.early_stopping:
        #         early_stopping(
        #             mean_valid_loss,
        #             model,
        #             save_path=model_save_path
        #             / "early_stopped_model_parameters.pth",
        #         )
        #         if early_stopping.early_stop:
        #             break
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt(f"Training was stopped: {e}") from e
    finally:
        writer.close()
        torch.save(
            model.state_dict(), model_save_path / "model_parameters.pth"
        )

    print("Completed without any happening.")


if __name__ == "__main__":
    main()
