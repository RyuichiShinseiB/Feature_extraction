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
from src.utilities import (
    EarlyStopping,
    get_dataloader,
    train_step,
    weight_init,
)


@hydra.main(version_base=None, config_path="", config_name="")
def main(cfg: MyConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_define(cfg.model_cfg, device=device)
    model.apply(weight_init)

    early_stopping: Optional[EarlyStopping]
    if cfg.train_cfg.early_stopping:
        early_stopping = EarlyStopping()
    else:
        early_stopping = None

    dataloader = get_dataloader(
        cfg.dataset_path, cfg.train_cfg.batch_size, "train"
    )

    criterion: nn.BCELoss | nn.MSELoss | Any
    if cfg.train_cfg.loss == "bce":
        criterion = nn.BCELoss()
    elif cfg.train_cfg.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise RuntimeError("Please select another loss function")

    optimizer = optim.Adam(model.parameters(), cfg.train_cfg.lr)

    reconst_images: list[Tensor] = []
    train_losses: list[float] = []
    for epoch in range(cfg.train_cfg.epochs):
        train_loss = 0.0
        test_loss = 0.0

        for i, (x, _) in enumerate(dataloader, 0):
            model.train()
            loss, x_pred = train_step(x, model, criterion, optimizer, device)
            train_losses.append(loss.cpu().item())
            train_loss += loss.cpu().item()

        early_stopping(train_loss, model, cfg.trained_save_path)
