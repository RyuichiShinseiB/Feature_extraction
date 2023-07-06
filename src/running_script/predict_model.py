# Standard Library
import os
from pathlib import Path

# Third Party Library
import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from torchinfo import summary

# First Party Library
from src import Tensor
from src.configs.model_configs import MyConfig, dictconfig2dataclass
from src.predefined_models import model_define
from src.utilities import get_dataloader


@hydra.main(
    version_base=None,
    config_path="../configs/train_conf",
    config_name="configs",
)
def main(_cfg: DictConfig) -> None:
    cfg = dictconfig2dataclass(_cfg, MyConfig)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 事前学習済みのモデルのロード
    # Load pretrained model
    model = model_define(cfg.model, device=device).to(device)
    # model.load_state_dict(torch.load(cfg.pretrained_path))
    summary(model, (1, 1, 32, 32))

    # データローダーを設定
    dataloader = get_dataloader(
        cfg.dataset.path,
        cfg.dataset.transform,
        cfg.train.batch_size,
        generator_seed=None,
    )

    features_list: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            _, features = model(x.to(device))
            features_list.extend(
                torch.flatten(features, start_dim=1).detach().cpu().tolist()
            )

            # features_list.append(torch.flatten(features, start_dim=1).)


if __name__ == "__main__":
    main()
