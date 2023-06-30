# Standard Library
import os
from pathlib import Path

# Third Party Library
import torch
from omegaconf import OmegaConf

# First Party Library
from src.predefined_models import model_define

cfg = OmegaConf.load(
    "/home/shinsei/MyResearchs/feat_extrc/models/SimpleCAE64/2023-06-30/17-23-20/.hydra/config.yaml"
)
print(OmegaConf.to_container(cfg))
