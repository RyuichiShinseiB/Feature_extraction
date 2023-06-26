# Standard Library
from dataclasses import asdict

# Third Party Library
import torch
from omegaconf import DictConfig, OmegaConf

# First Party Library
from src.configs.model_configs import ModelConfig

# Local Library
from .. import Device
from ._SECAE32 import SECAE32
from ._SECAE64 import SECAE64
from ._SEConvVAE64 import SECVAE64
from ._SEConvVAE_softplus64 import SECVAEsoftplus64
from ._SimpleCAE32 import SimpleCAE32
from ._SimpleCAE64 import SimpleCAE64
from ._SimpleCAE128 import SimpleCAE128
from ._SimpleConvVAE64 import SimpleCVAE64
from ._SimpleConvVAE_softplus64 import SimpleCVAEsoftplus64


def set_hyper_parameters(
    model_cfg: ModelConfig | DictConfig,
) -> dict[str, int | str]:
    if isinstance(model_cfg, ModelConfig):
        return asdict(model_cfg.hyper_parameters)
    elif isinstance(model_cfg, DictConfig):
        return OmegaConf.to_object(model_cfg.hyper_parameters)
    else:
        raise ValueError("model_cfg is not ModelConfig or DictConfig")


def model_define(
    model_cfg: ModelConfig | DictConfig, device: Device = "cpu"
) -> (
    torch.nn.Module
    # SECAE32
    # | SECAE64
    # | SECVAE64
    # | SECVAE_softplus64
    # | SimpleCAE32
    # | SimpleCAE64
    # | SimpleCAE128
    # | SimpleCVAE64
    # | SimpleCVAE_softplus64
):
    model_name = model_cfg.name
    hyper_parameters = set_hyper_parameters(model_cfg)

    if model_name == "SECAE32":
        return SECAE32(**hyper_parameters, device=device)
    elif model_name == "SECAE64":
        return SECAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAE64":
        return SECVAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAEsoftplus64":
        return SECVAEsoftplus64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE32":
        return SimpleCAE32(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE64":
        return SimpleCAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE128":
        return SimpleCAE128(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE64":
        return SimpleCVAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE_softplus64":
        return SimpleCVAEsoftplus64(**hyper_parameters, device=device)
    else:
        raise RuntimeError(f'There is no defined model such as "{model_name}"')
