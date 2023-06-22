# Standard Library
from dataclasses import asdict
from typing import Literal, TypeAlias

# Third Party Library
import torch

# First Party Library
from src.configs.model_configs import ModelConfig

# Local Library
from ._SECAE32 import SECAE as SECAE32
from ._SECAE64 import SECAE as SECAE64
from ._SEConvVAE64 import SECVAE as SECVAE64
from ._SEConvVAE_softplus64 import SECVAE as SECVAE_softplus64
from ._SimpleCAE32 import SimpleCAE as SimpleCAE32
from ._SimpleCAE64 import SimpleCAE as SimpleCAE64
from ._SimpleCAE128 import SimpleCAE as SimpleCAE128
from ._SimpleConvVAE64 import SimpleCVAE as SimpleCVAE64
from ._SimpleConvVAE_softplus64 import SimpleCVAE as SimpleCVAE_softplus64

ActivationName = Literal[
    "relu", "selu", "leakyrelu", "sigmoid", "tanh", "identity"
]
Device: TypeAlias = Literal["cpu", "cuda"] | torch.device
Tensor: TypeAlias = torch.Tensor


def model_define(
    model_cfg: ModelConfig, device: Device = "cpu"
) -> (
    SECAE32
    | SECAE64
    | SECVAE64
    | SECVAE_softplus64
    | SimpleCAE32
    | SimpleCAE64
    | SimpleCAE128
    | SimpleCVAE64
    | SimpleCVAE_softplus64
):
    model_name = model_cfg.name
    hyper_parameters = asdict(model_cfg.hyper_parameters)
    if model_name == "SECAE32":
        return SECAE32(**hyper_parameters, device=device)
    elif model_name == "SECAE64":
        return SECAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAE64":
        return SECVAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAE_softplus64":
        return SECVAE_softplus64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE32":
        return SimpleCAE32(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE64":
        return SimpleCAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE128":
        return SimpleCAE128(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE64":
        return SimpleCVAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE_softplus64":
        return SimpleCVAE_softplus64(**hyper_parameters, device=device)
    else:
        raise RuntimeError(f'There is no defined model such as "{model_name}"')
