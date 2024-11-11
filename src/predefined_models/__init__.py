# Standard Library
from dataclasses import asdict
from typing import Any

# Third Party Library
from omegaconf import DictConfig, OmegaConf

# First Party Library
from src.configs.model_configs import AutoencoderModelConfig, MAEViTModelConfig

# Local Library
from ..mytyping import Device, Model, ModelName
from ._ResNetVAE import ResNetVAE
from ._SECAE32 import SECAE32
from ._SECAE64 import SECAE64
from ._SEConvVAE64 import SECVAE64
from ._SEConvVAE_softplus64 import SECVAEsoftplus64
from ._SimpleCAE16 import SimpleCAE16
from ._SimpleCAE32 import SimpleCAE32
from ._SimpleCAE64 import SimpleCAE64
from ._SimpleCAE128 import SimpleCAE128
from ._SimpleConvVAE64 import SimpleCVAE64
from ._SimpleConvVAE_softplus32 import SimpleCVAEsoftplus32
from ._SimpleConvVAE_softplus64 import SimpleCVAEsoftplus64


def set_hyper_parameters(
    model_cfg: AutoencoderModelConfig | MAEViTModelConfig,
) -> dict[str, int | str] | Any:
    if isinstance(model_cfg, AutoencoderModelConfig | MAEViTModelConfig):
        return asdict(
            model_cfg.hyper_parameters,
            dict_factory=lambda x: {k: v for (k, v) in x if v is not None},
        )
    elif isinstance(model_cfg, DictConfig):
        return OmegaConf.to_container(model_cfg.hyper_parameters)
    else:
        raise ValueError(
            "model_cfg is not AutoencoderModelConfig, MAEViTModelConfig or DictConfig"  # noqa: E501
        )


def model_define(  # noqa: C901
    model_cfg: AutoencoderModelConfig | MAEViTModelConfig,
    device: Device = "cpu",
) -> Model:
    model_name: ModelName = model_cfg.name
    hyper_parameters = set_hyper_parameters(model_cfg)

    if model_name == "SECAE32":
        return SECAE32(**hyper_parameters, device=device)
    elif model_name == "SECAE64":
        return SECAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAE64":
        return SECVAE64(**hyper_parameters, device=device)
    elif model_name == "SECVAEsoftplus64":
        return SECVAEsoftplus64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE16":
        return SimpleCAE16(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE32":
        return SimpleCAE32(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE64":
        return SimpleCAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCAE128":
        return SimpleCAE128(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE64":
        return SimpleCVAE64(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE_softplus32":
        return SimpleCVAEsoftplus32(**hyper_parameters, device=device)
    elif model_name == "SimpleCVAE_softplus64":
        return SimpleCVAEsoftplus64(**hyper_parameters, device=device)
    elif model_name == "ResNetVAE":
        return ResNetVAE(**hyper_parameters, device=device)
    else:
        raise ValueError(f'There is no defined model such as "{model_name}"')
