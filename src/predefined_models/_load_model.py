from enum import Enum
from pathlib import Path
from typing import Any

import torch

from ..configs.model_configs.base_configs import (
    NetworkConfig,
    RecursiveDataclass,
)
from ..mytyping import Model, ModelName
from ..utilities import find_project_root
from ._MLP import MLP
from ._ResNetVAE import DownSamplingResNet, ResNetVAE, UpSamplingResNet
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


class LoadModel(Enum):
    MLP = MLP
    DOWNSAMPLINGRESNET = DownSamplingResNet
    UPSAMPLINGRESNET = UpSamplingResNet
    RESNETVAE = ResNetVAE
    SECAE32 = SECAE32
    SECAE64 = SECAE64
    SECVAE64 = SECVAE64
    SECVAESOFTPLUS64 = SECVAEsoftplus64
    SIMPLECAE16 = SimpleCAE16
    SIMPLECAE32 = SimpleCAE32
    SIMPLECAE64 = SimpleCAE64
    SIMPLECAE128 = SimpleCAE128
    SIMPLECVAE64 = SimpleCVAE64
    SIMPLECVAESOFTPLUS32 = SimpleCVAEsoftplus32
    SIMPLECVAESOFTPLUS64 = SimpleCVAEsoftplus64

    @classmethod
    def load_model(
        cls,
        model_name: ModelName,
        model_cfg: dict[str, Any] | RecursiveDataclass,
        pretrained_params_path: str | Path | None = None,
    ) -> Model:
        if isinstance(model_cfg, RecursiveDataclass):
            model_cfg = model_cfg.to_dict()

        try:
            model: Model = cls[model_name.upper()].value(**model_cfg)
        except KeyError as e:
            raise KeyError(f"Unmatched arguments entered: {e}") from e
        if pretrained_params_path is not None:
            print("loading parameter to model")
            root = find_project_root()
            model.load_state_dict(
                torch.load(
                    root / "models" / pretrained_params_path,
                )
            )
        return model

    @classmethod
    def load_model_from_config(cls, cfg: NetworkConfig) -> Model:
        return cls.load_model(
            cfg.network_type, cfg.hyper_parameters, cfg.pretrained_path
        )
