from enum import Enum
from pathlib import Path
from typing import Any

import torch

from ..mytyping import Model, ModelName
from ._ResNetVAE import DownSamplingResNet, ResNetVAE
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
    DOWNSAMPLINGRESNET = DownSamplingResNet
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
        model_cfg: dict[str, Any],
        pretrained_params_path: str | Path | None = None,
    ) -> Model:
        try:
            model: Model = cls[model_name.upper()].value(**model_cfg)
        except KeyError as e:
            raise KeyError(f"Unmatched arguments entered: {e}") from e
        if pretrained_params_path is not None:
            model.load_state_dict(
                torch.load(
                    pretrained_params_path,
                )
            )
        return model
