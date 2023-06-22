# Standard Library
from typing import Literal, TypeAlias

# Third Party Library
import torch

ActivationName = Literal[
    "relu", "selu", "leakyrelu", "sigmoid", "tanh", "identity"
]
Device: TypeAlias = Literal["cpu", "cuda"] | torch.device
Tensor: TypeAlias = torch.Tensor
