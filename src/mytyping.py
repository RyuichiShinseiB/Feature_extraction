# Standard Library
from typing import Literal, TypeAlias

# Third Party Library
import torch
from torchvision import transforms

Model: TypeAlias = torch.nn.Module
Device: TypeAlias = Literal["cpu", "cuda"] | torch.device
Tensor: TypeAlias = torch.Tensor
ModelName = Literal[
    # Autoencoder type model
    "SECAE32",
    "SECAE64",
    "SECVAE64",
    "SECVAEsoftplus64",
    "SimpleCAE16",
    "SimpleCAE32",
    "SimpleCAE64",
    "SimpleCAE128",
    "SimpleCVAE64",
    "SimpleCVAEsoftplus32",
    "SimpleCVAEsoftplus64",
    "ResNetVAE"
    # ViT type model
    "MAEViT",
]

ActivationName = Literal[
    "relu", "selu", "leakyrelu", "sigmoid", "tanh", "identity"
]

TransformsName = Literal[
    "Grayscale",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "Normalize",
    "ToTensor",
]

TransformsNameValue = dict[
    TransformsName, int | float | tuple[float, float] | None
]

Transforms: TypeAlias = (
    transforms.Grayscale
    | transforms.RandomVerticalFlip
    | transforms.RandomHorizontalFlip
    | transforms.Normalize
    | transforms.ToTensor
)
