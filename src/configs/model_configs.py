# Standard Library
from dataclasses import dataclass
from typing import Any, Literal

# Third Party Library
import hydra
from omegaconf import OmegaConf

ModelName = Literal[
    "SECAE32",
    "SECAE64",
    "SECVAE64",
    "SECVAE_softplus64",
    "SimpleCAE32",
    "SimpleCAE64",
    "SimpleCAE128",
    "SimpleCVAE64",
    "SimpleCVAE_softplus64",
]
ActivationName = Literal[
    "relu", "selu", "leakyrelu", "sigmoid", "tanh", "identity"
]


@dataclass
class HyperParameterConfig:
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActivationName = "relu"
    decoder_activation: ActivationName = "relu"
    encoder_output_activation: ActivationName = "relu"
    decoder_output_activation: ActivationName = "relu"


@dataclass
class ModelConfig:
    name: ModelName = "SimpleCAE64"
    hyper_parameters: HyperParameterConfig = HyperParameterConfig()


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    loss: str = "bce"
    early_stopping: bool = False


@dataclass
class MyConfig:
    model_cfg: ModelConfig = ModelConfig()
    train_cfg: TrainConfig = TrainConfig()
    dataset_path: str = ""
    trained_save_path: str = "model"
    hydra: Any | None = None


@hydra.main(version_base=None, config_path="conf", config_name="configs")
def main(cfg: MyConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    # Standard Library
    from dataclasses import asdict
    from pprint import pprint

    pprint(asdict(MyConfig()))
