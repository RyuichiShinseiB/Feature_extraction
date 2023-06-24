# Standard Library
from dataclasses import dataclass
from typing import Any, Literal

# Third Party Library
import hydra
from omegaconf import OmegaConf

# Local Library
from .. import ActivationName, ModelName, TransformsNameValue


@dataclass
class HyperParameterConfig:
    input_channels: int = 1
    latent_dimensions: int = 128
    encoder_base_channels: int = 64
    decoder_base_channels: int = 64
    encoder_activation: ActivationName = "relu"
    decoder_activation: ActivationName = "relu"
    encoder_output_activation: ActivationName = "relu"
    decoder_output_activation: ActivationName = "sigmoid"


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
    num_save_reconst_image: int = 5
    early_stopping: bool = False
    trained_save_path: str = "./model"


@dataclass
class DatasetConfig:
    image_target: Literal["CNTForest", "CNTPaint"] = "CNTForest"
    path: str = "../../data/processed/CNTForest/cnt_sem_64x64/10k"
    transform: TransformsNameValue = {
        "Grayscale": 1,
        "RandomVerticalFlip": 0.5,
        "RandomHorizontalFlip": 0.5,
        "ToTensor": None,
    }


@dataclass
class MyConfig:
    model_cfg: ModelConfig = ModelConfig()
    train_cfg: TrainConfig = TrainConfig()
    dataset_cfg: DatasetConfig = DatasetConfig()
    hydra: Any | None = None


@hydra.main(version_base=None, config_path="conf", config_name="configs")
def main(cfg: MyConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    # Standard Library
    from dataclasses import asdict
    from pprint import pprint

    pprint(asdict(MyConfig()))
