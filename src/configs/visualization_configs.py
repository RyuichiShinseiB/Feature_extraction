from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class ExtractionModelConfig:
    model_name: str
    image_size: int
    run_datetime: str


@dataclass
class ClusteringModelConfig:
    model_name: str
    max_iter: int
    num_used_features: int
    num_clusters: int
    run_datetime: str


@dataclass
class VisConfigs:
    extraction: ExtractionModelConfig
    clustering: ClusteringModelConfig

    @classmethod
    def load(cls, path: Path | str) -> "VisConfigs":
        path = path if isinstance(path, Path) else Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The path already existed. :{str(path)}")
        _cfg = OmegaConf.load(path)
        if not isinstance(_cfg, DictConfig):
            raise TypeError(
                "The loaded config file expected a dictionary-like structure."
                f"But it was loaded to the following type: {_cfg}"
            )
        return cls.from_dictconfig(_cfg)

    @classmethod
    def from_dictconfig(cls, dictcfg: DictConfig) -> "VisConfigs":
        _cfg = OmegaConf.to_container(dictcfg)
        if not isinstance(_cfg, dict):
            raise TypeError(
                "Expected `dictcfg` to be a Dictconfig with a structure"
                "that would be converted to a dictionary."
                f"But it was converted to the following type: {_cfg}"
            )
        ecfg = ExtractionModelConfig(**_cfg["extraction"])
        ccfg = ClusteringModelConfig(**_cfg["clustering"])
        return cls(ecfg, ccfg)
