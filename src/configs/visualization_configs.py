import datetime
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from ..utilities import find_project_root
from .model_configs import ExtractConfig


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


@dataclass
class FeatureDatetimeCfg:
    running_date: datetime.date
    running_time: datetime.time

    def running_datetime(self, f: str = "%Y-%m-%d/%H-%M-%S") -> str:
        dt = datetime.datetime(
            self.running_date.year,
            self.running_date.month,
            self.running_date.day,
            self.running_time.hour,
            self.running_time.minute,
            self.running_time.second,
        )
        return dt.strftime(f)

    def __post_init__(self) -> None:
        run_datetime = self.running_datetime()
        root = find_project_root(relative=True)
        feature_dir = root / "reports/features"
        res = list(feature_dir.glob(f"*/{run_datetime}/.hydra"))
        if res == []:
            raise FileNotFoundError(
                "Could not find the directory at ",
                "the specified run date and time.\n",
                f": `{self.running_datetime('%Y/%m/%d %H:%M:%S')}`",
            )

        self.feature_path = res[0].parent
        with initialize(str(".." / res[0]), version_base=None):
            _cfg = compose("config")
        self.extract_cfg = ExtractConfig.from_dictconfig(_cfg)

    @classmethod
    def from_datetime(
        cls, year: int, month: int, day: int, hour: int, minute: int, sec: int
    ) -> "FeatureDatetimeCfg":
        return cls(
            datetime.date(year, month, day), datetime.time(hour, minute, sec)
        )
