from hydra import compose, initialize

from src.configs.model_configs.base_configs import TrainDatasetConfig
from src.utilities._utils import get_dataloader


def test_get_dataloader() -> None:
    with initialize(
        "../src/configs/train_conf/classifilation", version_base=None
    ):
        _cfg = compose("ResNet-highlow")
    cfg = TrainDatasetConfig.from_dictconfig(_cfg.dataset)

    dl, _ = get_dataloader(
        cfg.path,
        cfg.transform,
        split_ratio=(0.8, 0.2),
        batch_size=10,
        shuffle=True,
        generator_seed=42,
        cls_conditions={0: ["1", "2", "3"], 1: ["4", "5", "6"]},
    )

    for i, (_, c, d) in enumerate(dl):
        print("=====================================")
        print(f"class  : {c}")
        print(f"dirname: {d}")
        print("")

        if i > 10:
            break
