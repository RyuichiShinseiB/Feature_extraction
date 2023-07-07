# Standard Library
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Optional, cast

# Third Party Library
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset, pil_loader
from torchvision.transforms import transforms

# Local Library
from . import Model, Tensor, Transforms, TransformsName, TransformsNameValue
from .configs.model_configs import MyConfig

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def weight_init(m: Any) -> None:
    """weight initializer


    Parameters
    ----------
    m : Any
        Layer of model

    Examples:
    >>> model = Network()
    >>> model.apply(weight_init)

    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def get_dataloader(
    dataset_path: str,
    dataset_transform: TransformsNameValue,
    batch_size: int = 64,
    shuffle: bool = True,
    split_ratio: tuple[float, float] | None = None,
    generator_seed: int | None = 42,
    extraction: bool = False,
) -> tuple[DataLoader, DataLoader] | DataLoader:
    """specifying parameters and output dataloader

    Parameters
    ----------
    dataset_path : str
        Path to dataset
    dataset_transform : TransformsNameValue
        dictionary type parameters. {"transform": parameter}
    batch_size : int, optional
        Unit of data used per study, by default 64
    shuffle : bool
        The dataloader is shuffled if True, by default True
    split_ratio : tuple[float, float] | None, optional
        Split training and validation if not None, by default None

    Returns
    -------
    DataLoader
    """
    transform = transforms.Compose(
        [
            str2transform(name, value)
            for name, value in dataset_transform.items()
        ]
    )
    if extraction:
        dataset = ForExtractFolder(dataset_path, transform=transform)
    else:
        dataset = ImageFolder(dataset_path, transform)

    if split_ratio is not None:
        splitted_dataset: list[
            Subset[ForExtractFolder | ImageFolder]
        ] = random_split(
            dataset,
            split_ratio,
            generator=torch.Generator().manual_seed(generator_seed)
            if generator_seed is not None
            else None,
        )
        return (
            DataLoader(
                splitted_dataset[0], batch_size, shuffle, num_workers=2
            ),
            DataLoader(
                splitted_dataset[1], batch_size, shuffle, num_workers=2
            ),
        )
    else:
        return DataLoader(dataset, batch_size, shuffle, num_workers=2)


def str2transform(
    transforms_name: TransformsName,
    transforms_param: Any,
) -> Transforms:
    """Call torchvision.transforms from string

    Parameters
    ----------
    transforms_name : TransformsName
        Set the name of the transformation method included
        in torchvision.transforms.\n
        Settable methods are Grayscale, RandomVerticalFlip,
        RandomHorizontalFlip, Normalize and ToTensor
    transforms_param : Any
        Parameters of transformation method

    Returns
    -------
    Transforms
        `torchvision.transforms.<transforms_name>`

    Raises
    ------
    RuntimeError
        If an undefined transformation method is specified.
    """
    if transforms_name == "Grayscale":
        return transforms.Grayscale(transforms_param)
    elif transforms_name == "RandomVerticalFlip":
        return transforms.RandomVerticalFlip(transforms_param)
    elif transforms_name == "RandomHorizontalFlip":
        return transforms.RandomHorizontalFlip(transforms_param)
    elif transforms_name == "Normalize":
        return transforms.Normalize(transforms_param[0], transforms_param[1])
    elif transforms_name == "ToTensor":
        return transforms.ToTensor()
    else:
        raise RuntimeError(
            f'There is no torchvision.transforms such as "{transforms_name}"'
        )


def calc_loss(
    x: Tensor,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    model: Model,
) -> tuple[Tensor, Tensor]:
    x_pred, _ = model(x)
    loss = loss_func(x_pred, x)
    return loss, x_pred


def display_cfg(cfg: MyConfig | DictConfig) -> None:
    if isinstance(cfg, MyConfig):
        pprint(asdict(cfg))
    elif isinstance(cfg, DictConfig):
        print(OmegaConf.to_yaml(cfg))


class ForExtractFolder(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        extensions: tuple[str, ...] | None = IMG_EXTENSIONS,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(
            self.root, class_to_idx, extensions, is_valid_file
        )

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: dict[str, int],
        extensions: tuple[str, ...] | None = None,
        is_valid_file: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        else:
            made_dataset = make_dataset(
                directory, class_to_idx, extensions, is_valid_file
            )
            return cast(list[tuple[str, int]], made_dataset)

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        found_classes = find_classes(directory)
        return cast(tuple[list[str], dict[str, int]], found_classes)

    def __getitem__(self, index: int) -> tuple[Any, Any, str, str]:
        (
            path,
            target,
        ) = self.samples[index]

        fname = path.split("/")[-1]
        dirname = path.split("/")[-2]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, dirname, fname

    def __len__(self) -> int:
        return len(self.samples)


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.Inf
        self.force_cancel: bool = False

    def __call__(
        self,
        val_loss: float | Any,
        model: Model,
        save_path: Optional[str | Path],
    ) -> None:
        score: float = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float | Any,
        model: Model,
        save_path: Optional[str | Path],
    ) -> None:
        if self.verbose:
            print(
                "Validation loss decreased",
                f"({self.val_loss_min:.3f} --> {val_loss:.3f}). \n",
                "Saving model ...",
            )
        save_path = (
            "./model/model_parameters_EA.pth"
            if save_path is None
            else save_path
        )
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    model = nn.Sequential(nn.BatchNorm2d(3))
    model.apply(weight_init)
