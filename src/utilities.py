# Standard Library
from typing import Any, Callable, Literal, Optional, TypeAlias

# Third Party Library
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# First Party Library
from src import Transforms, TransformsName, TransformsNameValue

Model: TypeAlias = nn.Module
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = Literal["cpu", "cuda"] | torch.device


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
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.kaiming_normal_(m.weight)


def get_dataloader(
    dataset_path: str,
    dataset_transform: TransformsNameValue,
    batch_size: int = 64,
    shuffle: bool = True,
    split_ratio: tuple[float, float] | None = None,
    generator_seed: int | None = 42,
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
        _description_
    """
    transform = transforms.Compose(
        [
            str2transform(name, value)
            for name, value in dataset_transform.items()
        ]
    )

    dataset = ImageFolder(dataset_path, transform)

    if split_ratio is not None:
        train_dataset, val_dataset = random_split(
            dataset,
            split_ratio,
            generator=torch.Generator().manual_seed(generator_seed)
            if generator_seed is not None
            else None,
        )
        return (
            DataLoader(train_dataset, batch_size, shuffle, num_workers=2),
            DataLoader(val_dataset, batch_size, shuffle, num_workers=2),
        )
    else:
        return DataLoader(dataset, batch_size, shuffle, num_workers=2)


def str2transform(
    transforms_name: TransformsName,
    transforms_param: Any,
) -> Transforms:
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
        self, val_loss: float | Any, model: Model, save_path: Optional[str]
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
        self, val_loss: float | Any, model: Model, save_path: Optional[str]
    ) -> None:
        if self.verbose:
            print(
                "Validation loss decreased",
                f"({self.val_loss_min:.3f} --> {val_loss:.3f}). \n",
                "Saving model ...",
            )
        save_path = "./model/model.pth" if save_path is None else save_path
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
