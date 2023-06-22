# Standard Library
from typing import Any, Callable, Literal, Optional, TypeAlias

# Third Party Library
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

Model: TypeAlias = nn.Module
Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = Literal["cpu", "cuda"] | torch.device


def weight_init(m: Any) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.kaiming_normal_(m.weight)


def get_dataloader(
    dataset_path: str,
    batch_size: int,
    running_mode: Literal["train", "pred"],
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(dataset_path, transform)
    return DataLoader(
        dataset,
        batch_size,
        shuffle=True if running_mode == "train" else False,
        num_workers=2,
    )


def train_step(
    input_data: Tensor,
    model: Model,
    loss_func: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    device: Device,
) -> tuple[Tensor, Tensor]:
    input_data = input_data.to(device)

    x_pred = model(input_data)
    loss, x_pred = loss_func(x_pred, input_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, x_pred


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
