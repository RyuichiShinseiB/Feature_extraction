# Standard Library
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    TypeAlias,
    TypeGuard,
    cast,
    overload,
)

# Third Party Library
import numpy as np
import torch
import torch.onnx as onnx
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import make_dataset, pil_loader
from torchvision.transforms import transforms
from tqdm import tqdm

# Local Library
from ..mytyping import (
    Device,
    Model,
    Tensor,
    Transforms,
    TransformsName,
    TransformsNameValue,
)

# from ..predefined_models import model_define

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
FeatureArray: TypeAlias = np.ndarray
DirList: TypeAlias = list[str]
FileNameList: TypeAlias = list[str]


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


@overload
def get_dataloader(
    dataset_path: Path | str,
    dataset_transform: TransformsNameValue,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    generator_seed: int | None = 42,
    extraction: bool = False,
    cls_conditions: dict[int, list[str]] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
) -> DataLoader: ...


@overload
def get_dataloader(
    dataset_path: Path | str,
    dataset_transform: TransformsNameValue,
    split_ratio: tuple[float, float],
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    generator_seed: int | None = 42,
    extraction: bool = False,
    cls_conditions: dict[int, list[str]] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
) -> tuple[DataLoader, DataLoader]: ...


@overload
def get_dataloader(
    dataset_path: Path | str,
    dataset_transform: TransformsNameValue,
    split_ratio: tuple[float, float] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    generator_seed: int | None = 42,
    extraction: bool = False,
    cls_conditions: dict[int, list[str]] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
) -> tuple[DataLoader, DataLoader] | DataLoader: ...


def get_dataloader(
    dataset_path: Path | str,
    dataset_transform: TransformsNameValue,
    split_ratio: tuple[float, float] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    generator_seed: int | None = 42,
    extraction: bool = False,
    cls_conditions: dict[int, list[str]] | None = None,
    is_valid_file: Callable[[str], bool] | None = None,
) -> tuple[DataLoader, DataLoader] | DataLoader:
    """specifying parameters and output dataloader

    Parameters
    ----------
    dataset_path : str
        Path to dataset
    dataset_transform : TransformsNameValue
        dictionary type parameters. {"transform": parameter}
    split_ratio : tuple[float, float] | None, optional
        Split training and validation if not None, by default None
    batch_size : int, optional
        Unit of data used per study, by default 64
    shuffle : bool
        The dataloader is shuffled if True, by default True
    extraction : bool
        Whether to use for extraction, False by default, by default False
    cls_conditions: dict[int, list[str]] | None, optional
        Works as a substitute for `target_transform`. The key is the new class and the value is the list of source classes corresponding to that class.

    Returns
    -------
    DataLoader
    """  # noqa: E501
    transform = transforms.Compose(
        [
            str2transform(name, value)
            for name, value in dataset_transform.items()
        ]
    )

    if extraction:
        dataset = ForExtractFolder(
            dataset_path,
            transform=transform,
            cls_conditions=cls_conditions,
            is_valid_file=is_valid_file,
        )
    else:
        dataset = MyImageFolder(
            dataset_path,
            transform=transform,
            cls_conditions=cls_conditions,
            is_valid_file=is_valid_file,
        )

    if split_ratio is None:
        return DataLoader(dataset, batch_size, shuffle, num_workers=2)
    else:
        splitted_dataset: list[Subset[ForExtractFolder | ImageFolder]] = (
            random_split(
            dataset,
            split_ratio,
            generator=torch.Generator().manual_seed(generator_seed)
            if generator_seed is not None
            else None,
            )
        )
        return (
            DataLoader(
                splitted_dataset[0], batch_size, shuffle, num_workers=2
            ),
            DataLoader(
                splitted_dataset[1], batch_size, shuffle, num_workers=2
            ),
        )


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


def save_onnx(
    model: Model, input_size: tuple[int, ...], path: str | Path | BytesIO
) -> None:
    model.eval()

    dummy_shape = [1] + list(input_size)
    dummy_input = torch.randn(dummy_shape, requires_grad=True)

    onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
    )


def extract_features(
    model: Model, dataloader: DataLoader["ForExtractFolder"], device: Device
) -> tuple[FeatureArray, DirList, FileNameList]:
    features_list: list[Tensor] = []
    dirnames_list: list[str] = []
    filenames_list: list[str] = []
    model.eval()
    with torch.no_grad():
        for x, *_, dirnames, filenames in tqdm(dataloader):
            _, features = model(x.to(device))
            if isinstance(features, tuple):
                if len(features) == 2:
                    features = features[0]
                elif len(features) == 3:
                    features = features[1]
            features_list.extend(
                torch.flatten(features, start_dim=1).detach().cpu().tolist()
            )
            dirnames_list.extend(dirnames)
            filenames_list.extend(filenames)
    features_array = np.array(features_list)

    return features_array, dirnames_list, filenames_list


def is_tuple_of_pairs(
    value: tuple[tuple[int, int] | int, ...],
) -> TypeGuard[tuple[tuple[int, int], ...]]:
    if not isinstance(value, tuple):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(i, int) for i in item)
        for item in value
    )


def is_tuple_of_ints(
    value: tuple[tuple[int, int] | int, ...],
) -> TypeGuard[tuple[int, ...]]:
    if not isinstance(value, tuple):
        return False

    return all(isinstance(item, int) for item in value)


def find_project_root(
    start_path: Path = Path.cwd(),
    marker: str = "pyproject.toml",
    relative: bool = False,
) -> Path:
    current_path = start_path
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            if relative:
                relative_start = start_path.relative_to(current_path)
                num_up_lvl = len(relative_start.parts)
                root = Path("../" * num_up_lvl or "./")
                return root

            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"{marker} not found in any parent directories.")


class BaseMyDataset(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        extensions: tuple[str, ...] | None = IMG_EXTENSIONS,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        cls_conditions: dict[int, list[str]] | None = None,
    ) -> None:
        super().__init__(root, transform=transform)

        self.is_using_cls = self._is_using_cls(cls_conditions)

        classes, class_to_idx = self.find_classes(self.root)

        if is_valid_file is not None:
            extensions = None
        samples = self.make_dataset(
            self.root, class_to_idx, extensions, is_valid_file
        )

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.target_transform = target_transform or self._target_transform(
            cls_conditions
        )

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

    def find_classes(
        self, root: str | Path
    ) -> tuple[list[str], dict[str, int]]:
        if isinstance(root, str):
            root = Path(root)

        dir_names = sorted(p.name for p in root.iterdir() if p.is_dir())
        cls_to_idx = {
            c: i for i, c in enumerate(dir_names) if self.is_using_cls(c)
        }
        classes = list(cls_to_idx.keys())

        return classes, cls_to_idx

    @staticmethod
    def _is_using_cls(
        conditions: dict[int, list[str]] | None = None,
    ) -> Callable[[str], bool]:
        if conditions is None:
            return lambda _: True
        valid_classes = set(chain.from_iterable(conditions.values()))

        return lambda c: c in valid_classes

    @staticmethod
    def _target_transform(
        conditions: dict[int, list[str]] | None = None,
    ) -> Callable[[int], int]:
        if conditions is None:
            return lambda c: c

        def _tt(c: int) -> int:
            for cls, class_names in conditions.items():
                if str(c) in class_names:
                    return cls
            raise ValueError

        return _tt

    def __len__(self) -> int:
        return len(self.samples)


class MyImageFolder(BaseMyDataset):
    def __init__(
        self,
        root: str | Path,
        extensions: tuple[str, ...] | None = IMG_EXTENSIONS,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        cls_conditions: dict[int, list[str]] | None = None,
    ) -> None:
        super().__init__(
            root,
            extensions,
            transform,
            target_transform,
            loader,
            is_valid_file,
            cls_conditions,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        path, base_target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is None:
            target = base_target
        else:
            target = self.target_transform(base_target)

        return sample, target, base_target


class ForExtractFolder(BaseMyDataset):
    def __init__(
        self,
        root: str | Path,
        extensions: tuple[str, ...] | None = IMG_EXTENSIONS,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Callable[[str], bool] | None = None,
        cls_conditions: dict[int, list[str]] | None = None,
    ) -> None:
        super().__init__(
            root,
            extensions,
            transform,
            target_transform,
            loader,
            is_valid_file,
            cls_conditions,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int, int, str, str]:
        """Get data from dataset

        Parameters
        ----------
        index : int
            index

        Returns
        -------
        sample: Tensor
            Tensor data of image
        target: int
            Target data of `sample`.
        base_target: int
            `target` before `target_transform`.
        dirname: str
            Name of the directory containing the `sample` data. Target name
            in most cases.
        fname: str
            The file name of the `sample`.
        """
        path, base_target = self.samples[index]

        fname = path.split("/")[-1]
        dirname = path.split("/")[-2]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is None:
            target = base_target
        else:
            target = self.target_transform(base_target)

        return sample, target, base_target, dirname, fname


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        save_path: Path = Path("./model/temp/temp.pth"),
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter: int = 0
        self.best_score: float = -float("inf")
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.force_cancel: bool = False
        self.save_path = save_path

    def __call__(
        self,
        val_loss: float | Any,
        model: Model,
        save_path: str | Path | None = None,
    ) -> None:
        score: float = -val_loss

        if score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if save_path is None:
                save_path = self.save_path
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float | Any,
        model: Model,
        save_path: str | Path = "model_parameters.pth",
    ) -> None:
        if self.verbose:
            print(
                "Validation loss decreased",
                f"({self.val_loss_min:.3f} --> {val_loss:.3f}). \n",
                "Saving model ...",
            )
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    model = nn.Sequential(nn.BatchNorm2d(3))
    model.apply(weight_init)
