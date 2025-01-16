import numpy as np
import polars as pl
import torch
import tqdm
from torch.utils.data import DataLoader

from ..mytyping import Device, Tensor
from ._utils import ForExtractFolder


@torch.no_grad()
def _extract_features(
    model: torch.nn.Sequential,
    dataloader: DataLoader[ForExtractFolder],
    device: Device,
) -> tuple[np.ndarray, list[int], list[int], list[str], list[str]]:
    """Extract features from two-stage model such as autoencoder

    Parameters
    ----------
    model : torch.nn.Sequential
        two-stage model created by nn.Sequential(first, second)
    dataloader : DataLoader[ForExtractFolder]
        Dataloader returning (\
            `image`: Tensor,\
            `transformed_target`: Tensor,\
            `original_target` (converted from `original_target_name`): Tensor,\
            `original_target_name`: list[str],\
            `image_name`: list[str]\
        )
    device : Device
        Devices on which the model rides.

    Returns
    -------
    tuple[np.ndarray, list[int], list[int], list[str], list[str]]
        (\
            `features`, `transformed_targets`, `predictations`,\
            `directory_name`, `iput_image_name`\
        )\
        For `predictations`, if the model is an autoencoder, \
        the value will be a [-1, ..., -1]
    """
    features_list: list[Tensor] = []
    pred_list: list[int] = []
    target_list: list[int] = []
    dirnames_list: list[str] = []
    filenames_list: list[str] = []

    model.eval()
    for x, target, _, dirnames, filenames in tqdm.tqdm(dataloader):
        features = model[0](x.to(device))
        y: Tensor = model[1](features)
        if isinstance(features, tuple):
            if len(features) == 2:
                features = features[0]
            elif len(features) == 3:
                features = features[1]
        features_list.append(
            torch.flatten(features, start_dim=1).detach().cpu()
        )
        # If `model[1]` is a classifier.
        if y.dim() == 2:
            if y.shape[1] == 1:
                pred = (
                    torch.where(y > 0.5, 1, 0)
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )
            else:
                pred = torch.argmax(y).detach().cpu().flatten().tolist()
        else:
            pred = [-1 for _ in range(len(y))]

        target_list.extend(target)
        pred_list.extend(pred)
        dirnames_list.extend(dirnames)
        filenames_list.extend(filenames)
    features_array = torch.concat(features_list, dim=0).numpy()

    return (
        features_array,
        target_list,
        pred_list,
        dirnames_list,
        filenames_list,
    )


def _sort_out_extracted_data(
    features: np.ndarray,
    targets: list[int],
    preds: list[int],
    dirnames: list[str],
    filenames: list[str],
) -> pl.DataFrame:
    df = pl.DataFrame(features).with_columns(
        [
            pl.lit(pl.Series("target", targets, dtype=pl.Int32)),
            pl.lit(pl.Series("prediction", preds, dtype=pl.Int32)),
            pl.lit(pl.Series("dirname", dirnames, dtype=pl.Utf8)),
            pl.lit(pl.Series("filename", filenames, dtype=pl.Utf8)),
        ]
    )
    return df


def get_feature_table(
    model: torch.nn.Sequential,
    dataloader: DataLoader[ForExtractFolder],
    device: Device,
) -> pl.DataFrame:
    """Get feature table from `model`

    Parameters
    ----------
    model : torch.nn.Sequential
        two-stage model created by nn.Sequential(first, second)
    dataloader : DataLoader[ForExtractFolder]
        Dataloader returning (\
            `image`: Tensor,\
            `transformed_target`: Tensor,\
            `original_target` (converted from `original_target_name`): Tensor,\
            `original_target_name`: list[str],\
            `image_name`: list[str]\
        )
    device : Device
        Devices on which the model rides.

    Returns
    -------
    tuple[np.ndarray, list[int], list[int], list[str], list[str]]
        (\
            `features`, `transformed_targets`, `predictations`,\
            `directory_name`, `iput_image_name`\
        )\
        For `predictations`, if the model is an autoencoder, \
        the value will be a [-1, ..., -1]

    Returns
    -------
    pl.DataFrame
        columns are \
            column_0, column_1, ..., target, prediction, dirname and filename\
    """
    features, targets, preds, dirnames, filenames = _extract_features(
        model, dataloader, device
    )
    df = _sort_out_extracted_data(
        features, targets, preds, dirnames, filenames
    ).sort(pl.col("filename"))
    return df
