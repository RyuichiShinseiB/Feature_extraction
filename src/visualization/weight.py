from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from matplotlib.figure import Figure
from torch import nn

from src.mytyping import Model

TiledParamsDict: TypeAlias = dict[str, dict[str, torch.Tensor]]


def visualize_cnn_kernels(tiled_parameters: TiledParamsDict) -> Figure:
    fig = plt.figure(figsize=(12, 9))

    num_row = len(tiled_parameters)
    num_col = len(tiled_parameters["encoder"])
    for i, (k1, v1) in enumerate(tiled_parameters.items()):
        for j, (k2, v2) in enumerate(v1.items()):
            axes_idx = (j + 1) + i * num_col
            ax = fig.add_subplot(num_row, num_col, axes_idx)
            ax.imshow(v2[0], cmap="gray")
            ax.set_title(f"{k2} at {k1}")
            ax.axis("off")

    return fig


def make_tiled_kernel_dict(model: Model, tile_col: int = 5) -> TiledParamsDict:
    params: TiledParamsDict = {}
    for ed_name, ed_module in model.named_children():
        params[ed_name] = {}
        for se_layer_name, se_layer in ed_module.named_children():
            for inner_layer in se_layer.children():
                if isinstance(inner_layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    collect_kernels = inner_layer.weight.data.view(
                        -1, 1, 4, 4
                    )[: tile_col**2]
                    params[ed_name][se_layer_name] = vutils.make_grid(
                        collect_kernels, tile_col, normalize=True, padding=1
                    )
    return params


def save_tiled_kernel_images(
    tiled_parameters: TiledParamsDict, dst_dir: Path
) -> None:
    for k1, layer_kernels in tiled_parameters.items():
        for k2, tiled_kernel in layer_kernels.items():
            dst_path = dst_dir / k1
            if not dst_path.exists():
                dst_path.mkdir(parents=True)
            vutils.save_image(tiled_kernel, dst_path / f"{k2}_kernels.png")
