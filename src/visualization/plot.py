# Standard Library
from pathlib import Path

# Third Party Library
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image


def scatter_each_classes(
    data: np.ndarray,
    class_labels: np.ndarray,
    rank: np.ndarray,
    markers: list[str],
    colors: list[str] | list[tuple[float, float, float, float]],
    xylabel: tuple[str, str],
    face_color: str = "valid",
    scatter_classes: list[int] | None = None,
    fontsize: int = 15,
    plot_range: tuple[float, float] | None = None,
    show_legend: bool = True,
    path: str | Path | None = None,
) -> None:
    labels: np.ndarray | list[int] = (
        np.unique(class_labels) if scatter_classes is None else scatter_classes
    )
    markers = (
        markers
        if markers is not None
        else ["o"] * np.unique(class_labels).shape[0]
    )
    colors = (
        [plt.get_cmap("tab10")(i) for i in range(len(np.unique(class_labels)))]
        if colors == "tab10"
        else colors
    )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)

    if isinstance(colors, matplotlib.colors.LinearSegmentedColormap):
        for label, r in zip(labels, rank):
            if (face_color == "None") and (markers[label] != "x"):
                ax.scatter(
                    data[class_labels == label, 0],
                    data[class_labels == label, 1],
                    edgecolors=colors[label],
                    label=f"cluster {label}",
                    marker=markers[label],
                    facecolor="None",
                    zorder=r,
                )
            else:
                ax.scatter(
                    data[class_labels == label, 0],
                    data[class_labels == label, 1],
                    color=colors[label],
                    label=f"cluster {label}",
                    marker=markers[label],
                    zorder=r,
                )
    else:
        for label, r in zip(labels, rank):
            if (face_color == "None") and (markers[label] != "x"):
                ax.scatter(
                    data[class_labels == label, 0],
                    data[class_labels == label, 1],
                    label=f"cluster {label}",
                    marker=markers[label],
                    edgecolors=colors[label],
                    facecolor="None",
                    zorder=r,
                )
            else:
                ax.scatter(
                    data[class_labels == label, 0],
                    data[class_labels == label, 1],
                    color=colors[label],
                    label=f"cluster {label}",
                    marker=markers[label],
                    zorder=r,
                )

    if plot_range is None:
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        max_abs_range = np.max(np.abs((y_range, x_range))) * 1.2
        print(f"{max_abs_range=}")

        ax.set_ylim(-max_abs_range, max_abs_range)
        ax.set_xlim(-max_abs_range, max_abs_range)
    else:
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
    ax.set_aspect("equal", "datalim")

    ax.set_box_aspect(1)
    ax.set_xlabel(xylabel[0], fontsize=fontsize)
    ax.set_ylabel(xylabel[1], fontsize=fontsize)
    if show_legend:
        ax.legend(fontsize=fontsize, loc="upper left", bbox_to_anchor=(1, 1))

    ax.grid(which="major")

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=fig.get_dpi(), bbox_inches="tight")


def concat_images(
    imgs: list[Image.Image], n_col: int, n_row: int, padding: int
) -> Image.Image:
    w, h = imgs[0].size
    w_with_pad = w + padding
    h_with_pad = h + padding
    width = (w_with_pad) * n_col + padding
    height = (h_with_pad) * n_row + padding

    dst = Image.new("L", (width, height))
    iter_imgs = iter(imgs)
    for j in range(n_row):
        for i in range(n_col):
            try:
                img = next(iter_imgs)
            except Exception:
                img = Image.new("L", img.size, color=255)

            dst.paste(
                img, (padding + w_with_pad * i, padding + h_with_pad * j)
            )
    return dst


def image_concat_and_imshow(
    df: pl.DataFrame,
    labels: list[int] | np.ndarray,
    col_row: tuple[int, int],
    image_root: str | Path,
    figsize: tuple[float, float] | None = None,
) -> tuple[list[Image.Image], pl.DataFrame]:
    if not isinstance(image_root, Path):
        image_root = Path(image_root)

    used_filepaths: list[pl.DataFrame] = []
    concat_imgs: list[Image.Image] = []
    num_labels = len(np.unique(labels))

    for label in np.unique(labels):
        imgs: list[Image.Image] = []
        hoge = df.filter(pl.col("cluster") == label)

        if len(hoge) > col_row[0] * col_row[1]:
            data_using_img_load = hoge.sample(col_row[0] * col_row[1])
        else:
            data_using_img_load = hoge
        # data_using_img_load = df.filter(pl.col("cluster") == label).sample(
        #     col_row[0] * col_row[1]
        # )
        used_filepaths.append(
            data_using_img_load.select(["filepath", "cluster"])
        )
        file_paths = (
            data_using_img_load.select("filepath").to_numpy().flatten()
        )
        for p in file_paths:
            imgs.append(Image.open(image_root / p))
        concat_imgs.append(concat_images(imgs, col_row[0], col_row[1], 2))

    fig = plt.figure(figsize=None if figsize is None else figsize)

    for i in range(num_labels):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(concat_imgs[i], "gray")
        ax.set_title(f"cluster {i}")
        ax.axis("off")

    plt.show()

    return concat_imgs, pl.concat(used_filepaths)
