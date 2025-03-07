# Standard Library
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

# Third Party Library
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.typing import ColorType, MarkerType
from PIL import Image

__UNIT_PER_METER: dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "inch": 1 / 0.0254,
}

NumArgCo = TypeVar(
    "NumArgCo", float, list[float], tuple[float, ...], covariant=True
)

IMG_EXT = {"pdf", "png", "svg"}


def set_mpl_styles(
    *,
    fontsize: int = 15,
    grid: bool = True,
    tick_direction: Literal["in", "out", "inout"] = "in",
    minor_ticks: bool = True,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.direction"] = tick_direction
    plt.rcParams["ytick.direction"] = tick_direction
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["xtick.minor.visible"] = minor_ticks
    plt.rcParams["ytick.minor.visible"] = minor_ticks


def cvt_unit(v: NumArgCo, old: str = "cm", new: str = "inch") -> NumArgCo:
    if isinstance(v, list):
        return [_v * __UNIT_PER_METER[new] * __UNIT_PER_METER[old] for _v in v]
    elif isinstance(v, tuple):
        return tuple(
            _v * __UNIT_PER_METER[new] * __UNIT_PER_METER[old] for _v in v
        )
    return v * __UNIT_PER_METER[new] * __UNIT_PER_METER[old]


def plot_cumulative_contribution_rate(
    rates: Sequence[float],
    ax: Axes | None = None,
    cumulated: bool = False,
    threshold: float = 0.7,
    show_threshold: bool = False,
) -> (
    tuple[Figure, Axes, list[Line2D], float] | tuple[Axes, list[Line2D], float]
):
    fig: Figure | None = None
    if ax is None:
        fig = plt.figure(figsize=(6.4, 4.4), layout="constrained", dpi=300)
        ax = fig.add_subplot()
    if not cumulated:
        rates = list(itertools.accumulate(rates))

    num_comp = len(rates)
    x = list(range(0, num_comp + 1))
    y = [0] + list(rates)

    lines = ax.plot(x, y, marker="o")

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative contribution rate")
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    ax.set_xlim(0, num_comp + 1)
    ax.set_ylim(bottom=0)

    max_index_thresh = np.argmin(
        [abs(rate - threshold) for rate in rates]
    ).astype(int)
    num_use_features = max_index_thresh + 1
    if show_threshold:
        _, xaxis_right = ax.get_xlim()
        ax.hlines(
            threshold, 0, xaxis_right, linestyles="--", colors="red", alpha=0.5
        )
        ax.vlines(
            num_use_features,
            0,
            rates[num_use_features - 1],
            colors="red",
        )
        ax.text(
            x=num_use_features,
            y=-0.02,
            s=f"{num_use_features}",
            color="red",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
        )

    if fig is None:
        return ax, lines, num_use_features
    else:
        return fig, ax, lines, num_use_features


def scatter_each_classes(
    data: np.ndarray,
    class_labels: np.ndarray,
    rank: np.ndarray,
    markers: Sequence[MarkerType],
    colors: Sequence[ColorType],
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
    col_row: tuple[int, int],
    image_root: str | Path,
    labels: list[int] | np.ndarray | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[list[Image.Image], pl.DataFrame]:
    if not isinstance(image_root, Path):
        image_root = Path(image_root)

    used_filepaths: list[pl.DataFrame] = []
    concat_imgs: list[Image.Image] = []
    if labels is None:
        labels = df.select("cluster").to_numpy().flatten()

    unique_label = np.unique(labels)

    for label in unique_label:
        hoge = df.filter(pl.col("cluster") == label)

        # 該当するクラスタのデータが少ない場合の処理
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
        imgs = [Image.open(image_root / p) for p in file_paths]
        print(f"{label}: {len(imgs)=}")
        concat_imgs.append(concat_images(imgs, col_row[0], col_row[1], 2))

    fig = plt.figure(figsize=figsize, layout="constrained")

    for i in unique_label:
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(concat_imgs[i], "gray")
        ax.set_title(f"cluster {i}")
        ax.axis("off")

    plt.show()

    return concat_imgs, pl.concat(used_filepaths)


@overload
def plot_scatter(
    data: Sequence[np.ndarray] | np.ndarray,
) -> tuple[Figure, Axes, PathCollection]:
    ...


@overload
def plot_scatter(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes | None = None,
) -> tuple[Axes, PathCollection]:
    ...


@overload
def plot_scatter(
    data: Sequence[np.ndarray] | np.ndarray,
    *,
    size: float = 20,
    color: ColorType | list[ColorType] = "tab:blue",
    marker: MarkerType = "o",
    ploting_axes: tuple[int, int] = (0, 1),
    # zorder: float = 0,
    scatter_label: str | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Figure, Axes, PathCollection]:
    ...


@overload
def plot_scatter(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes | None = None,
    *,
    size: float = 20,
    color: ColorType | list[ColorType] = "tab:blue",
    marker: MarkerType = "o",
    ploting_axes: tuple[int, int] = (0, 1),
    # zorder: float = 0,
    scatter_label: str | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Axes, PathCollection]:
    ...


def plot_scatter(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes | None = None,
    *,
    size: float = 20,
    color: ColorType | list[ColorType] = "tab:blue",
    marker: MarkerType = "o",
    ploting_axes: tuple[int, int] = (0, 1),
    # zorder: float = 0,
    scatter_label: str | int | Any | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Axes, PathCollection] | tuple[Figure, Axes, PathCollection]:
    fig = None
    if ax is None:
        fig = plt.figure(layout="constrained")
        ax = fig.add_subplot()
    pc = ax.scatter(
        data[ploting_axes[0]],
        data[ploting_axes[1]],
        s=size,
        marker=marker,
        color=color,
        label=scatter_label,
        # zorder=zorder,
        alpha=0.5,
    )
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    plot_range = np.max(np.abs((x_range, y_range)))
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal")
    if fig is None:
        return ax, pc
    else:
        return fig, ax, pc


@overload
def plot_bar(
    data: Sequence[np.ndarray] | np.ndarray,
) -> tuple[Figure, Axes, BarContainer]:
    ...


@overload
def plot_bar(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes,
) -> tuple[Axes, BarContainer]:
    ...


@overload
def plot_bar(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes,
    *,
    num_bins: int = 10,
    width: float = 0.7,
    bin_tick_label: Sequence[float] | np.ndarray | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Axes, BarContainer]:
    ...


@overload
def plot_bar(
    data: Sequence[np.ndarray] | np.ndarray,
    *,
    num_bins: int = 10,
    width: float = 0.7,
    bin_tick_label: Sequence[float] | np.ndarray | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Figure, Axes, BarContainer]:
    ...


def plot_bar(
    data: Sequence[np.ndarray] | np.ndarray,
    ax: Axes | None = None,
    *,
    num_bins: int = 10,
    width: float = 0.7,
    bin_tick_label: Sequence[float] | np.ndarray | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> tuple[Figure, Axes, BarContainer] | tuple[Axes, BarContainer]:
    fig = None
    if ax is None:
        fig = plt.figure(layout="constrained")
        ax = fig.add_subplot()
    hist, _ = np.histogram(data, np.arange(num_bins + 1))
    if bin_tick_label is None:
        bin_tick_label = np.arange(num_bins)
    bar_cont = ax.bar(bin_tick_label, hist, width, tick_label=bin_tick_label)
    if axis_labels is None:
        axis_labels = ("x", "y")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if fig is None:
        return ax, bar_cont
    else:
        return fig, ax, bar_cont


def align_xy_ticks_interval(ax: Axes) -> None:
    """Align the scale interval of the x-axis and y-axis to the larger one.

    Parameters
    ----------
    ax : Axes
        A target `Axes` instance

    Examples
    ---------
    >>> x = np.arange(0, 2 * np.pi, 0.01)
    >>> y = np.sin(x)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot()
    >>> ax.plot(x, y)
    >>> print(ax.get_xticks())
    [-1.  0.  1.  2.  3.  4.  5.  6.  7.]
    >>> print(ax.get_yticks())
    [-1.25 -1.   -0.75 -0.5  -0.25  0.    0.25  0.5   0.75  1.    1.25]
    >>> _align_xy_ticks_interval(ax)
    >>> print(ax.get_xticks())
    [-1.  0.  1.  2.  3.  4.  5.  6.  7.]
    >>> print(ax.get_yticks())
    [-2. -1.  0.  1.  2.]

    """
    xticks = ax.get_xticks()
    xinterval = xticks[1] - xticks[0]
    yticks = ax.get_yticks()
    yinterval = yticks[1] - yticks[0]
    interval = xinterval if xinterval > yinterval else yinterval

    ax.xaxis.set_major_locator(MultipleLocator(interval))
    ax.yaxis.set_major_locator(MultipleLocator(interval))


def set_xy_axes_range(ax: Axes | Sequence[Axes]) -> None:
    if isinstance(ax, Axes):
        xrange = ax.get_xlim()
        yrange = ax.get_ylim()
        max_val = np.max(np.abs((xrange, yrange)))

        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
    elif isinstance(ax, Sequence):
        xyranges = [(each_ax.get_xlim(), each_ax.get_ylim()) for each_ax in ax]
        max_val = np.max(np.abs(xyranges))
        _ = [
            (
                each_ax.set_xlim(-max_val, max_val),
                each_ax.set_ylim(-max_val, max_val),
            )
            for each_ax in ax
        ]


@overload
def plot_scatter_reflclsf_features(
    data: np.ndarray,
    metadata_df: pl.DataFrame,
    ax: Axes | None = None,
    plotting_dims: tuple[int, int] = (0, 1),
    cmap: Colormap | None = None,
    markers: dict[int, MarkerType] | None = None,
    is_pca_data: bool = True,
    is_rescaled_data: bool = False,
    showing_legend: bool = False,
) -> None:
    ...


@overload
def plot_scatter_reflclsf_features(
    data: np.ndarray,
    metadata_df: pl.DataFrame,
    figsize_cm: tuple[float, float] = (14, 8),
    plotting_dims: tuple[int, int] = (0, 1),
    cmap: Colormap | None = None,
    markers: dict[int, MarkerType] | None = None,
    is_pca_data: bool = True,
    is_rescaled_data: bool = False,
    showing_legend: bool = False,
) -> Figure:
    ...


@overload
def plot_scatter_reflclsf_features(
    data: np.ndarray,
    metadata_df: pl.DataFrame,
    ax: Axes | None = None,
    plotting_dims: tuple[int, int] = (0, 1),
    cmap: Colormap | None = None,
    markers: dict[int, MarkerType] | None = None,
    is_pca_data: bool = True,
    is_rescaled_data: bool = False,
    showing_legend: bool = True,
) -> list[Artist]:
    ...


@overload
def plot_scatter_reflclsf_features(
    data: np.ndarray,
    metadata_df: pl.DataFrame,
    figsize_cm: tuple[float, float] = (14, 8),
    plotting_dims: tuple[int, int] = (0, 1),
    cmap: Colormap | None = None,
    markers: dict[int, MarkerType] | None = None,
    is_pca_data: bool = True,
    is_rescaled_data: bool = False,
    showing_legend: bool = True,
) -> tuple[Figure, list[Artist]]:
    ...


def plot_scatter_reflclsf_features(
    data: np.ndarray,
    metadata_df: pl.DataFrame,
    ax: Axes | None = None,
    figsize_cm: tuple[float, float] = (14, 8),
    plotting_dims: tuple[int, int] = (0, 1),
    cmap: Colormap | None = None,
    markers: dict[int, MarkerType] | None = None,
    is_pca_data: bool = True,
    is_rescaled_data: bool = False,
    showing_legend: bool = True,
) -> Figure | list[Artist] | tuple[Figure, list[Artist]] | None:
    targets = metadata_df.select("target").unique()[:, 0]
    if is_pca_data and is_rescaled_data:
        label_prefix = "Rescaled PC"
    elif is_pca_data:
        label_prefix = "PC"
    elif is_rescaled_data:
        label_prefix = "Rescaled component "
    else:
        label_prefix = "Component "
    axis_labels = tuple(
        label_prefix + str(i + 1)
        for i in range(data.shape[1])
        if i in plotting_dims
    )
    target_label = {0: "Low", 1: "High", -1: "none"}

    if ax is None:
        figsize = cvt_unit(figsize_cm, old="cm", new="inch")
        fig = plt.figure(figsize=figsize, layout="constrained", dpi=100)  # type: ignore[arg-type]
        ax = fig.add_subplot()
    else:
        fig = None

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    if markers is None:
        markers = {0: "v", 1: "^", -1: "o"}

    for t in targets:
        target_df = metadata_df.filter(pl.col("target") == t)
        dirnames = target_df.select("dirname").to_numpy().flatten()
        colors = [cmap(d) for d in dirnames]
        idxes = target_df.select("id").to_numpy().flatten().tolist()
        plot_scatter(
            data[idxes].T,
            ax,
            color=colors,
            marker=markers[t],
            ploting_axes=plotting_dims,
            scatter_label=t,
        )

    legend_list: None | list[Artist] = None
    if showing_legend:
        # legend
        sample_legend_elems = [
            Patch(color=cmap(d), label=d, alpha=0.5)
            for d in metadata_df.select("dirname").unique()[:, 0]
        ]
        target_legend_elms = [
            Line2D(
                [0],
                [0],
                marker=markers[t],
                linestyle="",
                markeredgecolor="black",
                markerfacecolor="none",
                label=target_label[t],
            )
            for t in targets
        ]

        target_legend = ax.legend(
            handles=target_legend_elms,
            title="Reflectance",
            loc="lower left",
            bbox_to_anchor=(1.04, 0.0),
        )
        sample_id_legend = ax.legend(  # noqa: F841
            handles=sample_legend_elems,
            title="Sample Number",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        # 複数回`ax.legend()`をしても、最後に追加した凡例しか表示されない
        # ここで最初に追加した凡例を明示的に表示する
        ax.add_artist(target_legend)
        legend_list = [target_legend, sample_id_legend]

        align_xy_ticks_interval(ax)

    if fig is not None and legend_list is not None:
        return fig, legend_list
    elif fig is None and legend_list is not None:
        return legend_list
    elif fig is not None and legend_list is None:
        return fig
