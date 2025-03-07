from collections.abc import Generator, Sequence
from pathlib import Path

import polars as pl
from PIL import Image


class PasteLocation:
    def __init__(
        self,
        imsize: tuple[int, int],
        tile_size: tuple[int, int],
        pad_size: tuple[int, int],
    ):
        """Output location for paste image likely tile.

        Parameters
        ----------
        imsize : tuple[int, int]
            Unit is pixel `(width, height)` .
        tile_size : tuple[int, int]
            The number of horizontal and vertical directions to tile \
            `(horizontal, vertical)`. \
            The unit is the number of sheets.
        pad_size : tuple[int, int]
            The size of the margins per image `(left-right, top-bottom)`. \
            The unit is pixels.
        """
        self.img_width = imsize[0]
        self.img_height = imsize[1]
        self.tile_size = tile_size
        self.num_tile = self.tile_size[0] * self.tile_size[1]
        self.pad_x = pad_size[0]
        self.pad_y = pad_size[1]

    def __getitem__(self, idx: int) -> tuple[int, int]:
        if idx < self.num_tile:
            nx = idx % self.tile_size[0]
            ny = idx // self.tile_size[0]
            x, y = self._calc_location(nx, ny)
            return x, y
        raise IndexError()

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        for i in range(self.num_tile):
            yield self.__getitem__(i)

    def _calc_location(self, nx: int, ny: int) -> tuple[int, int]:
        next_x = self.pad_x + (self.img_width + self.pad_x) * nx
        next_y = self.pad_y + (self.img_height + self.pad_y) * ny
        return next_x, next_y

    @property
    def max_size(self) -> tuple[int, int]:
        return self._calc_location(*self.tile_size)

    def concat_images(self, images: Sequence[Image.Image]) -> Image.Image:
        canvas = Image.new(images[0].mode, self.max_size)
        for img, loc in zip(images, self):
            canvas.paste(img, loc)

        return canvas


def sample_images(
    df: pl.DataFrame, img_dir: Path, num_sample: int
) -> dict[int, dict[int, list[Image.Image]]]:
    targets = df.select("target").unique().to_series()
    each_target_images: dict[int, dict[int, list[Image.Image]]] = {}
    for t in targets.sort():
        unique_dirname = (
            df.filter(pl.col("target") == t)
            .select(pl.col("dirname"))
            .unique("dirname")
            .to_series()
        )
        images: dict[int, list[Image.Image]] = {}
        for i in unique_dirname.sort():
            paths = (
                df.filter((pl.col("dirname") == i) & (pl.col("target") == t))
                .select(
                    pl.concat_str(
                        [pl.col("dirname").cast(pl.Utf8), pl.col("filename")],
                        separator="/",
                    )
                )
                .to_series()
            )
            paths = paths.sample(
                min(num_sample, len(paths)),
                with_replacement=False,
                shuffle=True,
            )
            images[i] = [Image.open(img_dir / p).convert("L") for p in paths]
        each_target_images[t] = images
    return each_target_images


def concat_each_target_images(
    sampled_imgs: dict[int, dict[int, list[Image.Image]]],
    pasteloc: PasteLocation,
) -> dict[int, dict[int, Image.Image]]:
    each_target_cat_imgs: dict[int, dict[int, Image.Image]] = {}
    for t, images in sampled_imgs.items():
        cat_imgs: dict[int, Image.Image] = {}
        for d, image in images.items():
            concated = pasteloc.concat_images(image)
            cat_imgs[d] = concated
        each_target_cat_imgs[t] = cat_imgs
    return each_target_cat_imgs


def save_cated_images(
    cated_imgs: dict[int, dict[int, Image.Image]],
    dst_dir: Path,
    *,
    fname_format: str | None = None,
    target_format: dict[int, str] | None = None,
    sample_format: dict[int, str] | None = None,
) -> None:
    for t, imgs in cated_imgs.items():
        for d, concated in imgs.items():
            if not dst_dir.exists():
                dst_dir.mkdir(parents=True)
            if fname_format is None:
                fname = f"label_{t}_sample_{d}.jpg"
            else:
                fname = format_fname(
                    fname_format,
                    t,
                    d,
                    target_format,
                    sample_format,
                )

            concated.save(dst_dir / fname)


def concat_images(
    df: pl.DataFrame,
    img_dir: Path,
    imsize: int,
    concat_tile_size: tuple[int, int] = (3, 3),
    *,
    filter_expr: pl.Expr | None = None,
) -> dict[int, dict[int, Image.Image]]:
    if filter_expr is None:
        flted_df = df
    else:
        flted_df = df.filter(filter_expr)

    num_sample = concat_tile_size[0] * concat_tile_size[1]

    sampled_imgs = sample_images(flted_df, img_dir, num_sample)

    pasteloc = PasteLocation(
        imsize=(imsize, imsize),
        tile_size=concat_tile_size,
        pad_size=(2, 2),
    )

    each_target_cat_imgs = concat_each_target_images(sampled_imgs, pasteloc)
    return each_target_cat_imgs


def format_fname(
    base: str,
    t: int,
    d: int,
    t_cvtr: dict[int, str] | None = None,
    d_cvtr: dict[int, str] | None = None,
) -> str:
    t_str = None
    if t_cvtr is not None:
        t_str = t_cvtr[t]
    d_str = None
    if d_cvtr is not None:
        d_str = d_cvtr[d]

    num_t_d_format = (base.count(r"{t}"), base.count(r"{d}"))
    if num_t_d_format == (0, 0):
        return base
    elif num_t_d_format == (0, 1):
        return base.format(d=d_str or d)
    elif num_t_d_format == (1, 0):
        return base.format(t=t_str or t)
    elif num_t_d_format == (1, 1):
        return base.format(t=t_str or t, d=d_str or d)
    else:
        raise ValueError(
            "Incomplete keyword specification.",
            f"num_t={num_t_d_format[0]}, " f"num_d={num_t_d_format[1]}",
        )
