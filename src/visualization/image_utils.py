from collections.abc import Generator, Sequence

from PIL import Image


class PasteLocation:
    def __init__(
        self,
        imsize: tuple[int, int],
        tile_size: tuple[int, int],
        pad_size: tuple[int, int],
    ):
        self.imsize_x = imsize[0]
        self.imsize_y = imsize[1]
        self.tile_size = tile_size
        self.num_tile = self.tile_size[0] * self.tile_size[1]
        self.pad_x = pad_size[0]
        self.pad_y = pad_size[1]

    def __getitem__(self, idx: int) -> tuple[int, int]:
        if idx < self.num_tile:
            nx = idx % self.tile_size[0]
            ny = idx // self.tile_size[1]
            x, y = self._calc_location(nx, ny)
            return x, y
        raise StopIteration()

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        for i in range(self.num_tile):
            yield self.__getitem__(i)

    def _calc_location(self, nx: int, ny: int) -> tuple[int, int]:
        next_x = self.pad_x + (self.imsize_x + self.pad_x) * nx
        next_y = self.pad_y + (self.imsize_y + self.pad_y) * ny
        return next_x, next_y

    @property
    def max_size(self) -> tuple[int, int]:
        return self._calc_location(*self.tile_size)

    def concat_images(self, images: Sequence[Image.Image]) -> Image.Image:
        canvas = Image.new(images[0].mode, self.max_size)
        for img, loc in zip(images, self):
            canvas.paste(img, loc)

        return canvas
