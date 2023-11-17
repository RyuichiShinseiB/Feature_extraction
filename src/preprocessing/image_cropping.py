# Standard Library
import os
from pathlib import Path

# Third Party Library
from PIL import Image


def crop_image(
    image_path: Path | str, crop_size: tuple[int, int], stride: int
) -> tuple[list[Image.Image], list[str]]:
    """Image cropping specifying crop size and stride

    Parameters
    ----------
    image_path : Path | str
        Path to image
    crop_size : tuple[int, int]
        cropping size to specify.
        (cropped_width, cropped_height)
    stride : int
        Width and height sliding range

    Returns
    -------
    cropped_images : list[Image.Image]
        Image cropped from a single image.
    position : list[str]
        Position of cropping
    """
    image = Image.open(image_path)
    width, height = image.size

    cropped_images: list[Image.Image] = []
    positions: list[str] = []

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            right = left + crop_size[0]
            bottom = top + crop_size[1]

            if right > width or bottom > height:
                right = width
                bottom = height

                left = right - crop_size[0]
                top = bottom - crop_size[0]

            cropped_image = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
            position = f"h{top:03}_w{left:04}"
            positions.append(position)

    return cropped_images, positions


crop_size = (128, 128)
stride = 64
original_image_dir = Path(
    "/home/shinsei/MyResearchs/feat_extrc/"
    "data/interim/CNTForest"
    "/exp1-9_dataset_1280x960"
)
save_image_dir = Path(
    "/home/shinsei/MyResearchs/feat_extrc/"
    f"data/processed/CNTForest/cnt_sem_{crop_size[0]}x{crop_size[1]}"
)

if not original_image_dir.parent.exists():
    raise FileNotFoundError(f"Can not found File: {original_image_dir.parent}")
if not save_image_dir.parent.exists():
    raise FileNotFoundError(f"Can not found File: {save_image_dir.parent}")

for mag in original_image_dir.iterdir():
    print(mag.name)
    for sample in mag.iterdir():
        print(sample.name)
        for image_path in sample.iterdir():
            cropped_images, positions = crop_image(
                image_path, crop_size, stride
            )
            for cropped_image, position in zip(cropped_images, positions):
                save_dir = save_image_dir / mag.name / sample.name
                if not save_dir.exists():
                    os.makedirs(save_dir)
                    print("Made Path: ", save_dir)
                save_name = f"{mag.name}_{sample.name}_{position}.png"
                cropped_image.save(save_dir / save_name)
