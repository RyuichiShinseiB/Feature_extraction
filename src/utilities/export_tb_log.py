import argparse
import io
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import tqdm
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import (
    AudioEvent,
    EventAccumulator,
    HistogramEvent,
    ImageEvent,
    ScalarEvent,
    TensorEvent,
)


class TensorboardLogs:
    def __init__(self, ea: EventAccumulator, src: Path) -> None:
        self.src = src
        self.tags = ea.Tags()
        self.image_tags: list[str] = self.tags["images"]
        self.audio_tags: list[str] = self.tags["audio"]
        self.histogram_tags: list[str] = self.tags["histograms"]
        self.scalar_tags: list[str] = self.tags["scalars"]
        self.tensor_tags: list[str] = self.tags["tensors"]

        self.image_logs: dict[str, list[ImageEvent]] = {
            t: ea.Images(t) for t in self.tags["images"]
        }
        self.audio_logs: dict[str, list[AudioEvent]] = {
            t: ea.Audio(t) for t in self.tags["audio"]
        }
        self.histogram_logs: dict[str, list[HistogramEvent]] = {
            t: ea.Histograms(t) for t in self.tags["histograms"]
        }
        self.scalar_logs: dict[str, list[ScalarEvent]] = {
            t: ea.Scalars(t) for t in self.tags["scalars"]
        }
        self.tensor_logs: dict[str, list[TensorEvent]] = {
            t: ea.Tensors(t) for t in self.tags["tensors"]
        }

    def get_image_events_as_images(
        self,
    ) -> dict[str, list[tuple[int, Image.Image]]]:
        return {
            t: [
                (ie.step, encode_image_from(ie.encoded_image_string))
                for ie in ies
            ]
            for t, ies in self.image_logs.items()
        }

    def get_scalar_events_as_scalars(
        self,
    ) -> dict[str, list[tuple[int, float]]]:
        return {
            t: [(se.step, se.value) for se in ses]
            for t, ses in self.scalar_logs.items()
        }

    @classmethod
    def load_event_accumulator(cls, logdir: str | Path) -> "TensorboardLogs":
        ea = EventAccumulator(
            str(logdir),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.TENSORS: 0,
            },
        ).Reload()
        return cls(ea, Path(logdir))


def encode_image_from(buf: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(buf))
        img.load()
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e


def _mkdir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tensorboard logs.")
    parser.add_argument(
        "src", help="Path to the sensorboard log file or directory", type=Path
    )
    parser.add_argument(
        "dst", help="Path to log output destination", type=Path
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search logs in directories.",
    )
    parser.add_argument(
        "-t",
        "--target_subdirs",
        nargs="*",
        help="Subdirectories to target when `src` is a directory.",
    )
    parser.add_argument(
        "--dt",
        "--datatype",
        choices=["scalars", "images", "both"],
        default="scalars",
        help="Specify the datatype: 'scalars', 'images', or 'both'.",
    )
    return parser.parse_args()


def get_src_dirs(
    src_dir: Path, target_subdirs: list[str], recursive: bool
) -> list[Path]:
    if not recursive and target_subdirs:
        raise ValueError(
            "The '-t' option must be used "
            "in conjunction with '-r' for recursive search."
        )

    src_dirs = [src_dir]
    if recursive:
        src_dirs = (
            [src_dir / subdir for subdir in target_subdirs]
            if target_subdirs
            else [src_dir]
        )
        src_dirs = list(
            itertools.chain.from_iterable(
                [list(dir.glob("**/run-tb")) for dir in src_dirs]
            )
        )
    return src_dirs


def export_scalars(log: TensorboardLogs, dst_parent: Path) -> None:
    tag_lines = {
        t: [f"{s},{v}\n" for s, v in events]
        for t, events in log.get_scalar_events_as_scalars().items()
    }
    with ThreadPoolExecutor() as executor:
        for tag, lines in tag_lines.items():
            tag = tag.replace("/", "_")
            file_name = f"run-{log.src.parent.name}_run-tb-tag-{tag}.csv"
            executor.submit(
                write_to_file, path=dst_parent / file_name, lines=lines
            )


def write_to_file(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Step,Value\n")
        f.writelines(lines)


def export_images(log: TensorboardLogs, dst_parent: Path) -> None:
    figures_dir = dst_parent / "figures"
    _mkdir(figures_dir)
    with ThreadPoolExecutor() as executor:
        for tag, steps_images in log.get_image_events_as_images().items():
            tag = tag.replace(" ", "_")
            for step, image in steps_images:
                executor.submit(
                    image.save, figures_dir / f"{tag}_step{step}.png"
                )


def run() -> None:
    args = parse_args()
    src_dirs = get_src_dirs(
        Path(args.src), args.target_subdirs, args.recursive
    )

    print("Found the following logs:")
    for src in src_dirs:
        print(f" - {src}")

    print("Loading source logs...")
    with ProcessPoolExecutor() as executor:
        logs = list(
            executor.map(TensorboardLogs.load_event_accumulator, src_dirs)
        )
    print("Completed to load source logs!")

    for log in tqdm.tqdm(logs, desc="Exporting logs"):
        run_time = log.src.parent
        run_date = run_time.parent
        dst_parent = Path(args.dst) / run_date.name / run_time.name
        _mkdir(dst_parent)

        if args.dt in {"scalars", "both"}:
            export_scalars(log, dst_parent)

        if args.dt in {"images", "both"}:
            export_images(log, dst_parent)

        if args.dt not in {"scalars", "images", "both"}:
            raise ValueError(
                f"A unexpected datatype option was entered: {args.datatype}\n",
                "You can select 'scalars', images' or 'both'.",
            )
