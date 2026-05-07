import argparse
from pathlib import Path

from app import VideoFrameSlicerApp
from config import SlicerConfig


Size = tuple[int, int]


def parse_size(values: list[str] | None) -> Size | None:
    if values is None:
        return None

    if len(values) != 2:
        raise argparse.ArgumentTypeError("target size must contain width and height")

    width, height = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("target size values must be positive")

    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open UI for slicing frames from video files.")
    parser.add_argument(
        "--video-folder",
        default=r"D:\Matller\ProductionCounters\yolo\video",
        help="Folder with input videos.",
    )
    parser.add_argument(
        "--output-folder",
        default=r"D:\Matller\ProductionCounters\yolo\dataset",
        help="Folder for saved frames.",
    )
    parser.add_argument(
        "--target-size",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional resize before saving/ROI. Omit to keep original frame size.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Start with auto-save disabled.",
    )
    parser.add_argument(
        "--auto-step",
        type=int,
        default=1,
        help="Save every N-th frame when auto-save is enabled.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Start with ROI enabled. Use the Select ROI button to choose it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = VideoFrameSlicerApp(
        SlicerConfig(
            video_folder=Path(args.video_folder),
            output_folder=Path(args.output_folder),
            target_size=parse_size(args.target_size),
            auto_enabled=not args.manual,
            auto_step=max(args.auto_step, 1),
            use_roi=args.roi,
        )
    )
    app.run()


if __name__ == "__main__":
    main()
