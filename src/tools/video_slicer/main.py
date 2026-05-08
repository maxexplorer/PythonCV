import argparse
from pathlib import Path

from app import VideoFrameSlicerApp
from config import SlicerConfig
from settings import load_slicer_config


Size = tuple[int, int]
DEFAULT_OUTPUT_FOLDER = Path(__file__).resolve().parents[2] / "dataset"


def parse_size(values: list[str] | None) -> Size | None:
    if values is None:
        return None

    if len(values) != 2:
        raise argparse.ArgumentTypeError("target size must contain width and height")

    width, height = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("target size values must be positive")

    return width, height


def parse_args(default_config: SlicerConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open UI for slicing frames from video files.")
    parser.add_argument(
        "--video-folder",
        default=str(default_config.video_folder),
        help="Folder with input videos.",
    )
    parser.add_argument(
        "--output-folder",
        default=str(default_config.output_folder),
        help="Folder for saved frames.",
    )
    parser.add_argument(
        "--target-size",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(default_config.target_size) if default_config.target_size else None,
        help="Optional resize before saving/ROI. Omit to keep original frame size.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        default=not default_config.auto_enabled,
        help="Start with auto-save disabled.",
    )
    parser.add_argument(
        "--auto-step",
        type=int,
        default=default_config.auto_step,
        help="Save every N-th frame when auto-save is enabled.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        default=default_config.use_roi,
        help="Start with ROI enabled. Use the Select ROI button to choose it.",
    )
    return parser.parse_args()


def main() -> None:
    default_config = load_slicer_config(
        Path(r"D:\Matller\ProductionCounters\yolo\video"),
        DEFAULT_OUTPUT_FOLDER,
    )
    args = parse_args(default_config)
    app = VideoFrameSlicerApp(
        SlicerConfig(
            video_folder=Path(args.video_folder),
            output_folder=Path(args.output_folder),
            selected_video_files=default_config.selected_video_files,
            target_size=parse_size(args.target_size),
            scale_mode=default_config.scale_mode,
            auto_enabled=not args.manual,
            auto_step=max(args.auto_step, 1),
            use_roi=args.roi,
            roi=default_config.roi if args.roi else None,
        )
    )
    app.run()


if __name__ == "__main__":
    main()
