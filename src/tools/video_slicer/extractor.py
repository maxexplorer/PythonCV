from pathlib import Path

import cv2

from config import Roi, SlicerConfig


class FrameProcessingError(RuntimeError):
    pass


class VideoFrameExtractor:
    SUPPORTED_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".m4v"}

    def __init__(self, config: SlicerConfig) -> None:
        self.config = config
        self.roi: Roi | None = None
        self.config.output_folder.mkdir(parents=True, exist_ok=True)

    def list_videos(self) -> list[Path]:
        if self.config.selected_video_files:
            return [
                path for path in self.config.selected_video_files
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

        if not self.config.video_folder.exists():
            return []

        return sorted(
            path for path in self.config.video_folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def resize_if_needed(self, frame):
        target_size = self.config.target_size
        if target_size is None:
            return frame
        if frame.shape[1] == target_size[0] and frame.shape[0] == target_size[1]:
            return frame
        return cv2.resize(frame, target_size)

    def prepare_frame(self, frame):
        frame = self.resize_if_needed(frame)

        if self.roi is None:
            return frame

        x, y, w, h = self.roi
        cropped = frame[y:y + h, x:x + w]
        if cropped.size == 0:
            raise FrameProcessingError(
                f"ROI produced an empty frame: x={x}, y={y}, w={w}, h={h}"
            )
        return cropped

    def select_roi_from_frame(self, frame) -> Roi | None:
        preview = self.resize_if_needed(frame)
        window_name = "Select ROI and press Enter"
        roi = cv2.selectROI(window_name, preview, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        x, y, w, h = (int(value) for value in roi)
        if w <= 0 or h <= 0:
            self.roi = None
            return None

        self.roi = (x, y, w, h)
        return self.roi

    def build_output_path(self, video_path: Path, prefix: str, frame_id: int) -> Path:
        filename = f"{video_path.stem}_{prefix}_{frame_id}.jpg"
        output_path = self.config.output_folder / filename

        if not output_path.exists():
            return output_path

        suffix = 1
        while True:
            candidate = self.config.output_folder / (
                f"{video_path.stem}_{prefix}_{frame_id}_{suffix}.jpg"
            )
            if not candidate.exists():
                return candidate
            suffix += 1

    def save_frame(self, video_path: Path, frame_id: int, frame, prefix: str) -> Path:
        path = self.build_output_path(video_path, prefix, frame_id)
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise FrameProcessingError(f"Failed to save frame: {path}")
        return path
