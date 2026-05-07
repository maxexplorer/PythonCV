from pathlib import Path

import cv2

from config import Roi, SlicerConfig


class FrameProcessingError(RuntimeError):
    pass


class VideoFrameExtractor:
    SUPPORTED_EXTENSIONS = {".avi", ".mov", ".mp4", ".mkv", ".m4v"}

    def __init__(self, config: SlicerConfig) -> None:
        self.config = config
        self.roi: Roi | None = config.roi
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
        window_name = "Select ROI: drag mouse, Enter to apply, Esc to cancel"
        roi = self._select_roi_with_size_overlay(window_name, preview)
        cv2.destroyWindow(window_name)

        if roi is None:
            self.roi = None
            return None

        x, y, w, h = roi
        if w <= 0 or h <= 0:
            self.roi = None
            return None

        self.roi = (x, y, w, h)
        return self.roi

    def _select_roi_with_size_overlay(self, window_name: str, frame) -> Roi | None:
        state = {
            "dragging": False,
            "start": None,
            "end": None,
            "roi": self.roi,
        }

        def on_mouse(event, x, y, flags, param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["start"] = (x, y)
                state["end"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
                state["end"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["dragging"] = False
                state["end"] = (x, y)
                state["roi"] = self._normalize_roi(state["start"], state["end"])

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            canvas = frame.copy()
            active_roi = state["roi"]
            if state["start"] is not None and state["end"] is not None:
                active_roi = self._normalize_roi(state["start"], state["end"])

            if active_roi is not None:
                self._draw_roi_overlay(canvas, active_roi)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(16) & 0xFF
            if key in (13, 10):
                return state["roi"]
            if key == 27:
                return None

    def _normalize_roi(self, start: tuple[int, int] | None, end: tuple[int, int] | None) -> Roi | None:
        if start is None or end is None:
            return None

        x1, y1 = start
        x2, y2 = end
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return x, y, width, height

    def _draw_roi_overlay(self, frame, roi: Roi) -> None:
        x, y, width, height = roi
        if width <= 0 or height <= 0:
            return

        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = f"{width} x {height}"
        text_origin = (x, max(18, y - 8))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        bg_top_left = (text_origin[0], text_origin[1] - text_height - baseline)
        bg_bottom_right = (text_origin[0] + text_width + 8, text_origin[1] + baseline)
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

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
