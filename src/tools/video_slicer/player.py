from pathlib import Path

import cv2


class VideoPlayer:
    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.current_frame_id = 0

    def open(self, video_path: Path) -> None:
        self.close()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.capture = capture
        self.video_path = video_path
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.current_frame_id = 0

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.video_path = None
        self.frame_count = 0
        self.current_frame_id = 0

    def read_next(self) -> tuple[bool, object | None, int]:
        if self.capture is None:
            return False, None, self.current_frame_id

        ok, frame = self.capture.read()
        if not ok:
            return False, None, self.current_frame_id

        self.current_frame_id = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return True, frame, self.current_frame_id

    def read_at(self, zero_based_frame_id: int) -> tuple[bool, object | None, int]:
        if self.capture is None:
            return False, None, self.current_frame_id

        if self.frame_count:
            zero_based_frame_id = max(0, min(zero_based_frame_id, self.frame_count - 1))
        else:
            zero_based_frame_id = max(0, zero_based_frame_id)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, zero_based_frame_id)
        return self.read_next()
