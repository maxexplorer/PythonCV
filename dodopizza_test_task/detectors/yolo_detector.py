import numpy as np
from ultralytics import YOLO

from dodopizza_test_task.detectors.base import BaseDetector



class YOLODetector(BaseDetector):
    """
    Детекция через YOLO в ROI.
    """

    def __init__(self, video_path: str):
        super().__init__(video_path)
        self.model = YOLO("yolo26n.pt")

    def detect_person(self, frame: np.ndarray) -> bool:
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]
        results = self.model(roi_frame, conf=0.3, verbose=False)

        for res in results:
            for box in res.boxes:
                cls_id = int(box.cls.item())
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += x
                x2 += x
                y1 += y
                y2 += y
                if self._is_inside_roi((x1, y1, x2, y2), self.roi):
                    return True
        return False

    @staticmethod
    def _is_inside_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        rx, ry, rw, rh = roi
        return not (x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh)