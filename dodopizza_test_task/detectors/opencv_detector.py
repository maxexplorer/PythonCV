import cv2
import numpy as np

from .base import BaseDetector

class OpenCVDetector(BaseDetector):
    """
    Детекция через OpenCV (вычитание фона) в ROI.
    """

    def __init__(self, video_path: str, record: bool = False):
        super().__init__(video_path, record)
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.output_name = "opencv_output.mp4"

    def detect_person(self, frame: np.ndarray) -> bool:
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]
        fg_mask = self.back_sub.apply(roi_frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                return True
        return False