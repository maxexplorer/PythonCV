# base.py

from abc import ABC, abstractmethod

import cv2
import numpy as np

from dodopizza_test_task.utils import TableStateTracker, Analytics, select_roi, draw_table_box


class BaseDetector(ABC):
    """
    Абстрактный базовый класс для детекторов.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = TableStateTracker()
        self.roi: tuple[int, int, int, int] = (0, 0, 0, 0)

    @abstractmethod
    def detect_person(self, frame: np.ndarray) -> bool:
        """
        Возвращает True, если человек найден в ROI на кадре.
        """
        pass

    def run(self) -> None:
        """
        Основной цикл обработки видео.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Не удалось открыть видео")
            return

        self.roi = select_roi(frame)

        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 1280, 720)

        fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        frame_id: int = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_id / fps
            frame_id += 1

            is_person_present = self.detect_person(frame)
            self.tracker.update(is_person_present, timestamp)
            draw_table_box(frame, self.roi, self.tracker.current_state)

            cv2.imshow("Result", frame)
            if cv2.waitKey(30) == 27:
                break

        df = self.tracker.to_dataframe()
        mean_time = Analytics.compute_mean_idle_time(df)
        print(df)
        print(f"Mean idle time: {mean_time:.2f} sec")

        self.cap.release()
        cv2.destroyAllWindows()