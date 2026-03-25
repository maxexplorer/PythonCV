from abc import ABC, abstractmethod
import cv2
import numpy as np
from dodopizza_test_task.utils import TableStateTracker, Analytics, select_roi, draw_table_box


class BaseDetector(ABC):
    """
    Абстрактный базовый класс для детекторов.
    """

    def __init__(self, video_path: str, record: bool = False):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = TableStateTracker()
        self.roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.record = record
        self.out: cv2.VideoWriter | None = None

    @abstractmethod
    def detect_person(self, frame: np.ndarray) -> bool:
        """
        Проверяет наличие человека в ROI на кадре.
        Возвращает True, если человек найден.
        """
        pass

    def run(self) -> None:
        """
        Основной цикл обработки видео с визуализацией, FSM и опциональной записью.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Не удалось открыть видео")
            return

        self.roi = select_roi(frame)

        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 1280, 720)

        # Настройка записи видео
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_name = "output.mp4"  # можно переопределять в YOLO/OPENCV
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(output_name, fourcc, self.cap.get(cv2.CAP_PROP_FPS), (width, height))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_id / fps
            frame_id += 1

            is_person_present = self.detect_person(frame)
            self.tracker.update(is_person_present, timestamp)
            draw_table_box(frame, self.roi, self.tracker.current_state)

            if self.record and self.out is not None:
                self.out.write(frame)

            cv2.imshow("Result", frame)
            if cv2.waitKey(30) == 27:
                break

        df = self.tracker.to_dataframe()
        mean_time = Analytics.compute_mean_idle_time(df)
        print(df)
        print(f"Mean idle time: {mean_time:.2f} sec")

        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()