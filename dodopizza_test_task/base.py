from abc import ABC, abstractmethod

import cv2
import numpy as np

from utils import TableStateTracker, Analytics, select_roi, draw_table_box


class BaseDetector(ABC):
    """
    Абстрактный базовый класс для детекторов людей на видео.

    Обеспечивает:
    - работу с видео через OpenCV,
    - FSM для отслеживания состояний столика,
    - визуализацию ROI и состояния,
    - опциональную запись выходного видео.

    Атрибуты:
        video_path (str): Путь к исходному видеофайлу.
        cap (cv2.VideoCapture): Объект OpenCV для чтения видео.
        tracker (TableStateTracker): FSM для отслеживания событий.
        roi (tuple[int, int, int, int]): Выбранная пользователем область интереса (ROI).
        record (bool): Флаг включения записи видео.
        out (cv2.VideoWriter | None): Объект записи видео, если record=True.
    """

    def __init__(self, video_path: str, record: bool = False):
        """
        Инициализация детектора.

        Args:
            video_path (str): Путь к видеофайлу.
            record (bool): Нужно ли записывать выходное видео. По умолчанию False.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = TableStateTracker()
        self.roi: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.record = record
        self.out: cv2.VideoWriter | None = None

    @abstractmethod
    def detect_person(self, frame: np.ndarray) -> bool:
        """
        Абстрактный метод детекции человека на кадре в ROI.

        Args:
            frame (np.ndarray): Кадр видео.

        Returns:
            bool: True, если человек найден, иначе False.
        """
        pass

    def run(self) -> None:
        """
        Основной цикл обработки видео:
        1. Чтение кадра и выбор ROI пользователем.
        2. Настройка окна для визуализации.
        3. Настройка записи видео (если включена).
        4. Обход кадров видео, детекция человека, обновление FSM и отрисовка ROI.
        5. Отображение кадра, запись при необходимости.
        6. Вывод таблицы событий и средней продолжительности пустого состояния.
        """
        # Чтение первого кадра для выбора ROI
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Не удалось открыть видео")
            return

        # Пользователь выбирает ROI на первом кадре
        self.roi = select_roi(frame)

        # Настройка окна для отображения результатов
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 1280, 720)

        # Настройка записи видео при включенном флаге
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_name = "output.mp4"  # Можно переопределять в дочерних классах
            height, width = frame.shape[:2]  # OpenCV требует (width, height)
            self.out = cv2.VideoWriter(
                output_name, fourcc, self.cap.get(cv2.CAP_PROP_FPS), (width, height)
            )

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_id / fps
            frame_id += 1

            # Детекция человека на текущем кадре
            is_person_present = self.detect_person(frame)

            # Обновление FSM с текущим состоянием
            self.tracker.update(is_person_present, timestamp)

            # Отрисовка ROI и состояния на кадре
            draw_table_box(frame, self.roi, self.tracker.current_state)

            # Запись кадра в видео (если включено)
            if self.record and self.out is not None:
                self.out.write(frame)

            # Отображение кадра пользователю
            cv2.imshow("Result", frame)
            if cv2.waitKey(30) == 27:  # ESC для выхода
                break

        # Вывод итоговой таблицы событий и средней продолжительности пустого состояния
        df = self.tracker.to_dataframe()
        mean_time = Analytics.compute_mean_idle_time(df)
        print(df)
        print(f"Mean idle time: {mean_time:.2f} sec")

        # Освобождение ресурсов
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()