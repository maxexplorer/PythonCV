import cv2
import numpy as np

from base import BaseDetector


class OpenCVDetector(BaseDetector):
    """
    Детектор людей на видео с использованием OpenCV и вычитания фона.

    Наследует BaseDetector и реализует метод detect_person для определения
    наличия человека в выбранной ROI.

    Атрибуты:
        back_sub (cv2.BackgroundSubtractorMOG2): Созданный фоновый субтрактивный детектор.
        output_name (str): Имя выходного видео при записи (по умолчанию "opencv_output.mp4").
    """

    def __init__(self, video_path: str, record: bool = False):
        """
        Инициализация OpenCV детектора.

        Args:
            video_path (str): Путь к видеофайлу.
            record (bool): Флаг записи видео. Если True, кадры будут сохраняться.
        """
        super().__init__(video_path, record)
        # Создаем субтрактивный фоновый детектор
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.output_name = "opencv_output.mp4"

    def detect_person(self, frame: np.ndarray) -> bool:
        """
        Детекция человека на кадре внутри выбранной ROI с использованием фонового вычитания.

        Алгоритм:
        1. Вырезается ROI из кадра.
        2. Применяется фоновое вычитание для выделения движущихся объектов.
        3. Применяется медианный фильтр для сглаживания маски.
        4. Находятся контуры объектов.
        5. Если контур достаточно большой (> 500 пикселей), считаем, что человек присутствует.

        Args:
            frame (np.ndarray): Кадр видео.

        Returns:
            bool: True, если человек найден в ROI, иначе False.
        """
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]

        # Применяем субтрактивный детектор фона
        fg_mask = self.back_sub.apply(roi_frame)

        # Сглаживание маски для уменьшения шума
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Поиск контуров на маске
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Проверка каждого контура на достаточный размер
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                return True

        return False