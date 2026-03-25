import numpy as np
from ultralytics import YOLO

from base import BaseDetector


class YOLODetector(BaseDetector):
    """
    Детектор людей на видео с использованием модели YOLO.

    Наследует BaseDetector и реализует метод detect_person для определения
    наличия человека в выбранной ROI.

    Атрибуты:
        model (YOLO): Загруженная модель YOLO.
        output_name (str): Имя выходного видео при записи (по умолчанию "yolo_output.mp4").
    """

    def __init__(self, video_path: str, record: bool = False):
        """
        Инициализация YOLO детектора.

        Args:
            video_path (str): Путь к видеофайлу.
            record (bool): Флаг записи видео. Если True, кадры будут сохраняться.
        """
        super().__init__(video_path, record)
        self.model = YOLO("yolo26n.pt")
        self.output_name = "yolo_output.mp4"

    def detect_person(self, frame: np.ndarray) -> bool:
        """
        Детекция человека на кадре внутри выбранной ROI.

        Алгоритм:
        1. Вырезается ROI из кадра.
        2. Запускается инференс YOLO на ROI.
        3. Проверяются классы обнаруженных объектов.
        4. Если найден объект класса 'person' и он пересекается с ROI, возвращается True.

        Args:
            frame (np.ndarray): Кадр видео.

        Returns:
            bool: True, если человек найден в ROI, иначе False.
        """
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]

        # Запуск YOLO инференса на ROI
        results = self.model(roi_frame, conf=0.3, verbose=False)

        for res in results:
            class_ids = res.boxes.cls
            confidence = res.boxes.conf
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)

            if class_ids is None or confidence is None or boxes is None:
                continue

            # Проверка каждого объекта на кадре
            for cls_id, box, conf in zip(class_ids, boxes, confidence):
                cls_id = int(cls_id.item())
                cls_name = res.names.get(cls_id)
                confidence = round(conf.item(), 2)

                # Игнорируем все объекты кроме людей
                if cls_name != 'person':
                    continue

                # Координаты объекта относительно исходного кадра
                x1, y1, x2, y2 = box
                x1 += x
                x2 += x
                y1 += y
                y2 += y

                # Проверка пересечения с ROI
                if self._is_inside_roi((x1, y1, x2, y2), self.roi):
                    return True

        return False

    @staticmethod
    def _is_inside_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
        """
        Проверяет, пересекается ли bbox с ROI.

        Args:
            box (tuple[int, int, int, int]): Координаты объекта (x1, y1, x2, y2).
            roi (tuple[int, int, int, int]): Координаты ROI (x, y, w, h).

        Returns:
            bool: True, если есть пересечение, иначе False.
        """
        x1, y1, x2, y2 = box
        rx, ry, rw, rh = roi
        return not (x2 < rx or x1 > rx + rw or y2 < ry or y1 > ry + rh)