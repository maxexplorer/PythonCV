# tracker.py

__all__ = ['YOLOTracker']

import numpy as np
import torch
import cv2
from ultralytics import YOLO
from configs.config import Config


class YOLOTracker:
    """
    Класс для трекинга транспортных средств с использованием модели YOLO и встроенного трекера.
    """

    def __init__(self, roi_resized=(1280, 720)):
        self._model = None
        self._config = Config
        self._device = self._config.Device.CUDA.value if torch.cuda.is_available() else self._config.Device.CPU.value

        self.cap = None
        self.roi_resized = roi_resized
        self.roi = None

    def load_model(self) -> YOLO:
        """
        Загружает YOLO модель и переносит её на нужное устройство.
        """
        try:
            self._model = YOLO(self._config.TrackerConfig.model_path)
            self._model.to(self._device)
            self._model.fuse()
        except Exception as e:
            print(f"Не удалось загрузить модель '{self._config.TrackerConfig.model_path}'. Ошибка: {e}")
            raise
        return self._model

    def select_roi(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Не удалось прочитать первый кадр.")

        if frame.shape[1] != self.roi_resized[0] or frame.shape[0] != self.roi_resized[1]:
            frame = cv2.resize(frame, self.roi_resized)

        # Добавим текст на кадр
        text = "Select ROI with mouse and press ENTER"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        x = (frame.shape[1] - w) // 2
        y = frame.shape[0] // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Теперь вызов selectROI — оно само откроет и закроет окно
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        self.roi = roi
        print(f"Selected ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    def _draw_box(self, frame: np.ndarray, box: list[int], track_id: int, cls_name: str, conf: float) -> None:
        """
        Отрисовывает bounding box, ID и confidence на кадре.
        """
        x1, y1, x2, y2 = box
        color = (0, 0, 255)
        label = f'{cls_name} {conf:.2f} | ID: {track_id}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def predict(self, frame: np.ndarray) -> None:
        """
        Выполняет трекинг объектов на кадре и отображает результат только для автомобилей и мотоциклов.
        """

        # Обрезка по ROI, если выбран
        if self.roi:
            x, y, w, h = self.roi
            frame = frame[y:y + h, x:x + w]

        results = self._model.track(
            frame,
            tracker=self._config.TrackerConfig.track_path,
            persist=self._config.TrackerConfig.persist,
            verbose=self._config.TrackerConfig.verbose
        )

        # Целевые классы
        target_classes = {'car', 'motorcycle', 'bus', 'truck'}

        for res in results:
            if res.boxes is None:
                continue

            class_ids = res.boxes.cls
            track_ids = res.boxes.id
            confidences = res.boxes.conf
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)

            if class_ids is None or track_ids is None or confidences is None:
                continue

            for cls_id, track_id, box, conf in zip(class_ids, track_ids, boxes, confidences):
                cls_id = int(cls_id.item())
                track_id = int(track_id.item())
                conf = float(conf.item())
                cls_name = res.names.get(cls_id, 'unknown')

                if cls_name not in target_classes:
                    continue

                self._draw_box(frame, box, track_id, cls_name, conf)

        resized_frame = cv2.resize(frame, self.roi_resized)
        cv2.imshow('frame', resized_frame)
