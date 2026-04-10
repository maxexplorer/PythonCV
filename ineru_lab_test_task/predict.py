"""
Предсказание и оценка качества модели преобразования координат
"""

import os
import json
import pickle
import numpy as np
import cv2

from utils import (
    read_split,
    load_pair_annotations,
    extract_matches
)


class HPolyPredictor:
    """
    Выполняет:
    - предсказание координат
    - оценку ошибки (MAE)
    """

    def __init__(self, models: dict[str, dict]) -> None:
        self.models = models

    @classmethod
    def load(cls, model_path: str) -> "HPolyPredictor":
        """
        Загружает обученную модель
        """
        with open(model_path, "rb") as f:
            models = pickle.load(f)
        return cls(models)

    def _normalize_point(self, x: float, y: float) -> np.ndarray:
        return np.array([x / 3200, y / 1800], dtype=np.float32)

    def _denormalize_point(self, x: float, y: float) -> np.ndarray:
        return np.array([x * 3200, y * 1800], dtype=np.float32)

    def predict_point(self, x: float, y: float, source: str) -> np.ndarray:
        """
        Преобразует точку из source → door2
        """
        model = self.models[source]

        pt = self._normalize_point(x, y).reshape(1, 1, 2)

        if model["type"] == "homography":
            out = cv2.perspectiveTransform(pt, model["matrix"])[0][0]
        else:
            A = model["matrix"]
            x_n, y_n = pt[0, 0]
            out = A[:, :2] @ np.array([x_n, y_n]) + A[:, 2]

        return self._denormalize_point(out[0], out[1])

    def evaluate(self, sessions: list[str]) -> dict[str, float | None]:
        """
        Средняя ошибка (MAE в пикселях)
        """
        results: dict[str, float | None] = {}

        for source in ["top", "bottom"]:
            errors: list[float] = []

            for session in sessions:
                ann = load_pair_annotations(session, source)
                if not ann:
                    continue

                src, dst = extract_matches(ann)

                if len(src) == 0:
                    continue

                preds = np.array([
                    self.predict_point(x, y, source)
                    for x, y in src
                ])

                err = np.linalg.norm(dst - preds, axis=1)
                errors.extend(err.tolist())

            results[source] = float(np.mean(errors)) if errors else None

        return results

    def run(self, split_path: str) -> None:
        """
        Запуск оценки + сохранение метрик
        """
        _, val_sessions = read_split(split_path)

        results = self.evaluate(val_sessions)

        print("\n===== METRICS =====")
        print(f"TOP → DOOR2    : {results['top']:.3f}")
        print(f"BOTTOM → DOOR2 : {results['bottom']:.3f}")

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/metrics.json", "w") as f:
            json.dump(
                {
                    "top_to_door2_med": results["top"],
                    "bottom_to_door2_med": results["bottom"]
                },
                f,
                indent=2
            )

        with open("artifacts/metrics_report.txt", "w") as f:
            f.write("=== METRICS REPORT ===\n")
            f.write(f"TOP -> DOOR2: {results['top']:.3f}\n")
            f.write(f"BOTTOM -> DOOR2: {results['bottom']:.3f}\n")