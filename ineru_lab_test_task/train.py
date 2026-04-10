"""
Обучение модели геометрического преобразования координат:
- Homography (RANSAC)
- fallback: Affine transformation
"""

import os
import pickle
import numpy as np
import cv2

from ineru_lab_test_task.utils import (
    read_split,
    load_pair_annotations,
    extract_matches
)


class HPolyMapperTrainer:
    """
    Обучает модели преобразования координат:
    top → door2
    bottom → door2
    """

    def __init__(self) -> None:
        # словарь обученных моделей
        self.models: dict[str, dict | None] = {
            "top": None,
            "bottom": None
        }

    def _normalize(self, pts: np.ndarray) -> np.ndarray:
        """
        Нормализация координат в диапазон [0..1]
        (уменьшает влияние масштаба изображения)
        """
        pts = pts.astype(np.float32)
        pts[:, 0] /= 3200
        pts[:, 1] /= 1800
        return pts

    def _denormalize(self, pts: np.ndarray) -> np.ndarray:
        """
        Возврат координат в пиксельное пространство
        """
        pts = pts.copy()
        pts[:, 0] *= 3200
        pts[:, 1] *= 1800
        return pts

    def train_single(self, sessions: list[str], source: str) -> dict:
        """
        Обучение одной модели (top или bottom)

        Args:
            sessions: список тренировочных сессий
            source: источник координат ("top" | "bottom")

        Returns:
            dict:
                {
                    "type": "homography" | "affine",
                    "matrix": np.ndarray
                }
        """

        src_all: list[np.ndarray] = []
        dst_all: list[np.ndarray] = []

        # -----------------------------
        # сбор всех точек из датасета
        # -----------------------------
        for session in sessions:
            ann = load_pair_annotations(session, source)
            if not ann:
                continue

            src, dst = extract_matches(ann)

            if len(src) < 4:
                continue

            src_all.append(src)
            dst_all.append(dst)

        if not src_all:
            raise ValueError(f"No training data for source={source}")

        src_all_np = np.vstack(src_all)
        dst_all_np = np.vstack(dst_all)

        # -----------------------------
        # нормализация координат
        # -----------------------------
        src_n = self._normalize(src_all_np)
        dst_n = self._normalize(dst_all_np)

        # -----------------------------
        # RANSAC homography
        # -----------------------------
        H, mask = cv2.findHomography(src_n, dst_n, cv2.RANSAC, 5.0)

        if H is None:
            H = np.eye(3, dtype=np.float32)

        # -----------------------------
        # фильтрация inliers
        # -----------------------------
        if mask is not None:
            inliers = mask.ravel().astype(bool)

            if np.sum(inliers) > 10:
                src_n = src_n[inliers]
                dst_n = dst_n[inliers]
                H, _ = cv2.findHomography(src_n, dst_n, 0)

        # -----------------------------
        # fallback: affine model
        # -----------------------------
        if H is None or np.isnan(H).any():
            A, _ = cv2.estimateAffine2D(src_n, dst_n)

            if A is None:
                A = np.eye(2, 3, dtype=np.float32)

            return {
                "type": "affine",
                "matrix": A
            }

        return {
            "type": "homography",
            "matrix": H
        }

    def train(self, split_path: str) -> None:
        """
        Обучение моделей top и bottom
        """
        train_sessions, _ = read_split(split_path)

        self.models["top"] = self.train_single(train_sessions, "top")
        self.models["bottom"] = self.train_single(train_sessions, "bottom")

    def save(self, path: str) -> None:
        """
        Сохранение модели (pickle)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.models, f)