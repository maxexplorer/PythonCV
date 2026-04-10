"""
Утилиты для загрузки данных и обработки аннотаций.
"""

import json
import os
import numpy as np


def load_json(path: str) -> dict:
    """
    Загрузка JSON файла
    """
    with open(path, "r") as f:
        return json.load(f)


def read_split(split_path: str) -> tuple[list[str], list[str]]:
    """
    Читает split.json и возвращает train/val сессии
    """
    split = load_json(split_path)
    return split["train"], split["val"]


def load_pair_annotations(session_path: str, source: str) -> list:
    """
    Загружает аннотации пары изображений.

    source:
        - top
        - bottom
    """
    file_name = f"coords_{source}.json"
    path = os.path.join(session_path, file_name)

    if not os.path.exists(path):
        return []

    return load_json(path)


def extract_matches(annotation: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Извлекает соответствия точек между изображениями.

    Returns:
        src_points (top/bottom)
        dst_points (door2)
    """

    src = []
    dst = []

    for pair in annotation:
        m1 = {p["number"]: p for p in pair["image1_coordinates"]}
        m2 = {p["number"]: p for p in pair["image2_coordinates"]}

        common = set(m1.keys()) & set(m2.keys())

        for k in common:
            src.append([m2[k]["x"], m2[k]["y"]])
            dst.append([m1[k]["x"], m1[k]["y"]])

    return (
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32)
    )