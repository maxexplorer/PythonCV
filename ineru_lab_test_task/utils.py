# utils.py

import json
import os
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_split(split_path):
    split = load_json(split_path)
    return split["train"], split["val"]


def load_pair_annotations(session_path, source):
    """
    source: top / bottom
    """
    file_name = f"coords_{source}.json"
    path = os.path.join(session_path, file_name)

    if not os.path.exists(path):
        return []

    return load_json(path)


def extract_matches(annotation):
    """
    returns:
    src_points, dst_points
    """
    src = []
    dst = []

    for pair in annotation:
        m1 = {p["number"]: p for p in pair["image1_coordinates"]}
        m2 = {p["number"]: p for p in pair["image2_coordinates"]}

        common = set(m1.keys()) & set(m2.keys())

        for k in common:
            src.append([m2[k]["x"], m2[k]["y"]])  # top/bottom
            dst.append([m1[k]["x"], m1[k]["y"]])  # door2

    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)
