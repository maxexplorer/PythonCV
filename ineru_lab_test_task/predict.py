# predict.py

import os
import json
import pickle
import numpy as np
import cv2

from utils import read_split, load_pair_annotations, extract_matches


class HPolyPredictor:
    def __init__(self, models):
        self.models = models

    @classmethod
    def load(cls, model_path):
        with open(model_path, "rb") as f:
            models = pickle.load(f)
        return cls(models)

    def _normalize_point(self, x, y, w=3200, h=1800):
        return np.array([x / w, y / h], dtype=np.float32)

    def _denormalize_point(self, x, y, w=3200, h=1800):
        return np.array([x * w, y * h], dtype=np.float32)

    def predict_point(self, x, y, source):
        model = self.models[source]
        pt = self._normalize_point(x, y).reshape(1, 1, 2)

        if model["type"] == "homography":
            out = cv2.perspectiveTransform(pt, model["matrix"])
            out = out[0][0]
        else:
            A = model["matrix"]
            x_n, y_n = pt[0, 0]

            out = np.dot(A[:, :2], np.array([x_n, y_n])) + A[:, 2]

        return self._denormalize_point(out[0], out[1])

    # -----------------------------
    # FIXED MED (PRODUCTION GRADE)
    # -----------------------------
    def evaluate(self, sessions):
        results = {}

        for source in ["top", "bottom"]:
            all_errors = []

            for s in sessions:
                ann = load_pair_annotations(s, source)
                if not ann:
                    continue

                src, dst = extract_matches(ann)
                if len(src) == 0:
                    continue

                preds = np.array([
                    self.predict_point(x, y, source)
                    for x, y in src
                ])

                errors = np.linalg.norm(dst - preds, axis=1)
                all_errors.extend(errors)

            results[source] = float(np.mean(all_errors)) if all_errors else None

        return results

    def run(self, split_path):
        _, val_sessions = read_split(split_path)

        results = self.evaluate(val_sessions)

        print("\n===== FINAL METRICS =====")
        print(f"TOP → DOOR2    : {results['top']:.3f}")
        print(f"BOTTOM → DOOR2 : {results['bottom']:.3f}")

        os.makedirs("artifacts", exist_ok=True)

        output = {
            "top_to_door2_med": results["top"],
            "bottom_to_door2_med": results["bottom"]
        }

        with open("artifacts/metrics.json", "w") as f:
            json.dump(output, f, indent=2)

        with open("artifacts/metrics_report.txt", "w") as f:
            f.write("=== METRICS REPORT ===\n")
            f.write(f"TOP -> DOOR2: {results['top']:.3f}\n")
            f.write(f"BOTTOM -> DOOR2: {results['bottom']:.3f}\n")