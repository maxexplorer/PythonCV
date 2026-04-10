# train.py

import os
import pickle
import numpy as np
import cv2

from ineru_lab_test_task.utils import read_split, load_pair_annotations, extract_matches


class HPolyMapperTrainer:
    def __init__(self):
        self.models = {
            "top": None,
            "bottom": None
        }

    def _normalize(self, pts, w=3200, h=1800):
        pts = pts.astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        return pts

    def _denormalize(self, pts, w=3200, h=1800):
        pts = pts.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts

    def train_single(self, sessions, source):
        all_src = []
        all_dst = []

        for s in sessions:
            ann = load_pair_annotations(s, source)
            if not ann:
                continue

            src, dst = extract_matches(ann)

            if len(src) < 4:
                continue

            all_src.append(src)
            all_dst.append(dst)

        if len(all_src) == 0:
            raise ValueError(f"No data for {source}")

        all_src = np.vstack(all_src)
        all_dst = np.vstack(all_dst)

        # ----------------------------
        # NORMALIZATION (VERY IMPORTANT)
        # ----------------------------
        src_n = self._normalize(all_src)
        dst_n = self._normalize(all_dst)

        # ----------------------------
        # HOMOGRAPHY WITH RANSAC
        # ----------------------------
        H, mask = cv2.findHomography(src_n, dst_n, cv2.RANSAC, 5.0)

        if H is None:
            H = np.eye(3, dtype=np.float32)

        # ----------------------------
        # FILTER INLIERS ONLY
        # ----------------------------
        if mask is not None:
            mask = mask.ravel().astype(bool)
            if np.sum(mask) > 10:
                src_n = src_n[mask]
                dst_n = dst_n[mask]
                H, _ = cv2.findHomography(src_n, dst_n, 0)

        # ----------------------------
        # FALLBACK: AFFINE if bad homography
        # ----------------------------
        if H is None or np.any(np.isnan(H)):
            H_aff, _ = cv2.estimateAffine2D(src_n, dst_n)
            if H_aff is None:
                H_aff = np.eye(2, 3, dtype=np.float32)

            return {
                "type": "affine",
                "matrix": H_aff
            }

        return {
            "type": "homography",
            "matrix": H
        }

    def train(self, split_path):
        train_sessions, _ = read_split(split_path)

        self.models["top"] = self.train_single(train_sessions, "top")
        self.models["bottom"] = self.train_single(train_sessions, "bottom")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.models, f)