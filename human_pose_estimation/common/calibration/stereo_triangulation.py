#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

def triangulate_point(P1: np.ndarray, P2: np.ndarray,
                      pt1: Tuple[float, float], pt2: Tuple[float, float]) -> np.ndarray:
    pts1 = np.asarray(pt1, dtype=np.float64).reshape(2, 1)
    pts2 = np.asarray(pt2, dtype=np.float64).reshape(2, 1)
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X = X_h[:3] / X_h[3]
    return X.reshape(3)

def triangulate_keypoints(P1: np.ndarray, P2: np.ndarray,
                          kpts_left: np.ndarray, kpts_right: np.ndarray) -> np.ndarray:
    assert kpts_left.shape == kpts_right.shape
    out = np.full((kpts_left.shape[0], 3), np.nan, dtype=np.float64)
    for i in range(kpts_left.shape[0]):
        if np.isfinite(kpts_left[i]).all() and np.isfinite(kpts_right[i]).all():
            out[i] = triangulate_point(P1, P2, tuple(kpts_left[i]), tuple(kpts_right[i]))
    return out
