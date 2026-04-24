#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Iterable, Optional
import numpy as np

def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))

def detection_rate(has_pose_flags: Iterable[int]) -> float:
    vals = [int(v) for v in has_pose_flags]
    if not vals:
        return 0.0
    return float(np.mean(vals))

def effective_fps_from_inference_ms(inference_ms_values: Sequence[float]) -> float:
    vals = [float(v) for v in inference_ms_values if float(v) > 0]
    if not vals:
        return 0.0
    fps = [1000.0 / v for v in vals]
    return float(np.mean(fps))

def effective_fps_from_wall(frames_processed: int, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return float(frames_processed / wall_seconds)

def jitter_from_world_frames(curr_pts: np.ndarray, prev_pts: np.ndarray, stable_idxs: Sequence[int]) -> float:
    valid = []
    for idx in stable_idxs:
        if idx < len(curr_pts) and idx < len(prev_pts):
            if np.isfinite(curr_pts[idx]).all() and np.isfinite(prev_pts[idx]).all():
                valid.append(idx)
    if not valid:
        return float("nan")
    diffs = curr_pts[valid] - prev_pts[valid]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))

def mean_metric(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(np.mean(vals))

def std_metric(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(np.std(vals))
