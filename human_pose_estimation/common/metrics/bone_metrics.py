#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Sequence
import numpy as np

def upper_arm_metrics(world_points: np.ndarray,
                      left_shoulder_idx: int,
                      left_elbow_idx: int,
                      arm_len_ref_m: float) -> Tuple[float, float, float]:
    if left_shoulder_idx >= len(world_points) or left_elbow_idx >= len(world_points):
        return float("nan"), float("nan"), float("nan")
    ls = world_points[left_shoulder_idx]
    le = world_points[left_elbow_idx]
    if not np.isfinite(ls).all() or not np.isfinite(le).all():
        return float("nan"), float("nan"), float("nan")

    arm_len = float(np.linalg.norm(le - ls))
    abs_err = abs(arm_len - arm_len_ref_m)
    rel_err = (100.0 * abs_err / arm_len_ref_m) if arm_len_ref_m > 0 else float("nan")
    return arm_len, abs_err, rel_err

def aggregate_bone_metrics(arm_lens: Sequence[float],
                           abs_errs: Sequence[float],
                           rel_errs: Sequence[float]) -> dict:
    def clean(xs):
        return [float(x) for x in xs if np.isfinite(float(x))]
    arm_lens = clean(arm_lens)
    abs_errs = clean(abs_errs)
    rel_errs = clean(rel_errs)

    return {
        "left_upper_arm_len_m_mean": float(np.mean(arm_lens)) if arm_lens else 0.0,
        "left_upper_arm_len_m_std": float(np.std(arm_lens)) if arm_lens else 0.0,
        "left_upper_arm_abs_error_m_mean": float(np.mean(abs_errs)) if abs_errs else 0.0,
        "left_upper_arm_abs_error_m_p95": float(np.percentile(abs_errs, 95)) if abs_errs else 0.0,
        "left_upper_arm_rel_error_mean_pct": float(np.mean(rel_errs)) if rel_errs else 0.0,
    }
