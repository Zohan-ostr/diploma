#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import math

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def unified_visibility(native_visibility: Optional[float] = None,
                       keypoint_confidence: Optional[float] = None,
                       threshold: float = 0.3) -> float:
    if native_visibility is not None and not math.isnan(float(native_visibility)):
        return clip01(float(native_visibility))
    if keypoint_confidence is None:
        return 0.0
    conf = float(keypoint_confidence)
    if conf <= 0:
        return 0.0
    return 1.0 if conf >= threshold else conf / max(threshold, 1e-9)

def unified_presence(native_presence: Optional[float] = None,
                     keypoint_confidence: Optional[float] = None) -> float:
    if native_presence is not None and not math.isnan(float(native_presence)):
        return clip01(float(native_presence))
    if keypoint_confidence is None:
        return 0.0
    return clip01(float(keypoint_confidence))
