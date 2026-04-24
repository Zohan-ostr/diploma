#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def load_camera_intrinsics(path: Path):
    data = load_yaml(path)
    K = np.asarray(data.get("K", []), dtype=np.float64).reshape(3, 3)
    D = np.asarray(data.get("D", []), dtype=np.float64).reshape(-1)
    return K, D, data

def load_extrinsics(path: Path):
    data = load_yaml(path)
    R = np.asarray(data.get("R", []), dtype=np.float64).reshape(3, 3)
    T = np.asarray(data.get("T", []), dtype=np.float64).reshape(3, 1)
    return R, T, data

def load_stereo_projection(path: Path):
    data = load_yaml(path)
    P1 = np.asarray(data.get("P1", []), dtype=np.float64).reshape(3, 4)
    P2 = np.asarray(data.get("P2", []), dtype=np.float64).reshape(3, 4)
    Q = np.asarray(data.get("Q", []), dtype=np.float64).reshape(4, 4) if data.get("Q", None) else None
    return P1, P2, Q, data
