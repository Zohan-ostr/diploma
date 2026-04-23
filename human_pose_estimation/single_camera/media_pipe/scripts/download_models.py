#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple helper: models are expected in ../models. If needed, reuse benchmark.py download command."""
from pathlib import Path
print('Place pose_landmarker_lite.task, pose_landmarker_full.task, pose_landmarker_heavy.task into:')
print(Path(__file__).resolve().parents[1] / 'models')
