#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("MMPose 3D uses MMPoseInferencer and model aliases.")
    print("For the default pipeline, the project uses pose3d alias: human3d")
    print("MMPose can automatically download needed checkpoints on first run if the environment is configured correctly.")
    print("models/ directory has been prepared:", model_dir)

if __name__ == "__main__":
    main()
