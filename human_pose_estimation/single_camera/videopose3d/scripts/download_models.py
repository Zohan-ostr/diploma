#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("videopose3d/models prepared.")
    print("Place VideoPose3D checkpoints here.")
    print("Also prepare a 2D keypoint backend, because VideoPose3D expects 2D keypoints as input.")

if __name__ == "__main__":
    main()
