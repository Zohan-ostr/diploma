#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("openpose_3d/models prepared.")
    print("This project expects OpenPose binaries / model assets to be provided externally or mounted from a dedicated OpenPose runtime.")
    print("Also check calibration_templates/ for required camera calibration files.")

if __name__ == "__main__":
    main()
