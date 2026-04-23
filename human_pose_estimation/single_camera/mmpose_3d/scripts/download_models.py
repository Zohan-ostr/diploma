#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("mmpose_3d/models prepared.")
    print("Place detector / 2d pose / 3d pose checkpoints here.")
    print("Recommended to install OpenMMLab stack via openmim inside the project docker/runtime.")
    print("This script is intentionally a placeholder, because concrete checkpoint choice depends on your final pipeline.")

if __name__ == "__main__":
    main()
