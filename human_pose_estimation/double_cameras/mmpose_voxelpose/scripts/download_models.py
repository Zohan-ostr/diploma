#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

def main():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print("mmpose_voxelpose/models prepared.")
    print("Place VoxelPose / MMPose checkpoints here.")
    print("Recommended workflow:")
    print("1) install OpenMMLab stack via openmim")
    print("2) prepare the chosen VoxelPose config and checkpoints")
    print("3) adapt benchmark.py to real inference")

if __name__ == "__main__":
    main()
