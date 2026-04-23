#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import urlretrieve
MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model_dir', default='models'); args = ap.parse_args()
    md = Path(args.model_dir).resolve(); md.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        p = md / name
        if p.exists():
            print(f'[skip] {name} already exists'); continue
        print(f'[download] {name}'); urlretrieve(url, p)
    print(f'Models are in: {md}')
if __name__ == '__main__':
    main()
