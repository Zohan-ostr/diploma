#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import yaml

REQUIRED = [
    "left.mp4", "right.mp4",
    "calib/intrinsics_left.yaml",
    "calib/intrinsics_right.yaml",
    "calib/extrinsics.yaml",
    "calib/stereo.yaml",
]

def main():
    project_root = Path(__file__).resolve().parents[1]
    sessions_dir = (project_root / "../../videos/double_cameras").resolve()
    if not sessions_dir.exists():
        raise RuntimeError(f"Sessions dir not found: {sessions_dir}")

    sessions = sorted([p for p in sessions_dir.iterdir() if p.is_dir()])
    if not sessions:
        raise RuntimeError(f"No stereo sessions found in {sessions_dir}")

    print("Checking stereo sessions for VoxelPose...")
    for session in sessions:
        missing = [rel for rel in REQUIRED if not (session / rel).exists()]
        if missing:
            print(f"[WARN] {session.name}: missing {missing}")
        else:
            print(f"[OK] {session.name}")
        meta = session / "meta.yaml"
        if meta.exists():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            print(f"      meta: sync_mode={data.get('sync_mode', 'unknown')} frame_offset_right={data.get('frame_offset_right', 0)}")
        else:
            print(f"[WARN] {session.name}: meta.yaml missing")

if __name__ == "__main__":
    main()
