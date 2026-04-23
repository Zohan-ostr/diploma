#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import yaml

@dataclass
class VideoRunSummary:
    set_name: str
    method: str
    video_or_session: str
    model: str
    mode: str
    delegate: str
    frames_total: int
    frames_processed: int
    effective_fps: float
    detection_rate: float
    mean_visibility: float
    mean_presence: float
    jitter_world_m_mean: float
    jitter_world_m_p95: float
    arm_len_ref_m: float
    left_upper_arm_len_m_mean: float
    left_upper_arm_len_m_std: float
    left_upper_arm_abs_error_m_mean: float
    left_upper_arm_abs_error_m_p95: float
    left_upper_arm_rel_error_mean_pct: float

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def merge_dict(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out

def load_config(project_root: Path) -> dict:
    return merge_dict(
        load_yaml(project_root / "configs" / "default.yaml"),
        load_yaml(project_root / "configs" / "local.yaml"),
    )

def list_videos(videos_dir: Path):
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

def write_empty_leaf(out_dir: Path):
    safe_mkdir(out_dir)
    with (out_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx","timestamp_ms","has_pose","inference_ms",
            "mean_visibility","mean_presence","jitter_frame_m",
            "left_upper_arm_len_m","left_upper_arm_abs_error_m","left_upper_arm_rel_error_pct",
        ])
    with (out_dir / "landmarks_2d.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx","timestamp_ms","landmark_idx","landmark_name","x","y","z","visibility","presence"])
    with (out_dir / "landmarks_3d_world.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx","timestamp_ms","landmark_idx","landmark_name","x_m","y_m","z_m","visibility","presence"])
    with (out_dir / "bones_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx","timestamp_ms","arm_len_ref_m",
            "left_upper_arm_len_m","left_upper_arm_abs_error_m","left_upper_arm_rel_error_pct"
        ])

def write_global_summary(out_root: Path, summaries: List[VideoRunSummary]) -> None:
    if not summaries:
        return
    header = list(asdict(summaries[0]).keys())
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in summaries:
            d = asdict(s)
            w.writerow([d[h] for h in header])
    (out_root / "summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default=cfg.get("videos_dir", "../../videos/single_camera"))
    ap.add_argument("--out_dir", default=str(project_root / "runs" / cfg.get("run_names", {}).get("baseline", "run")))
    ap.add_argument("--device", default=cfg.get("single_camera", {}).get("device", "cpu"))
    ap.add_argument("--arm_len_ref_m", type=float, default=float(cfg.get("arm_len_ref_m", 0.249)))
    args = ap.parse_args()

    videos_dir = (project_root / args.videos_dir).resolve() if not Path(args.videos_dir).is_absolute() else Path(args.videos_dir)
    out_root = Path(args.out_dir).resolve() if Path(args.out_dir).is_absolute() else (project_root / args.out_dir).resolve()
    safe_mkdir(out_root)

    videos = list_videos(videos_dir)
    if not videos:
        raise RuntimeError(f"No videos found in {videos_dir}")

    summaries: List[VideoRunSummary] = []
    model_name = cfg.get("pipeline", {}).get("pose3d_backend", "videopose3d")
    for video in videos:
        leaf = out_root / video.stem / model_name / "VIDEO" / args.device.upper()
        write_empty_leaf(leaf)
        summaries.append(VideoRunSummary(
            set_name=out_root.name,
            method="videopose3d",
            video_or_session=video.name,
            model=model_name,
            mode="VIDEO",
            delegate=args.device.upper(),
            frames_total=0,
            frames_processed=0,
            effective_fps=0.0,
            detection_rate=0.0,
            mean_visibility=0.0,
            mean_presence=0.0,
            jitter_world_m_mean=0.0,
            jitter_world_m_p95=0.0,
            arm_len_ref_m=float(args.arm_len_ref_m),
            left_upper_arm_len_m_mean=0.0,
            left_upper_arm_len_m_std=0.0,
            left_upper_arm_abs_error_m_mean=0.0,
            left_upper_arm_abs_error_m_p95=0.0,
            left_upper_arm_rel_error_mean_pct=0.0,
        ))

    write_global_summary(out_root, summaries)
    print("videopose3d template project prepared.")
    print("Next steps:")
    print("1) implement scripts/extract_2d_keypoints.py")
    print("2) implement scripts/run_lifting.py")
    print("3) merge outputs into common schema inside scripts/benchmark.py")

if __name__ == "__main__":
    main()
