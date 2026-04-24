#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_summary_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def to_float(x, default=np.nan) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def save_grouped_bar(group_labels, series_names, series_values, title, ylabel, out_path):
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)
    x = np.arange(len(group_labels), dtype=float)
    width = 0.8 / max(1, len(series_names))
    for i, name in enumerate(series_names):
        vals = series_values[i]
        ax.bar(x + (i - (len(series_names)-1)/2) * width, vals, width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, group_labels, rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def main():
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "common" / "compare" / "compare_out"
    safe_mkdir(out_dir)

    method_roots = [
        project_root / "single_camera" / "media_pipe" / "runs" / "run",
        project_root / "single_camera" / "mmpose_3d" / "runs" / "run",
        project_root / "single_camera" / "videopose3d" / "runs" / "run",
        project_root / "double_cameras" / "openpose_3d" / "runs" / "run",
        project_root / "double_cameras" / "stereo_triangulation" / "runs" / "run",
        project_root / "double_cameras" / "mmpose_voxelpose" / "runs" / "run",
    ]

    rows = []
    for root in method_roots:
        rows.extend(load_summary_csv(root / "summary.csv"))

    if not rows:
        raise RuntimeError("No summary.csv files found across methods.")

    methods = sorted(set(r["method"] for r in rows))
    metrics = [
        ("effective_fps", "Compare methods: FPS", "FPS", "compare_fps.png"),
        ("mean_presence", "Compare methods: Presence", "Presence", "compare_presence.png"),
        ("jitter_world_m_mean", "Compare methods: Jitter", "Jitter", "compare_jitter.png"),
        ("left_upper_arm_abs_error_m_mean", "Compare methods: Arm abs error", "Abs error (m)", "compare_arm_abs_error.png"),
    ]

    for metric, title, ylabel, filename in metrics:
        values = []
        for method in methods:
            vals = [to_float(r.get(metric)) for r in rows if r["method"] == method]
            vals = [v for v in vals if np.isfinite(v)]
            values.append([float(np.mean(vals)) if vals else 0.0])
        save_grouped_bar(["all_runs_mean"], methods, values, title, ylabel, out_dir / filename)

    print(f"Saved common comparison plots to {out_dir}")

if __name__ == "__main__":
    main()
