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

def short_video_name(video_name: str) -> str:
    return Path(video_name).stem

def short_label(video_name: str, model_name: str) -> str:
    return f"{short_video_name(video_name)}|{model_name}"

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

def save_bar(labels, values, title, ylabel, out_path):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def main():
    project_root = Path(__file__).resolve().parents[1]
    run_root = project_root / "runs" / "run"
    rows = load_summary_csv(run_root / "summary.csv")
    if not rows:
        raise RuntimeError(f"No summary.csv found in {run_root}")

    out_dir = project_root / "runs" / "compare_out"
    safe_mkdir(out_dir)

    labels = [short_label(r["video_or_session"], r["model"]) for r in rows]
    save_bar(labels, [to_float(r["effective_fps"]) for r in rows], "Mean FPS by each video/model", "FPS", out_dir / "fps_by_video_and_model.png")
    save_bar(labels, [to_float(r["mean_presence"]) for r in rows], "Mean presence by each video/model", "Presence", out_dir / "presence_by_video_model.png")
    save_bar(labels, [to_float(r["jitter_world_m_mean"]) for r in rows], "Jitter by each video/model", "Jitter", out_dir / "jitter_by_video_model.png")
    print(f"Graphs saved to {out_dir}")

if __name__ == "__main__":
    main()
