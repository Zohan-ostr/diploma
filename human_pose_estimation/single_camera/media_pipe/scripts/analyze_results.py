#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

def safe_mkdir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def short_model_name(model_name: str) -> str: return model_name.replace('pose_landmarker_','').replace('pose_landmarker','').strip('_')
def short_video_name(video_name: str) -> str: return Path(video_name).stem
def short_label(video_name: str, model_name: str) -> str: return f"{short_video_name(video_name)}|{short_model_name(model_name)}"
def load_summary_csv(path: Path) -> List[dict]:
    if not path.exists(): return []
    with path.open('r', newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def to_float(x, default=np.nan) -> float:
    try:
        if x is None or x == '': return float(default)
        return float(x)
    except Exception: return float(default)
def save_bar(labels, values, title, ylabel, out_path):
    fig = plt.figure(figsize=(12,6)); ax = fig.add_subplot(111)
    ax.bar(labels, values); ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis='x', rotation=20)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)
def save_scatter(x,y,labels,title,xlabel,ylabel,out_path):
    if not x or not y: return
    fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111)
    ax.scatter(x,y); ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for xi,yi,lab in zip(x,y,labels): ax.annotate(lab,(xi,yi), fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)
def grouped_mean(rows: List[dict], key_x: str, key_y: str):
    groups: Dict[str, List[float]] = {}
    for r in rows:
        val = to_float(r.get(key_y))
        if np.isfinite(val): groups.setdefault(r[key_x], []).append(val)
    labels = sorted(groups.keys()); values = [float(np.mean(groups[k])) for k in labels]
    return labels, values
def make_graphs(run_root: Path):
    rows = load_summary_csv(run_root / 'summary.csv')
    if not rows: return
    out_dir = run_root.parent / 'compare_out' / run_root.name; safe_mkdir(out_dir)
    labels = [short_label(r['video_or_session'], r['model']) for r in rows]
    save_bar(labels, [to_float(r['effective_fps']) for r in rows], 'Mean FPS by each video/model', 'FPS', out_dir / 'fps_by_video_and_model.png')
    save_bar(labels, [to_float(r['mean_visibility']) for r in rows], 'Mean visibility by each video/model', 'Visibility', out_dir / 'visibility_by_video_model.png')
    save_bar(labels, [to_float(r['mean_presence']) for r in rows], 'Mean presence by each video/model', 'Presence', out_dir / 'presence_by_video_model.png')
    for metric, title, filename, ylabel in [
        ('effective_fps', 'Mean FPS by model', 'fps_vs_model.png', 'FPS'),
        ('mean_visibility', 'Visibility vs model', 'visibility_vs_model.png', 'Visibility'),
        ('mean_presence', 'Presence vs model', 'presence_vs_model.png', 'Presence'),
        ('jitter_world_m_mean', 'Jitter vs model', 'jitter_vs_model.png', 'Jitter'),
        ('left_upper_arm_abs_error_m_mean', 'Arm abs error vs model', 'arm_abs_error_vs_model.png', 'Abs error (m)'),
        ('left_upper_arm_rel_error_mean_pct', 'Arm relative error vs model', 'arm_rel_error_vs_model.png', 'Relative error (%)')]:
        lm, vm = grouped_mean(rows, 'model', metric); lm = [short_model_name(x) for x in lm]; save_bar(lm, vm, title, ylabel, out_dir / filename)
    for metric, title, filename, ylabel in [
        ('effective_fps', 'Mean FPS by video', 'fps_vs_video.png', 'FPS'),
        ('mean_visibility', 'Visibility vs video', 'visibility_vs_video.png', 'Visibility'),
        ('mean_presence', 'Presence vs video', 'presence_vs_video.png', 'Presence'),
        ('jitter_world_m_mean', 'Jitter vs video', 'jitter_vs_video.png', 'Jitter'),
        ('left_upper_arm_abs_error_m_mean', 'Arm abs error vs video', 'arm_abs_error_vs_video.png', 'Abs error (m)')]:
        lv, vv = grouped_mean(rows, 'video_or_session', metric); lv = [short_video_name(x) for x in lv]; save_bar(lv, vv, title, ylabel, out_dir / filename)
    valid = [r for r in rows if np.isfinite(to_float(r['mean_presence'])) and np.isfinite(to_float(r['jitter_world_m_mean']))]
    save_scatter([to_float(r['mean_presence']) for r in valid],[to_float(r['jitter_world_m_mean']) for r in valid],[short_label(r['video_or_session'], r['model']) for r in valid],'Jitter vs mean presence','Mean presence','Jitter', out_dir / 'jitter_vs_mean_presence.png')
    valid2 = [r for r in rows if np.isfinite(to_float(r['left_upper_arm_abs_error_m_mean'])) and np.isfinite(to_float(r['jitter_world_m_mean']))]
    save_scatter([to_float(r['left_upper_arm_abs_error_m_mean']) for r in valid2],[to_float(r['jitter_world_m_mean']) for r in valid2],[short_label(r['video_or_session'], r['model']) for r in valid2],'Jitter vs arm length error','Arm abs error (m)','Jitter', out_dir / 'jitter_vs_arm_error.png')
    valid3 = [r for r in rows if np.isfinite(to_float(r['left_upper_arm_abs_error_m_mean'])) and np.isfinite(to_float(r['mean_presence']))]
    save_scatter([to_float(r['mean_presence']) for r in valid3],[to_float(r['left_upper_arm_abs_error_m_mean']) for r in valid3],[short_label(r['video_or_session'], r['model']) for r in valid3],'Presence vs arm length error','Mean presence','Arm abs error (m)', out_dir / 'presence_vs_arm_error.png')
    print(f'Graphs saved to {out_dir}')
def main():
    project_root = Path(__file__).resolve().parents[1]
    for run_name in ['run', 'run_opt']:
        make_graphs(project_root / 'runs' / run_name)
if __name__ == '__main__':
    main()
