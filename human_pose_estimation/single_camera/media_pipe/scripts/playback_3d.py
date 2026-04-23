#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/playback_3d.py

Запуск:
  python3 scripts/playback_3d.py

Что делает:
- Позволяет выбрать, с какой папкой результатов работать (например: runs или run_opt)
- Ищет внутри выбранной папки все файлы вида **/landmarks_3d_world*.csv
  (поддерживает landmarks_3d_world.csv и landmarks_3d_world_torso_arms.csv)
- Показывает интерактивный список и запускает проигрывание 3D позы в matplotlib
- Никаких преобразований осей (raw mp world coords). Опционально центрирование по бёдрам.

Опции:
  --root .          где искать папки результатов
  --default runs    дефолтная папка (если есть)
  --speed 1.0
  --fps 0
  --center hips|none
  --lim auto|fixed
  --start 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}
NUM_LANDMARKS = len(LANDMARK_NAMES)
HIP_IDXS = [LM["left_hip"], LM["right_hip"]]

POSE_CONNECTIONS = [
    (LM["nose"], LM["left_eye_inner"]), (LM["left_eye_inner"], LM["left_eye"]),
    (LM["left_eye"], LM["left_eye_outer"]), (LM["left_eye_outer"], LM["left_ear"]),
    (LM["nose"], LM["right_eye_inner"]), (LM["right_eye_inner"], LM["right_eye"]),
    (LM["right_eye"], LM["right_eye_outer"]), (LM["right_eye_outer"], LM["right_ear"]),
    (LM["mouth_left"], LM["mouth_right"]),
    (LM["left_shoulder"], LM["right_shoulder"]),
    (LM["left_shoulder"], LM["left_hip"]),
    (LM["right_shoulder"], LM["right_hip"]),
    (LM["left_hip"], LM["right_hip"]),
    (LM["left_shoulder"], LM["left_elbow"]),
    (LM["left_elbow"], LM["left_wrist"]),
    (LM["left_wrist"], LM["left_thumb"]),
    (LM["left_wrist"], LM["left_index"]),
    (LM["left_wrist"], LM["left_pinky"]),
    (LM["left_index"], LM["left_pinky"]),
    (LM["right_shoulder"], LM["right_elbow"]),
    (LM["right_elbow"], LM["right_wrist"]),
    (LM["right_wrist"], LM["right_thumb"]),
    (LM["right_wrist"], LM["right_index"]),
    (LM["right_wrist"], LM["right_pinky"]),
    (LM["right_index"], LM["right_pinky"]),
    (LM["left_hip"], LM["left_knee"]),
    (LM["left_knee"], LM["left_ankle"]),
    (LM["left_ankle"], LM["left_heel"]),
    (LM["left_heel"], LM["left_foot_index"]),
    (LM["left_ankle"], LM["left_foot_index"]),
    (LM["right_hip"], LM["right_knee"]),
    (LM["right_knee"], LM["right_ankle"]),
    (LM["right_ankle"], LM["right_heel"]),
    (LM["right_heel"], LM["right_foot_index"]),
    (LM["right_ankle"], LM["right_foot_index"]),
]


def discover_result_dirs(root: Path) -> List[Path]:
    """
    Ищем "папки результатов" в root: те, где есть хотя бы один файл landmarks_3d_world*.csv
    """
    if not root.exists():
        return []
    candidates = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        if list(p.rglob("landmarks_3d_world*.csv")):
            candidates.append(p)
    return candidates


def discover_csv_files(results_dir: Path) -> List[Path]:
    """
    Ищем любые world csv:
      landmarks_3d_world.csv
      landmarks_3d_world_torso_arms.csv
      landmarks_3d_world_*.csv
    """
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("landmarks_3d_world*.csv"))


def choose_dir_interactively(dirs: List[Path], root: Path, default_name: str) -> Path:
    if not dirs:
        # если ничего не нашли автоматически — дадим выбрать вручную вводом пути
        print(f"\nNo result folders with landmarks_3d_world*.csv found under: {root}")
        while True:
            s = input("Enter path to results folder (or 'q' to quit): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                sys.exit(0)
            p = Path(s).expanduser()
            if not p.is_absolute():
                p = (root / p).resolve()
            if p.exists() and p.is_dir() and list(p.rglob("landmarks_3d_world*.csv")):
                return p
            print("That folder doesn't contain landmarks_3d_world*.csv. Try again.")

    # если есть default — поднимем его вверх списка
    dirs_sorted = dirs[:]
    for i, d in enumerate(dirs_sorted):
        if d.name == default_name:
            dirs_sorted.insert(0, dirs_sorted.pop(i))
            break

    print("\nSelect results folder:\n")
    for i, d in enumerate(dirs_sorted, start=1):
        rel = d.relative_to(root) if d.is_relative_to(root) else d
        mark = " (default)" if d.name == default_name else ""
        print(f"{i:2d}) {rel}{mark}")
    print()

    while True:
        s = input(f"Select folder [1..{len(dirs_sorted)}] (or 'q' to quit): ").strip().lower()
        if s in ("q", "quit", "exit"):
            sys.exit(0)
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(dirs_sorted):
                return dirs_sorted[idx - 1]
        print("Invalid input. Try again.")


def choose_file_interactively(files: List[Path], base_dir: Path) -> Path:
    if not files:
        raise RuntimeError(
            f"No landmarks_3d_world*.csv found under: {base_dir}\n"
            f"Expected something like: <results>/<video>/<model>/<mode>/<delegate>/landmarks_3d_world*.csv"
        )

    print("\nFound 3D landmark files:\n")
    for i, p in enumerate(files, start=1):
        rel = p.relative_to(base_dir)
        print(f"{i:2d}) {rel}")
    print()

    while True:
        s = input(f"Select file [1..{len(files)}] (or 'q' to quit): ").strip().lower()
        if s in ("q", "quit", "exit"):
            sys.exit(0)
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        print("Invalid input. Try again.")


def load_world_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поддерживает два формата колонок:
    1) Полный:
       frame_idx,timestamp_ms,landmark_idx,landmark_name,x_m,y_m,z_m,visibility,presence
    2) Любой, где есть frame_idx,timestamp_ms,landmark_idx и (x_m,y_m,z_m) либо (x,y,z)

    Возвращает:
      ts_ms: (T,) int64
      xyz:   (T, 33, 3) float64
    """
    by_frame: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    ts_by_frame: Dict[int, int] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])

        xk = "x_m" if "x_m" in cols else "x"
        yk = "y_m" if "y_m" in cols else "y"
        zk = "z_m" if "z_m" in cols else "z"

        required = {"frame_idx", "timestamp_ms", "landmark_idx", xk, yk, zk}
        missing = required - cols
        if missing:
            raise RuntimeError(f"CSV {path} missing columns: {sorted(missing)}")

        for row in r:
            fi = int(row["frame_idx"])
            ts = int(row["timestamp_ms"])
            li = int(row["landmark_idx"])
            x = float(row[xk])
            y = float(row[yk])
            z = float(row[zk])
            ts_by_frame[fi] = ts
            by_frame.setdefault(fi, {})[li] = (x, y, z)

    frame_ids = sorted(by_frame.keys())
    if not frame_ids:
        raise RuntimeError(f"No frames found in {path}")

    ts_ms = np.array([ts_by_frame[fi] for fi in frame_ids], dtype=np.int64)
    xyz = np.full((len(frame_ids), NUM_LANDMARKS, 3), np.nan, dtype=np.float64)

    for t, fi in enumerate(frame_ids):
        for li, (x, y, z) in by_frame[fi].items():
            if 0 <= li < NUM_LANDMARKS:
                xyz[t, li] = (x, y, z)

    return ts_ms, xyz


def compute_limits(xyz: np.ndarray, mode: str):
    if mode == "fixed":
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    pts = xyz.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(1e-6, maxs - mins)
    pad = 0.10 * span
    mins -= pad
    maxs += pad
    return (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])


def is_mostly_nan(frame_pts: np.ndarray, max_nan_frac: float = 0.8) -> bool:
    valid = np.isfinite(frame_pts).all(axis=1)
    nan_frac = 1.0 - (np.count_nonzero(valid) / max(1, len(valid)))
    return nan_frac > max_nan_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Where to search results folders")
    ap.add_argument("--default", type=str, default="runs", help="Default results folder name")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=0.0, help="0=use timestamp_ms")
    ap.add_argument("--center", choices=["hips", "none"], default="hips")
    ap.add_argument("--lim", choices=["auto", "fixed"], default="auto")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    dirs = discover_result_dirs(root)
    results_dir = choose_dir_interactively(dirs, root, args.default)

    files = discover_csv_files(results_dir)
    chosen = choose_file_interactively(files, results_dir)

    print(f"\nResults folder: {results_dir}")
    print(f"Selected file : {chosen}\n")

    ts_ms, xyz = load_world_csv(chosen)

    # Only translation (centering), NO axis transforms
    if args.center == "hips":
        hips_center = np.nanmean(xyz[:, HIP_IDXS, :], axis=1)
        xyz = xyz - hips_center[:, None, :]

    xlim, ylim, zlim = compute_limits(xyz, args.lim)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"3D Pose playback — {chosen.parent.name} (raw mp world coords)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    scat = ax.scatter([], [], [], s=10)

    lines: List[Tuple[any, int, int]] = []
    for a, b in POSE_CONNECTIONS:
        ln, = ax.plot([], [], [], linewidth=2)
        lines.append((ln, a, b))

    info = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")

    # timing
    if args.fps and args.fps > 0:
        frame_delays = np.full((len(ts_ms),), 1.0 / args.fps, dtype=np.float64)
    else:
        dts = np.diff(ts_ms.astype(np.float64)) / 1000.0
        if len(dts) == 0:
            dts = np.array([1/30], dtype=np.float64)
        frame_delays = np.concatenate([dts, [dts[-1]]], axis=0)

    frame_delays = frame_delays / max(1e-6, args.speed)

    start = max(0, min(args.start, len(ts_ms) - 1))
    frame_indices = list(range(start, len(ts_ms)))

    # keep animation alive
    t_wall_prev: Optional[float] = None

    def update(k: int):
        nonlocal t_wall_prev
        i = frame_indices[k]

        if t_wall_prev is None:
            t_wall_prev = time.perf_counter()

        target = frame_delays[i]
        now = time.perf_counter()
        elapsed = now - t_wall_prev
        if elapsed < target:
            time.sleep(target - elapsed)
        t_wall_prev = time.perf_counter()

        pts = xyz[i]

        if is_mostly_nan(pts):
            scat._offsets3d = ([], [], [])
            for (ln, _, _) in lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
            info.set_text(f"frame {i+1}/{len(ts_ms)} | t={ts_ms[i]} ms | NO POSE")
            return [scat, info] + [ln for (ln, _, _) in lines]

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for (ln, a, b) in lines:
            pa = pts[a]; pb = pts[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                ln.set_3d_properties([pa[2], pb[2]])
            else:
                ln.set_data([], [])
                ln.set_3d_properties([])

        info.set_text(f"frame {i+1}/{len(ts_ms)} | t={ts_ms[i]} ms | speed={args.speed}x")
        return [scat, info] + [ln for (ln, _, _) in lines]

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=10, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()