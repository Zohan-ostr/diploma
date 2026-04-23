#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/analyze_results.py

Анализ результатов MediaPipe Pose Landmarker (runs/run_opt и т.п.)

Возможности:
- Сканирует корень с результатами, находит:
  - summary.csv / summary.json (глобальные)
  - summary.json в листовых папках прогонов
  - frame_metrics.csv
  - landmarks_3d_world*.csv
- Собирает общий датасет метрик и строит графики эффективности.
- Считает "дрожание" суставов (variance/std по 3D координатам + jitter/velocity/jerk).

Запуск:
  python3 scripts/analyze_results.py --root . --out_dir analysis_out

Типичный пример:
  python3 scripts/analyze_results.py --root /home/zohan/diploma/media_pipe --out_dir analysis_out

Параметры:
  --root         где искать папки results (runs, run_opt и т.п.)
  --out_dir      куда сохранять CSV/PNG
  --min_frames   минимальное число кадров для расчётов по траекториям
  --use_cache    использовать кэш парсинга (ускоряет повторные запуски)

Выход:
  analysis_out/metrics_all.csv
  analysis_out/joint_jitter_all.csv
  analysis_out/plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------- Pose landmark constants (33) --------
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
NUM_LM = len(LANDMARK_NAMES)


# ---------- Small utils ----------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha1_of_path(path: Path) -> str:
    h = hashlib.sha1(str(path).encode("utf-8"))
    return h.hexdigest()[:12]

def percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


# ---------- Discovery ----------
def discover_result_roots(root: Path) -> List[Path]:
    """
    Возвращает папки верхнего уровня, которые выглядят как результатные:
    - содержат summary.csv/summary.json
    - или содержат frame_metrics.csv/landmarks_3d_world*.csv глубже
    """
    candidates = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        has_summary = (d / "summary.csv").exists() or (d / "summary.json").exists()
        has_leaf = bool(list(d.rglob("frame_metrics.csv"))) or bool(list(d.rglob("landmarks_3d_world*.csv")))
        if has_summary or has_leaf:
            candidates.append(d)
    return candidates

def discover_leaf_runs(result_root: Path) -> List[Path]:
    """
    Листовые папки прогона: где есть frame_metrics.csv ИЛИ summary.json
    """
    leaf = set()
    for p in result_root.rglob("frame_metrics.csv"):
        leaf.add(p.parent)
    for p in result_root.rglob("summary.json"):
        leaf.add(p.parent)
    return sorted(leaf)

def find_file_upwards(start: Path, filename: str, stop: Path) -> Optional[Path]:
    """
    Поднимаемся вверх от start до stop (включительно), ищем filename.
    """
    cur = start
    stop = stop.resolve()
    while True:
        cand = cur / filename
        if cand.exists():
            return cand
        if cur.resolve() == stop:
            return None
        if cur.parent == cur:
            return None
        cur = cur.parent


# ---------- Parsers ----------
def parse_summary_any(path: Path) -> List[dict]:
    """
    Читает summary.json:
      - либо листовой (один объект)
      - либо глобальный (список объектов)
    """
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []

def parse_summary_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]

def parse_frame_metrics_csv(path: Path) -> dict:
    """
    Возвращает агрегаты из frame_metrics.csv (если есть):
      - has_pose_rate
      - infer_ms mean/p50/p95
      - occluded mean (если колонка есть)
      - wrist_speed mean/p95 (если есть)
    """
    infer = []
    has_pose = []
    occl = []
    wrist = []

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        for row in r:
            if "inference_ms" in cols:
                try: infer.append(float(row["inference_ms"]))
                except: pass
            if "has_pose" in cols:
                try: has_pose.append(int(row["has_pose"]))
                except: pass
            # разные варианты названий
            for k in ("occluded_frac_keep", "occluded_frac_keep_mean", "occluded_frac_keep"):
                if k in cols:
                    try: occl.append(float(row[k]))
                    except: pass
                    break
            for k in ("wrist_speed_m_s", "wrist_speed_m_s_mean"):
                if k in cols:
                    try: wrist.append(float(row[k]))
                    except: pass
                    break

    infer_np = np.array(infer, dtype=np.float64)
    has_pose_np = np.array(has_pose, dtype=np.int32)
    occl_np = np.array(occl, dtype=np.float64)
    wrist_np = np.array(wrist, dtype=np.float64)

    return {
        "frame_count": int(len(has_pose_np) if has_pose_np.size else (infer_np.size if infer_np.size else 0)),
        "has_pose_rate": float(has_pose_np.mean()) if has_pose_np.size else None,
        "infer_ms_mean": float(infer_np.mean()) if infer_np.size else None,
        "infer_ms_p50": percentile(infer_np, 50) if infer_np.size else None,
        "infer_ms_p95": percentile(infer_np, 95) if infer_np.size else None,
        "occluded_mean": float(occl_np.mean()) if occl_np.size else None,
        "wrist_speed_mean": float(wrist_np.mean()) if wrist_np.size else None,
        "wrist_speed_p95": percentile(wrist_np, 95) if wrist_np.size else None,
    }

def parse_landmarks_3d_world_csv(path: Path, min_frames: int = 50) -> Optional[dict]:
    """
    Считает "дрожание" суставов по world coords.

    Метрики на сустав:
      - std_x, std_y, std_z (м)
      - std_norm: std от нормы позиции относительно среднего (м)
      - jitter_rmssd: RMSSD по перемещениям |p_t - p_{t-1}| (м)
      - vel_mean: средняя скорость (м/с) по соседним кадрам
      - jerk_rmssd: RMSSD по jerk (м/с^3) по 2-й разности скорости

    Возвращает dict:
      { "frames": T, "per_joint": { idx: {...} }, "global": {...} }
    """
    # Поддержка колонок x_m/y_m/z_m или x/y/z
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        xk = "x_m" if "x_m" in cols else ("x" if "x" in cols else None)
        yk = "y_m" if "y_m" in cols else ("y" if "y" in cols else None)
        zk = "z_m" if "z_m" in cols else ("z" if "z" in cols else None)
        if not all([xk, yk, zk]) or "frame_idx" not in cols or "timestamp_ms" not in cols or "landmark_idx" not in cols:
            return None

        by_frame: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
        ts_by_frame: Dict[int, int] = {}

        for row in r:
            try:
                fi = int(row["frame_idx"])
                ts = int(row["timestamp_ms"])
                li = int(row["landmark_idx"])
                x = float(row[xk]); y = float(row[yk]); z = float(row[zk])
            except:
                continue
            ts_by_frame[fi] = ts
            by_frame.setdefault(fi, {})[li] = (x, y, z)

    frame_ids = sorted(by_frame.keys())
    if len(frame_ids) < min_frames:
        return None

    T = len(frame_ids)
    ts_ms = np.array([ts_by_frame[fi] for fi in frame_ids], dtype=np.float64)
    # dt между кадрами
    dts = np.diff(ts_ms) / 1000.0
    if dts.size == 0:
        return None
    dt = np.median(dts[dts > 0]) if np.any(dts > 0) else 1/30

    xyz = np.full((T, NUM_LM, 3), np.nan, dtype=np.float64)
    for t, fi in enumerate(frame_ids):
        for li, (x, y, z) in by_frame[fi].items():
            if 0 <= li < NUM_LM:
                xyz[t, li] = (x, y, z)

    per_joint = {}
    std_norms = []
    jitter_rmssds = []
    vel_means = []
    jerk_rmssds = []

    for j in range(NUM_LM):
        pts = xyz[:, j, :]  # (T,3)
        valid = np.isfinite(pts).all(axis=1)
        if valid.sum() < min_frames:
            continue

        p = pts[valid]
        # std по координатам
        std_xyz = p.std(axis=0)

        # std от нормы относительно среднего (чтобы измерять "разброс" вокруг среднего положения)
        p0 = p - p.mean(axis=0, keepdims=True)
        r = np.linalg.norm(p0, axis=1)
        std_norm = float(r.std())

        # jitter RMSSD по перемещениям
        dp = np.diff(p, axis=0)
        step = np.linalg.norm(dp, axis=1)
        jitter_rmssd = float(np.sqrt(np.mean(step**2)))

        # скорость
        v = step / dt  # (N-1,)
        vel_mean = float(np.mean(v))

        # jerk (2-я разность скорости)
        # j ≈ (v_{t+1} - v_t) / dt
        if v.size >= 2:
            a = np.diff(v) / dt  # "ускорение" (N-2,)
            if a.size >= 2:
                jerk = np.diff(a) / dt  # (N-3,)
                jerk_rmssd = float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0
            else:
                jerk_rmssd = 0.0
        else:
            jerk_rmssd = 0.0

        per_joint[j] = {
            "name": LANDMARK_NAMES[j],
            "valid_frames": int(valid.sum()),
            "std_x": float(std_xyz[0]),
            "std_y": float(std_xyz[1]),
            "std_z": float(std_xyz[2]),
            "std_norm": float(std_norm),
            "jitter_rmssd": float(jitter_rmssd),
            "vel_mean": float(vel_mean),
            "jerk_rmssd": float(jerk_rmssd),
        }

        std_norms.append(std_norm)
        jitter_rmssds.append(jitter_rmssd)
        vel_means.append(vel_mean)
        jerk_rmssds.append(jerk_rmssd)

    if not per_joint:
        return None

    global_stats = {
        "frames": int(T),
        "dt_median_s": float(dt),
        "std_norm_mean": float(np.mean(std_norms)) if std_norms else 0.0,
        "jitter_rmssd_mean": float(np.mean(jitter_rmssds)) if jitter_rmssds else 0.0,
        "vel_mean_mean": float(np.mean(vel_means)) if vel_means else 0.0,
        "jerk_rmssd_mean": float(np.mean(jerk_rmssds)) if jerk_rmssds else 0.0,
    }

    return {"frames": int(T), "per_joint": per_joint, "global": global_stats}


# ---------- Aggregation ----------
def normalize_run_tags(leaf_dir: Path, result_root: Path) -> Dict[str, str]:
    """
    Пытаемся восстановить video/model/mode/delegate из пути:
      <root>/<video>/<model>/<mode>/<delegate>/...
    Но если структура другая — оставим что получится.
    """
    rel = leaf_dir.relative_to(result_root)
    parts = rel.parts

    tag = {
        "result_root": result_root.name,
        "leaf_rel": str(rel),
        "video": "",
        "model": "",
        "mode": "",
        "delegate": "",
    }
    if len(parts) >= 1: tag["video"] = parts[0]
    if len(parts) >= 2: tag["model"] = parts[1]
    if len(parts) >= 3: tag["mode"] = parts[2]
    if len(parts) >= 4: tag["delegate"] = parts[3]
    return tag


def collect_all_metrics(root: Path,
                        out_dir: Path,
                        min_frames: int,
                        use_cache: bool) -> Tuple[List[dict], List[dict]]:
    """
    Возвращает:
      metrics_rows: список dict по каждому leaf run
      jitter_rows : список dict по каждому joint в каждом leaf run (если 3D csv найден)
    """
    cache_dir = out_dir / "_cache"
    safe_mkdir(cache_dir)

    metrics_rows: List[dict] = []
    jitter_rows: List[dict] = []

    result_roots = discover_result_roots(root)
    if not result_roots:
        raise RuntimeError(f"No result folders found under: {root}")

    for rr in result_roots:
        leaf_runs = discover_leaf_runs(rr)
        for leaf in leaf_runs:
            tags = normalize_run_tags(leaf, rr)

            # 1) summary.json рядом (если есть)
            leaf_summary_path = leaf / "summary.json"
            leaf_summary = None
            if leaf_summary_path.exists():
                leaf_summary = parse_summary_any(leaf_summary_path)[0]

            # 2) frame_metrics.csv (если есть)
            frame_path = leaf / "frame_metrics.csv"
            frame_aggr = parse_frame_metrics_csv(frame_path) if frame_path.exists() else {}

            # 3) 3D landmarks world csv (если есть)
            lm3d_files = sorted(list(leaf.glob("landmarks_3d_world*.csv")))
            lm3d_path = lm3d_files[0] if lm3d_files else None

            jitter_aggr = None
            if lm3d_path is not None:
                cache_key = sha1_of_path(lm3d_path) + "_" + str(lm3d_path.stat().st_mtime_ns)
                cache_file = cache_dir / f"jitter_{cache_key}.json"
                if use_cache and cache_file.exists():
                    jitter_aggr = read_json(cache_file)
                else:
                    jitter_aggr = parse_landmarks_3d_world_csv(lm3d_path, min_frames=min_frames)
                    if use_cache and jitter_aggr is not None:
                        write_json(cache_file, jitter_aggr)

            # Собираем строку метрик
            row = {
                **tags,
                "leaf_dir": str(leaf),
                "frame_metrics_path": str(frame_path) if frame_path.exists() else "",
                "landmarks_3d_path": str(lm3d_path) if lm3d_path else "",
            }

            # Вытаскиваем ключевые summary поля, если есть
            if isinstance(leaf_summary, dict):
                for k, v in leaf_summary.items():
                    # не тащим огромные поля
                    if k in ("per_joint",):
                        continue
                    row[f"summary_{k}"] = v

            # frame агрегаты
            for k, v in frame_aggr.items():
                row[f"frame_{k}"] = v

            # global jitter агрегаты
            if jitter_aggr and "global" in jitter_aggr:
                for k, v in jitter_aggr["global"].items():
                    row[f"jitter_{k}"] = v

            metrics_rows.append(row)

            # per joint jitter rows
            if jitter_aggr and "per_joint" in jitter_aggr:
                for j_str, jv in jitter_aggr["per_joint"].items():
                    # j_str может быть int (если из python) или str (если из json cache)
                    try:
                        j = int(j_str)
                    except:
                        j = -1
                    jitter_rows.append({
                        **tags,
                        "leaf_dir": str(leaf),
                        "joint_idx": j,
                        **jv
                    })

    return metrics_rows, jitter_rows


# ---------- Saving ----------
def write_csv_from_dicts(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    # стабильный порядок колонок
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ---------- Plotting ----------
def _group_key(row: dict) -> Tuple[str, str, str, str]:
    return (row.get("result_root",""), row.get("video",""), row.get("model",""), row.get("delegate",""))

def to_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except:
        return None

def plot_efficiency(metrics_rows: List[dict], out_dir: Path) -> None:
    """
    Делает набор графиков по summary_*/frame_* полям.
    """
    plots_dir = out_dir / "plots"
    safe_mkdir(plots_dir)

    # вытащим удобные численные поля
    data = []
    for r in metrics_rows:
        eff = to_float(r.get("summary_effective_fps")) or to_float(r.get("summary_effective_fps".lower())) or to_float(r.get("summary_effective_fps", None))
        if eff is None:
            eff = to_float(r.get("summary_effective_fps", None))
        data.append({
            "key": _group_key(r),
            "result_root": r.get("result_root",""),
            "video": r.get("video",""),
            "model": r.get("model",""),
            "delegate": r.get("delegate",""),
            "effective_fps": to_float(r.get("summary_effective_fps")) or to_float(r.get("summary_effective_fps", None)) or to_float(r.get("summary_effective_fps")),
            "detection_rate": to_float(r.get("summary_detection_rate")) or to_float(r.get("frame_has_pose_rate")),
            "infer_ms_mean": to_float(r.get("summary_mean_inference_ms")) or to_float(r.get("frame_infer_ms_mean")),
            "infer_ms_p95": to_float(r.get("summary_p95_inference_ms")) or to_float(r.get("frame_infer_ms_p95")),
            "tracking_loss_events": to_float(r.get("summary_tracking_loss_events")),
            "longest_loss_streak": to_float(r.get("summary_longest_loss_streak_frames")),
            "jitter_mean": to_float(r.get("summary_jitter_world_m_mean")) or to_float(r.get("jitter_jitter_rmssd_mean")),
            "wrist_speed_p95": to_float(r.get("summary_wrist_speed_m_s_p95")) or to_float(r.get("frame_wrist_speed_p95")),
        })

    # фильтруем только те, где есть хотя бы что-то
    data = [d for d in data if (d["detection_rate"] is not None or d["infer_ms_mean"] is not None)]
    if not data:
        return

    # 1) latency vs detection scatter
    xs = [d["infer_ms_p95"] for d in data if d["infer_ms_p95"] is not None and d["detection_rate"] is not None]
    ys = [d["detection_rate"] for d in data if d["infer_ms_p95"] is not None and d["detection_rate"] is not None]
    labels = [f'{d["video"]}|{d["model"]}' for d in data if d["infer_ms_p95"] is not None and d["detection_rate"] is not None]

    if xs and ys:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys)
        ax.set_xlabel("p95 inference (ms)")
        ax.set_ylabel("detection rate")
        ax.set_title("Latency vs Detection rate")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "latency_vs_detection.png", dpi=160)
        plt.close(fig)

    # 2) effective fps bar (grouped by model, averaged over videos)
    #    сделаем простой бар: модели по X, значение = среднее effective_fps
    by_model: Dict[str, List[float]] = {}
    for d in data:
        v = d["effective_fps"]
        if v is None:
            continue
        by_model.setdefault(d["model"], []).append(v)

    if by_model:
        models = sorted(by_model.keys())
        means = [float(np.mean(by_model[m])) for m in models]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(models, means)
        ax.set_ylabel("effective FPS (mean)")
        ax.set_title("Effective FPS by model (mean over runs)")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "effective_fps_by_model.png", dpi=160)
        plt.close(fig)

    # 3) detection rate by video (box-ish via scatter)
    #    точечный график: X=video, Y=detection, разные модели будут точками
    vids = sorted(set(d["video"] for d in data))
    xmap = {v:i for i,v in enumerate(vids)}
    xs = [xmap[d["video"]] for d in data if d["detection_rate"] is not None]
    ys = [d["detection_rate"] for d in data if d["detection_rate"] is not None]
    if xs and ys:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys)
        ax.set_xticks(list(xmap.values()), list(xmap.keys()), rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("detection rate")
        ax.set_title("Detection rate by video (all models)")
        fig.tight_layout()
        fig.savefig(plots_dir / "detection_rate_by_video.png", dpi=160)
        plt.close(fig)

    # 4) jitter mean vs wrist_speed_p95 (динамика/шум)
    xs = [d["wrist_speed_p95"] for d in data if d["wrist_speed_p95"] is not None and d["jitter_mean"] is not None]
    ys = [d["jitter_mean"] for d in data if d["wrist_speed_p95"] is not None and d["jitter_mean"] is not None]
    labels = [f'{d["video"]}|{d["model"]}' for d in data if d["wrist_speed_p95"] is not None and d["jitter_mean"] is not None]
    if xs and ys:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xs, ys)
        ax.set_xlabel("wrist speed p95 (m/s)")
        ax.set_ylabel("jitter (m) (mean)")
        ax.set_title("Motion vs Jitter")
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "motion_vs_jitter.png", dpi=160)
        plt.close(fig)


def plot_joint_jitter_heatmap(jitter_rows: List[dict], out_dir: Path) -> None:
    """
    Делает heatmap: суставы x модели (средний jitter_rmssd).
    """
    plots_dir = out_dir / "plots"
    safe_mkdir(plots_dir)

    if not jitter_rows:
        return

    # агрегируем jitter_rmssd по (model, joint)
    agg: Dict[Tuple[str,int], List[float]] = {}
    for r in jitter_rows:
        m = r.get("model","")
        j = int(r.get("joint_idx", -1))
        v = r.get("jitter_rmssd", None)
        try:
            v = float(v)
        except:
            continue
        if j < 0:
            continue
        agg.setdefault((m, j), []).append(v)

    models = sorted(set(m for (m, _) in agg.keys()))
    joints = list(range(NUM_LM))

    if not models:
        return

    mat = np.full((len(joints), len(models)), np.nan, dtype=np.float64)
    for mi, m in enumerate(models):
        for ji, j in enumerate(joints):
            vals = agg.get((m, j), [])
            if vals:
                mat[ji, mi] = float(np.mean(vals))

    fig = plt.figure(figsize=(max(6, len(models) * 1.2), 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")  # default colormap
    ax.set_title("Joint jitter RMSSD (m): joints x models (mean over runs)")
    ax.set_xlabel("model")
    ax.set_ylabel("joint")
    ax.set_xticks(range(len(models)), models, rotation=20)
    ax.set_yticks(range(NUM_LM), LANDMARK_NAMES)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "joint_jitter_heatmap.png", dpi=160)
    plt.close(fig)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Where to scan for result folders")
    ap.add_argument("--out_dir", type=str, default="analysis_out", help="Output dir for analysis artifacts")
    ap.add_argument("--min_frames", type=int, default=50, help="Min valid frames for 3D jitter stats")
    ap.add_argument("--use_cache", action="store_true", help="Cache parsed jitter results (faster reruns)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    safe_mkdir(out_dir)

    metrics_rows, jitter_rows = collect_all_metrics(
        root=root,
        out_dir=out_dir,
        min_frames=int(args.min_frames),
        use_cache=bool(args.use_cache),
    )

    # Save tables
    metrics_csv = out_dir / "metrics_all.csv"
    jitter_csv = out_dir / "joint_jitter_all.csv"
    write_csv_from_dicts(metrics_csv, metrics_rows)
    write_csv_from_dicts(jitter_csv, jitter_rows)

    # Plots
    plot_efficiency(metrics_rows, out_dir)
    plot_joint_jitter_heatmap(jitter_rows, out_dir)

    print("\nDONE.")
    print(f"- Metrics table: {metrics_csv}")
    print(f"- Joint jitter table: {jitter_csv}")
    print(f"- Plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()