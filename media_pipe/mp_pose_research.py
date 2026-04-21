#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mp_pose_research.py — исследовательский/бенчмарк-инструмент для MediaPipe Pose Landmarker.

Что умеет:
- Скачать официальные модели pose_landmarker_{lite,full,heavy}.task (опционально)
- Прогнать на видео в режимах IMAGE и VIDEO (сравнить качество/стабильность/скорость)
- Прогнать с веб-камеры (LIVE_STREAM)
- Сохранить:
  - frame_metrics.csv (метрики по кадрам)
  - landmarks_2d.csv (Normalized landmarks)
  - landmarks_3d_world.csv (World landmarks)
  - summary.json + summary.csv (агрегаты по видео)
  - annotated.mp4 (оверлей скелета) (опционально)
  - segmentation.mp4 (маска позы) (опционально)
- 3D-визуализация по сохранённым world-landmarks (окно/видео при наличии ffmpeg)

Заметка про "3D": WorldLandmarks — оценка 3D по RGB (не depth-сенсор),
координаты в метрах, origin ~ midpoint hips (как в документации).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp


# ----------------------------
# Константы / Landmark mapping
# ----------------------------

# ----------------------------
# Tasks-only Pose constants
# ----------------------------

# 33 landmark names (MediaPipe Pose Landmarker)
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
NUM_LANDMARKS = len(LANDMARK_NAMES)

# Индексы (чтобы не зависеть от solutions.pose.PoseLandmark)
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}

ARM_IDXS = [
    LM["left_shoulder"], LM["right_shoulder"],
    LM["left_elbow"], LM["right_elbow"],
    LM["left_wrist"], LM["right_wrist"],
]
WRIST_IDXS = [LM["left_wrist"], LM["right_wrist"]]
HIP_IDXS = [LM["left_hip"], LM["right_hip"]]

# Скелетные соединения (аналог POSE_CONNECTIONS)
# Это стандартный набор для Pose (33 точки).
POSE_CONNECTIONS = [
    # Face
    (LM["nose"], LM["left_eye_inner"]), (LM["left_eye_inner"], LM["left_eye"]),
    (LM["left_eye"], LM["left_eye_outer"]), (LM["left_eye_outer"], LM["left_ear"]),
    (LM["nose"], LM["right_eye_inner"]), (LM["right_eye_inner"], LM["right_eye"]),
    (LM["right_eye"], LM["right_eye_outer"]), (LM["right_eye_outer"], LM["right_ear"]),
    (LM["mouth_left"], LM["mouth_right"]),

    # Torso
    (LM["left_shoulder"], LM["right_shoulder"]),
    (LM["left_shoulder"], LM["left_hip"]),
    (LM["right_shoulder"], LM["right_hip"]),
    (LM["left_hip"], LM["right_hip"]),

    # Left arm
    (LM["left_shoulder"], LM["left_elbow"]),
    (LM["left_elbow"], LM["left_wrist"]),
    (LM["left_wrist"], LM["left_thumb"]),
    (LM["left_wrist"], LM["left_index"]),
    (LM["left_wrist"], LM["left_pinky"]),
    (LM["left_index"], LM["left_pinky"]),

    # Right arm
    (LM["right_shoulder"], LM["right_elbow"]),
    (LM["right_elbow"], LM["right_wrist"]),
    (LM["right_wrist"], LM["right_thumb"]),
    (LM["right_wrist"], LM["right_index"]),
    (LM["right_wrist"], LM["right_pinky"]),
    (LM["right_index"], LM["right_pinky"]),

    # Left leg
    (LM["left_hip"], LM["left_knee"]),
    (LM["left_knee"], LM["left_ankle"]),
    (LM["left_ankle"], LM["left_heel"]),
    (LM["left_heel"], LM["left_foot_index"]),
    (LM["left_ankle"], LM["left_foot_index"]),

    # Right leg
    (LM["right_hip"], LM["right_knee"]),
    (LM["right_knee"], LM["right_ankle"]),
    (LM["right_ankle"], LM["right_heel"]),
    (LM["right_heel"], LM["right_foot_index"]),
    (LM["right_ankle"], LM["right_foot_index"]),
]


# ----------------------------
# Model URLs (official GCS)
# ----------------------------

OFFICIAL_MODEL_URLS = {
    # сначала пробуем "latest" (как в документации), если не получится — fallback на /1/
    "pose_landmarker_lite": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    ],
    "pose_landmarker_full": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ],
    "pose_landmarker_heavy": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    ],
}


# ----------------------------
# Dataclasses for metrics
# ----------------------------

@dataclass
class VideoRunSummary:
    video: str
    model: str
    running_mode: str  # "IMAGE" or "VIDEO" or "LIVE_STREAM"
    delegate: str      # "CPU" or "GPU"

    frames_total: int
    frames_processed: int
    stride: int
    fps_src: float

    detection_rate: float

    mean_inference_ms: float
    p50_inference_ms: float
    p95_inference_ms: float
    effective_fps: float  # processed_frames / wall_time

    mean_visibility_all: float
    mean_presence_all: float
    mean_visibility_arms: float
    mean_presence_arms: float

    occluded_frac_mean: float  # доля landmark'ов с visibility < thr (среднее по кадрам)
    tracking_loss_events: int  # сколько раз "пропадала" поза (detected->not detected)
    longest_loss_streak_frames: int

    # "стабильность" (джиттер) на low-motion landmark'ах (в метрах, по WorldLandmarks)
    jitter_world_m_mean: float
    jitter_world_m_p95: float

    # динамика кистей (скорости) — удобно для ваших видео (движение/быстрые махи)
    wrist_speed_m_s_mean: float
    wrist_speed_m_s_p95: float


# ----------------------------
# Utils
# ----------------------------

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def download_with_fallback(urls: List[str], out_path: Path, timeout_s: int = 30) -> None:
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mp_pose_research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return
        except Exception as ex:
            last_err = ex
            continue
    raise RuntimeError(f"Failed to download model to {out_path}. Last error: {last_err}")


def list_videos_in_dir(videos_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    files = [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def mp_image_from_bgr(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_pose_on_rgb(rgb: np.ndarray, result) -> np.ndarray:
    """
    Рисует позу на RGB-кадре без landmark_pb2 (через OpenCV).
    result.pose_landmarks: List[List[NormalizedLandmark]]
    """
    annotated = np.copy(rgb)

    pose_landmarks = getattr(result, "pose_landmarks", None)
    if not pose_landmarks or len(pose_landmarks) == 0:
        return annotated

    h, w = annotated.shape[:2]

    # Берём первую позу (num_poses обычно 1)
    lms = pose_landmarks[0]

    # Преобразуем в пиксели
    pts = []
    vis = []
    for lm in lms:
        x_px = int(round(float(lm.x) * w))
        y_px = int(round(float(lm.y) * h))
        pts.append((x_px, y_px))
        vis.append(float(getattr(lm, "visibility", 1.0)))

    # Линии (кости)
    for (a, b) in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            # Можно условно фильтровать по видимости
            if vis[a] > 0.2 and vis[b] > 0.2:
                cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Точки (суставы)
    for i, (x, y) in enumerate(pts):
        if vis[i] > 0.2:
            cv2.circle(annotated, (x, y), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    return annotated


def segmentation_to_gray_uint8(seg_mask) -> Optional[np.ndarray]:
    """
    seg_mask может быть mp.Image или np.ndarray.
    Возвращает HxW uint8 (0..255) или None.
    """
    if seg_mask is None:
        return None

    try:
        if hasattr(seg_mask, "numpy_view"):
            m = seg_mask.numpy_view()
        else:
            m = np.asarray(seg_mask)
        if m is None:
            return None
        m = np.squeeze(m)
        if m.ndim != 2:
            return None
        # обычно float32 [0..1]
        if m.dtype != np.uint8:
            m = np.clip(m, 0.0, 1.0)
            m = (m * 255.0).astype(np.uint8)
        return m
    except Exception:
        return None


def mean_over_indices(values: Sequence[float], idxs: Sequence[int]) -> float:
    if not values:
        return 0.0
    v = np.array(values, dtype=np.float64)
    sel = v[list(idxs)]
    return float(np.mean(sel))


# ----------------------------
# Core runner
# ----------------------------

def create_landmarker(model_path: Path,
                      running_mode: str,
                      delegate: str,
                      num_poses: int,
                      min_pose_detection_conf: float,
                      min_pose_presence_conf: float,
                      min_tracking_conf: float,
                      output_segmentation_masks: bool,
                      result_callback=None):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    if running_mode == "IMAGE":
        rm = VisionRunningMode.IMAGE
    elif running_mode == "VIDEO":
        rm = VisionRunningMode.VIDEO
    elif running_mode == "LIVE_STREAM":
        rm = VisionRunningMode.LIVE_STREAM
    else:
        raise ValueError(f"Unknown running_mode={running_mode}")

    if delegate == "CPU":
        d = BaseOptions.Delegate.CPU
    elif delegate == "GPU":
        d = BaseOptions.Delegate.GPU
    else:
        raise ValueError(f"Unknown delegate={delegate}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=d),
        running_mode=rm,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_conf,
        min_pose_presence_confidence=min_pose_presence_conf,
        min_tracking_confidence=min_tracking_conf,
        output_segmentation_masks=output_segmentation_masks,
        result_callback=result_callback,
    )
    return PoseLandmarker.create_from_options(options)


def run_on_video(video_path: Path,
                 model_path: Path,
                 out_dir: Path,
                 running_mode: str,
                 delegate: str,
                 stride: int = 1,
                 max_frames: int = 0,
                 render_2d: bool = True,
                 render_segmentation: bool = False,
                 num_poses: int = 1,
                 min_pose_detection_conf: float = 0.5,
                 min_pose_presence_conf: float = 0.5,
                 min_tracking_conf: float = 0.5,
                 visibility_occlusion_thr: float = 0.5) -> VideoRunSummary:
    """
    running_mode: "IMAGE" or "VIDEO"
    """
    assert running_mode in ("IMAGE", "VIDEO")

    safe_mkdir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not fps_src or fps_src <= 1e-3:
        fps_src = 30.0

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # CSV writers
    frame_csv_path = out_dir / "frame_metrics.csv"
    lm2d_csv_path = out_dir / "landmarks_2d.csv"
    lm3d_csv_path = out_dir / "landmarks_3d_world.csv"

    frame_f = frame_csv_path.open("w", newline="", encoding="utf-8")
    lm2d_f = lm2d_csv_path.open("w", newline="", encoding="utf-8")
    lm3d_f = lm3d_csv_path.open("w", newline="", encoding="utf-8")

    frame_w = csv.writer(frame_f)
    lm2d_w = csv.writer(lm2d_f)
    lm3d_w = csv.writer(lm3d_f)

    frame_w.writerow([
        "frame_idx", "timestamp_ms", "has_pose",
        "inference_ms",
        "mean_visibility_all", "mean_presence_all",
        "mean_visibility_arms", "mean_presence_arms",
        "occluded_frac",
        "wrist_speed_m_s",
    ])

    lm2d_w.writerow(["frame_idx", "timestamp_ms", "landmark_idx", "landmark_name",
                     "x", "y", "z", "visibility", "presence"])
    lm3d_w.writerow(["frame_idx", "timestamp_ms", "landmark_idx", "landmark_name",
                     "x_m", "y_m", "z_m", "visibility", "presence"])

    # Video writers (optional)
    annotated_writer = None
    seg_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if render_2d:
        annotated_writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"),
            fourcc,
            fps_src / max(1, stride),
            (width, height),
            True,
        )
    if render_segmentation:
        seg_writer = cv2.VideoWriter(
            str(out_dir / "segmentation.mp4"),
            fourcc,
            fps_src / max(1, stride),
            (width, height),
            False,  # grayscale
        )

    inference_times: List[float] = []
    vis_all_list: List[float] = []
    pres_all_list: List[float] = []
    vis_arms_list: List[float] = []
    pres_arms_list: List[float] = []
    occluded_frac_list: List[float] = []

    # tracking loss
    prev_has_pose = False
    tracking_loss_events = 0
    current_loss_streak = 0
    longest_loss_streak = 0

    # jitter (world) + wrist speed
    prev_world: Optional[np.ndarray] = None  # (33, 3)
    prev_ts_ms: Optional[int] = None
    jitter_vals: List[float] = []
    wrist_speed_vals: List[float] = []

    # Для "джиттера" берём относительно стабильные точки (плечи+бёдра)
    stable_idxs = [
        LM["left_shoulder"], LM["right_shoulder"],
        LM["left_hip"], LM["right_hip"],
    ]


    arm_idxs = ARM_IDXS
    wrist_idxs = WRIST_IDXS


    # Create landmarker
    with create_landmarker(
        model_path=model_path,
        running_mode=running_mode,
        delegate=delegate,
        num_poses=num_poses,
        min_pose_detection_conf=min_pose_detection_conf,
        min_pose_presence_conf=min_pose_presence_conf,
        min_tracking_conf=min_tracking_conf,
        output_segmentation_masks=render_segmentation,
    ) as landmarker:

        processed = 0
        wall_t0 = time.perf_counter()

        frame_idx = -1
        pbar = tqdm(total=frames_total if frames_total > 0 else None,
                    desc=f"{video_path.name} | {model_path.stem} | {running_mode}",
                    unit="frame")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            pbar.update(1)

            if stride > 1 and (frame_idx % stride != 0):
                continue
            processed += 1
            if max_frames > 0 and processed > max_frames:
                break

            ts_ms = int(round(frame_idx * 1000.0 / fps_src))
            mp_image = mp_image_from_bgr(frame_bgr)

            t0 = time.perf_counter()
            if running_mode == "VIDEO":
                result = landmarker.detect_for_video(mp_image, ts_ms)
            else:
                result = landmarker.detect(mp_image)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            inference_times.append(infer_ms)

            has_pose = bool(getattr(result, "pose_landmarks", None)) and len(result.pose_landmarks) > 0

            # tracking loss events
            if prev_has_pose and (not has_pose):
                tracking_loss_events += 1
                current_loss_streak = 1
            elif (not has_pose):
                current_loss_streak += 1
            else:
                current_loss_streak = 0
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)
            prev_has_pose = has_pose

            # defaults
            mean_vis_all = 0.0
            mean_pres_all = 0.0
            mean_vis_arms = 0.0
            mean_pres_arms = 0.0
            occluded_frac = 1.0
            wrist_speed = 0.0

            # segmentation
            seg_gray = None
            if render_segmentation:
                segs = getattr(result, "segmentation_masks", None)
                if segs and len(segs) > 0:
                    seg_gray = segmentation_to_gray_uint8(segs[0])

            if has_pose:
                lm2d = result.pose_landmarks[0]
                lm3d = result.pose_world_landmarks[0] if getattr(result, "pose_world_landmarks", None) else None

                vis = [float(getattr(lm, "visibility", 0.0)) for lm in lm2d]
                pres = [float(getattr(lm, "presence", 0.0)) for lm in lm2d]

                mean_vis_all = float(np.mean(vis))
                mean_pres_all = float(np.mean(pres))
                mean_vis_arms = mean_over_indices(vis, arm_idxs)
                mean_pres_arms = mean_over_indices(pres, arm_idxs)

                occluded_frac = float(np.mean(np.array(vis) < visibility_occlusion_thr))

                # write landmarks 2d
                for i, lm in enumerate(lm2d):
                    lm2d_w.writerow([
                        frame_idx, ts_ms, i, LANDMARK_NAMES[i],
                        float(lm.x), float(lm.y), float(lm.z),
                        float(getattr(lm, "visibility", 0.0)),
                        float(getattr(lm, "presence", 0.0)),
                    ])

                # world landmarks
                if lm3d is not None:
                    world = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
                    world_vis = np.zeros((NUM_LANDMARKS,), dtype=np.float64)

                    for i, lm in enumerate(lm3d):
                        x, y, z = float(lm.x), float(lm.y), float(lm.z)
                        world[i] = (x, y, z)
                        world_vis[i] = float(getattr(lm, "visibility", 0.0))

                        lm3d_w.writerow([
                            frame_idx, ts_ms, i, LANDMARK_NAMES[i],
                            x, y, z,
                            float(getattr(lm, "visibility", 0.0)),
                            float(getattr(lm, "presence", 0.0)),
                        ])

                    # jitter (stable landmarks) and wrist speed
                    if prev_world is not None and prev_ts_ms is not None:
                        dt = max(1e-3, (ts_ms - prev_ts_ms) / 1000.0)

                        # jitter = среднее смещение stable точек между кадрами (в метрах)
                        diffs = world[stable_idxs] - prev_world[stable_idxs]
                        jitter = float(np.mean(np.linalg.norm(diffs, axis=1)))
                        jitter_vals.append(jitter)

                        # wrist speed (среднее по двум кистям)
                        wrist_d = world[wrist_idxs] - prev_world[wrist_idxs]
                        wrist_speed = float(np.mean(np.linalg.norm(wrist_d, axis=1)) / dt)
                        wrist_speed_vals.append(wrist_speed)

                    prev_world = world
                    prev_ts_ms = ts_ms

            # collect per-frame aggregates
            vis_all_list.append(mean_vis_all)
            pres_all_list.append(mean_pres_all)
            vis_arms_list.append(mean_vis_arms)
            pres_arms_list.append(mean_pres_arms)
            occluded_frac_list.append(occluded_frac)

            frame_w.writerow([
                frame_idx, ts_ms, int(has_pose),
                float(infer_ms),
                mean_vis_all, mean_pres_all,
                mean_vis_arms, mean_pres_arms,
                occluded_frac,
                wrist_speed,
            ])

            # render outputs
            if annotated_writer is not None:
                rgb = mp_image.numpy_view()
                ann_rgb = draw_pose_on_rgb(rgb, result)
                ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
                annotated_writer.write(ann_bgr)

            if seg_writer is not None and seg_gray is not None:
                # seg_writer ожидает HxW или HxWx1 при isColor=False
                if seg_gray.shape != (height, width):
                    seg_gray = cv2.resize(seg_gray, (width, height), interpolation=cv2.INTER_LINEAR)
                seg_writer.write(seg_gray)

        pbar.close()
        wall_t1 = time.perf_counter()

    cap.release()
    if annotated_writer is not None:
        annotated_writer.release()
    if seg_writer is not None:
        seg_writer.release()

    frame_f.close()
    lm2d_f.close()
    lm3d_f.close()

    wall_s = max(1e-6, wall_t1 - wall_t0)
    frames_processed = processed
    detection_rate = float(np.mean([v > 0.0 for v in pres_all_list])) if pres_all_list else 0.0
    # detection_rate выше — грубо; лучше по has_pose, пересчитаем:
    # (мы не сохранили has_pose list отдельно, но можно восстановить по mean_presence_all > 0)
    # В frame_metrics.csv есть точный has_pose.

    # более корректно: прочитаем has_pose из frame_metrics.csv (быстро)
    has_pose_count = 0
    with frame_csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            has_pose_count += int(row["has_pose"])
    detection_rate = has_pose_count / max(1, frames_processed)

    eff_fps = frames_processed / wall_s

    mean_infer = float(np.mean(inference_times)) if inference_times else 0.0
    p50 = percentile(inference_times, 50)
    p95 = percentile(inference_times, 95)

    mean_vis_all = float(np.mean(vis_all_list)) if vis_all_list else 0.0
    mean_pres_all = float(np.mean(pres_all_list)) if pres_all_list else 0.0
    mean_vis_arms = float(np.mean(vis_arms_list)) if vis_arms_list else 0.0
    mean_pres_arms = float(np.mean(pres_arms_list)) if pres_arms_list else 0.0
    occl_mean = float(np.mean(occluded_frac_list)) if occluded_frac_list else 1.0

    jitter_mean = float(np.mean(jitter_vals)) if jitter_vals else 0.0
    jitter_p95 = percentile(jitter_vals, 95)

    wrist_speed_mean = float(np.mean(wrist_speed_vals)) if wrist_speed_vals else 0.0
    wrist_speed_p95 = percentile(wrist_speed_vals, 95)

    summary = VideoRunSummary(
        video=video_path.name,
        model=model_path.stem,
        running_mode=running_mode,
        delegate=delegate,

        frames_total=frames_total,
        frames_processed=frames_processed,
        stride=stride,
        fps_src=float(fps_src),

        detection_rate=float(detection_rate),

        mean_inference_ms=mean_infer,
        p50_inference_ms=p50,
        p95_inference_ms=p95,
        effective_fps=float(eff_fps),

        mean_visibility_all=mean_vis_all,
        mean_presence_all=mean_pres_all,
        mean_visibility_arms=mean_vis_arms,
        mean_presence_arms=mean_pres_arms,

        occluded_frac_mean=occl_mean,
        tracking_loss_events=int(tracking_loss_events),
        longest_loss_streak_frames=int(longest_loss_streak),

        jitter_world_m_mean=jitter_mean,
        jitter_world_m_p95=jitter_p95,

        wrist_speed_m_s_mean=wrist_speed_mean,
        wrist_speed_m_s_p95=wrist_speed_p95,
    )

    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ----------------------------
# 3D Visualizer
# ----------------------------

def load_world_landmarks_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      frames_ts: (T,) timestamp_ms
      coords: (T, 33, 3) world coords
    """
    rows: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    ts_map: Dict[int, int] = {}

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fi = int(row["frame_idx"])
            ts = int(row["timestamp_ms"])
            li = int(row["landmark_idx"])
            x = float(row["x_m"]); y = float(row["y_m"]); z = float(row["z_m"])
            ts_map[fi] = ts
            rows.setdefault(fi, {})[li] = (x, y, z)

    frame_ids = sorted(rows.keys())
    if not frame_ids:
        return np.array([], dtype=np.int64), np.zeros((0, NUM_LANDMARKS, 3), dtype=np.float64)

    frames_ts = np.array([ts_map[i] for i in frame_ids], dtype=np.int64)
    coords = np.zeros((len(frame_ids), NUM_LANDMARKS, 3), dtype=np.float64)

    for t, fi in enumerate(frame_ids):
        for li in range(NUM_LANDMARKS):
            coords[t, li] = rows[fi].get(li, (np.nan, np.nan, np.nan))

    return frames_ts, coords


def visualize_3d(world_csv: Path, show: bool = True, out_mp4: Optional[Path] = None, fps: float = 30.0) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames_ts, coords = load_world_landmarks_csv(world_csv)
    if coords.shape[0] == 0:
        raise RuntimeError(f"No world landmarks found in {world_csv}")

    # Нормализация камеры: чтобы было удобно смотреть, центрируем по hips
    hip_center = np.nanmean(coords[:, [HIPS[0].value, HIPS[1].value], :], axis=1)  # (T,3)
    coords_c = coords - hip_center[:, None, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"3D Pose (world) — {world_csv.parent.name}")

    scat = ax.scatter([], [], [], s=10)

    # линии скелета
    lines = []
    for (a, b) in POSE_CONNECTIONS:
        ln, = ax.plot([], [], [], linewidth=2)
        lines.append((ln, a, b))

    def set_axes_limits(all_coords: np.ndarray):
        valid = all_coords[np.isfinite(all_coords).all(axis=1)]
        if valid.size == 0:
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            return
        mins = valid.min(axis=0)
        maxs = valid.max(axis=0)
        # чуть расширим
        pad = 0.1 * np.maximum(1e-6, maxs - mins)
        mins -= pad; maxs += pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    set_axes_limits(coords_c.reshape(-1, 3))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    def update(i: int):
        pts = coords_c[i]
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for (ln, a, b) in lines:
            pa = pts[a]; pb = pts[b]
            ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            ln.set_3d_properties([pa[2], pb[2]])

        return [scat] + [ln for (ln, _, _) in lines]

    interval_ms = 1000.0 / max(1e-6, fps)
    anim = FuncAnimation(fig, update, frames=len(coords_c), interval=interval_ms, blit=False)

    if out_mp4 is not None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(str(out_mp4), fps=fps)
            print(f"Saved 3D animation to: {out_mp4}")
        except Exception as ex:
            eprint("Failed to save MP4 (likely no ffmpeg). Error:", ex)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# LIVE_STREAM webcam demo
# ----------------------------

def run_webcam(model_path: Path,
               delegate: str,
               camera_id: int,
               render_segmentation: bool,
               num_poses: int,
               min_pose_detection_conf: float,
               min_pose_presence_conf: float,
               min_tracking_conf: float) -> None:
    """
    Демонстрация LIVE_STREAM: читает webcam, отправляет detect_async,
    показывает оверлей (и маску, если включена).
    """
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    if delegate == "CPU":
        d = BaseOptions.Delegate.CPU
    elif delegate == "GPU":
        d = BaseOptions.Delegate.GPU
    else:
        raise ValueError(f"Unknown delegate={delegate}")

    latest_result = {"result": None, "ts": 0}

    def callback(res, out_img: mp.Image, ts_ms: int):
        latest_result["result"] = res
        latest_result["ts"] = ts_ms

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=d),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_conf,
        min_pose_presence_confidence=min_pose_presence_conf,
        min_tracking_confidence=min_tracking_conf,
        output_segmentation_masks=render_segmentation,
        result_callback=callback,
    )

    cap = cv2.VideoCapture(int(camera_id))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id={camera_id}")

    with PoseLandmarker.create_from_options(options) as landmarker:
        t0 = time.perf_counter()
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts_ms = int((time.perf_counter() - t0) * 1000.0)
            mp_image = mp_image_from_bgr(frame_bgr)

            # async inference
            landmarker.detect_async(mp_image, ts_ms)

            # визуализация последнего результата
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = latest_result["result"]
            if res is not None:
                rgb = draw_pose_on_rgb(rgb, res)

                if render_segmentation:
                    segs = getattr(res, "segmentation_masks", None)
                    if segs and len(segs) > 0:
                        seg_gray = segmentation_to_gray_uint8(segs[0])
                        if seg_gray is not None:
                            seg_gray = cv2.resize(seg_gray, (rgb.shape[1], rgb.shape[0]))
                            # маленьким окошком справа сверху
                            h, w = seg_gray.shape
                            hh = max(1, h // 4); ww = max(1, w // 4)
                            seg_small = cv2.resize(seg_gray, (ww, hh))
                            seg_small_rgb = cv2.cvtColor(seg_small, cv2.COLOR_GRAY2RGB)
                            rgb[0:hh, (w-ww):w, :] = seg_small_rgb

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("MediaPipe Pose Landmarker (LIVE_STREAM) — press ESC to exit", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# CLI
# ----------------------------

def cmd_download_models(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).resolve()
    safe_mkdir(model_dir)

    for name, urls in OFFICIAL_MODEL_URLS.items():
        out = model_dir / f"{name}.task"
        if out.exists() and not args.force:
            print(f"[skip] {out.name} already exists")
            continue

        print(f"[download] {name} -> {out}")
        download_with_fallback(urls, out)
        print(f"[ok] {out}")

    print(f"Models are in: {model_dir}")


def find_models(model_dir: Path) -> List[Path]:
    # ищем .task файлы
    models = sorted(model_dir.glob("*.task"))
    return models


def cmd_benchmark_videos(args: argparse.Namespace) -> None:
    videos: List[Path] = []
    if args.videos_dir:
        videos = list_videos_in_dir(Path(args.videos_dir))
    if args.videos:
        videos.extend([Path(v) for v in args.videos])

    videos = [v.resolve() for v in videos]
    if not videos:
        raise RuntimeError("No videos provided. Use --videos_dir or list video paths.")

    model_dir = Path(args.model_dir).resolve()
    models = find_models(model_dir)
    if args.models:
        # фильтруем по stem
        wanted = set(args.models)
        models = [m for m in models if m.stem in wanted or m.name in wanted]
    if not models:
        raise RuntimeError(f"No models found in {model_dir}. Run download-models first or place .task files there.")

    out_root = Path(args.out_dir).resolve()
    safe_mkdir(out_root)

    running_modes = [m.strip().upper() for m in args.modes]
    for rm in running_modes:
        if rm not in ("IMAGE", "VIDEO"):
            raise ValueError("For video benchmarking, --modes must be IMAGE and/or VIDEO.")

    summaries: List[VideoRunSummary] = []

    for video in videos:
        for model_path in models:
            for rm in running_modes:
                run_dir = out_root / video.stem / model_path.stem / rm / args.delegate
                summary = run_on_video(
                    video_path=video,
                    model_path=model_path,
                    out_dir=run_dir,
                    running_mode=rm,
                    delegate=args.delegate,
                    stride=int(args.stride),
                    max_frames=int(args.max_frames),
                    render_2d=bool(args.render_2d),
                    render_segmentation=bool(args.segmentation),
                    num_poses=int(args.num_poses),
                    min_pose_detection_conf=float(args.min_pose_detection_confidence),
                    min_pose_presence_conf=float(args.min_pose_presence_confidence),
                    min_tracking_conf=float(args.min_tracking_confidence),
                    visibility_occlusion_thr=float(args.occlusion_thr),
                )
                summaries.append(summary)

                # 3D (опционально) — только если есть world landmarks
                if args.render_3d:
                    world_csv = run_dir / "landmarks_3d_world.csv"
                    if world_csv.exists():
                        out_mp4 = run_dir / "pose3d.mp4" if args.save_3d_mp4 else None
                        try:
                            visualize_3d(world_csv, show=False, out_mp4=out_mp4, fps=max(5.0, summary.fps_src / max(1, summary.stride)))
                        except Exception as ex:
                            eprint("[warn] 3D visualization failed:", ex)

    # write global summary
    summary_csv = out_root / "summary.csv"
    summary_json = out_root / "summary.json"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = list(asdict(summaries[0]).keys()) if summaries else []
        w.writerow(header)
        for s in summaries:
            d = asdict(s)
            w.writerow([d[h] for h in header])

    summary_json.write_text(json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDONE. Results in: {out_root}")
    print(f"- {summary_csv}")
    print(f"- {summary_json}")


def cmd_visualize_3d(args: argparse.Namespace) -> None:
    world_csv = Path(args.world_csv).resolve()
    out_mp4 = Path(args.out_mp4).resolve() if args.out_mp4 else None
    visualize_3d(world_csv, show=not args.no_show, out_mp4=out_mp4, fps=float(args.fps))


def cmd_webcam(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).resolve()
    run_webcam(
        model_path=model_path,
        delegate=args.delegate,
        camera_id=int(args.camera_id),
        render_segmentation=bool(args.segmentation),
        num_poses=int(args.num_poses),
        min_pose_detection_conf=float(args.min_pose_detection_confidence),
        min_pose_presence_conf=float(args.min_pose_presence_confidence),
        min_tracking_conf=float(args.min_tracking_confidence),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mp_pose_research.py",
        description="MediaPipe Pose Landmarker research tool (models/metrics/2D+3D outputs)."
    )
    sub = p.add_subparsers(dest="cmd", required=False)

    # download-models
    p_dl = sub.add_parser("download-models", help="Download official pose landmarker models (lite/full/heavy).")
    p_dl.add_argument("--model_dir", default="models", help="Directory to store .task models.")
    p_dl.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p_dl.set_defaults(func=cmd_download_models)

    # benchmark-videos
    p_b = sub.add_parser("benchmark-videos", help="Benchmark on video files (IMAGE/VIDEO).")
    p_b.add_argument("--videos_dir", default="", help="Directory with videos (mp4/mov/mkv/avi/webm).")
    p_b.add_argument("videos", nargs="*", help="Explicit video paths (optional).")
    p_b.add_argument("--model_dir", default="models", help="Directory with .task models.")
    p_b.add_argument("--models", nargs="*", default=[], help="Filter by model stem/name (e.g., pose_landmarker_full).")
    p_b.add_argument("--out_dir", default="runs", help="Output root directory.")

    p_b.add_argument("--modes", nargs="+", default=["IMAGE", "VIDEO"], help="Running modes to test: IMAGE VIDEO")
    p_b.add_argument("--delegate", choices=["CPU", "GPU"], default="CPU", help="Execution delegate.")
    p_b.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    p_b.add_argument("--max_frames", type=int, default=0, help="Limit processed frames per video (0=all).")

    p_b.add_argument("--num_poses", type=int, default=1)
    p_b.add_argument("--min_pose_detection_confidence", type=float, default=0.5)
    p_b.add_argument("--min_pose_presence_confidence", type=float, default=0.5)
    p_b.add_argument("--min_tracking_confidence", type=float, default=0.5)

    p_b.add_argument("--render_2d", action="store_true", help="Write annotated.mp4 with skeleton overlay.")
    p_b.add_argument("--segmentation", action="store_true", help="Enable output_segmentation_masks and write segmentation.mp4.")
    p_b.add_argument("--occlusion_thr", type=float, default=0.5, help="visibility<thr => occluded")
    p_b.add_argument("--render_3d", action="store_true", help="Generate 3D animation (headless).")
    p_b.add_argument("--save_3d_mp4", action="store_true", help="Try saving pose3d.mp4 (requires ffmpeg).")

    p_b.set_defaults(func=cmd_benchmark_videos)

    # visualize-3d
    p_v = sub.add_parser("visualize-3d", help="Visualize 3D from landmarks_3d_world.csv.")
    p_v.add_argument("world_csv", help="Path to landmarks_3d_world.csv")
    p_v.add_argument("--fps", type=float, default=30.0)
    p_v.add_argument("--out_mp4", default="", help="Optional path to save mp4 (needs ffmpeg).")
    p_v.add_argument("--no_show", action="store_true", help="Do not open window.")
    p_v.set_defaults(func=cmd_visualize_3d)

    # webcam
    p_w = sub.add_parser("webcam", help="Live stream demo (LIVE_STREAM) from webcam.")
    p_w.add_argument("--model_path", required=True, help="Path to a .task model file.")
    p_w.add_argument("--delegate", choices=["CPU", "GPU"], default="CPU")
    p_w.add_argument("--camera_id", type=int, default=0)
    p_w.add_argument("--segmentation", action="store_true")

    p_w.add_argument("--num_poses", type=int, default=1)
    p_w.add_argument("--min_pose_detection_confidence", type=float, default=0.5)
    p_w.add_argument("--min_pose_presence_confidence", type=float, default=0.5)
    p_w.add_argument("--min_tracking_confidence", type=float, default=0.5)

    p_w.set_defaults(func=cmd_webcam)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # If no subcommand is provided, run default benchmark-videos with your standard parameters.
    if getattr(args, "cmd", None) is None:
        # Equivalent to:
        # python3 mp_pose_research.py benchmark-videos \
        #   --videos_dir videos \
        #   --model_dir models \
        #   --out_dir runs \
        #   --modes IMAGE VIDEO \
        #   --delegate CPU \
        #   --render_2d \
        #   --segmentation
        default_argv = [
            "benchmark-videos",
            "--videos_dir", "videos",
            "--model_dir", "models",
            "--out_dir", "runs",
            "--modes", "IMAGE", "VIDEO",
            "--delegate", "CPU",
            "--render_2d",
            "--segmentation",
        ]
        args = parser.parse_args(default_argv)

    args.func(args)


if __name__ == "__main__":
    main()
