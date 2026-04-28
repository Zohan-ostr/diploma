#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    from mmpose.apis import MMPoseInferencer
except Exception as ex:
    MMPoseInferencer = None
    IMPORT_ERROR = ex
else:
    IMPORT_ERROR = None

# Unified 33-landmark namespace used across the project
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
NUM_LM = len(LANDMARK_NAMES)

# COCO-17 / Human3D mapping into unified 33-landmark namespace
# 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lel,8 rel,9 lwr,10 rwr,11 lhip,12 rhip,13 lknee,14 rknee,15 lank,16 rank
COCO17_TO_UNIFIED = {
    0: LM["nose"],
    1: LM["left_eye"],
    2: LM["right_eye"],
    3: LM["left_ear"],
    4: LM["right_ear"],
    5: LM["left_shoulder"],
    6: LM["right_shoulder"],
    7: LM["left_elbow"],
    8: LM["right_elbow"],
    9: LM["left_wrist"],
    10: LM["right_wrist"],
    11: LM["left_hip"],
    12: LM["right_hip"],
    13: LM["left_knee"],
    14: LM["right_knee"],
    15: LM["left_ankle"],
    16: LM["right_ankle"],
}

STABLE_IDXS = [LM["left_shoulder"], LM["right_shoulder"], LM["left_hip"], LM["right_hip"]]
UPPER_ARM = (LM["left_shoulder"], LM["left_elbow"])

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

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))

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

def list_videos(videos_dir: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def unified_visibility(native_visibility: Optional[float] = None,
                       keypoint_confidence: Optional[float] = None,
                       threshold: float = 0.3) -> float:
    if native_visibility is not None and np.isfinite(float(native_visibility)):
        return clip01(float(native_visibility))
    if keypoint_confidence is None or not np.isfinite(float(keypoint_confidence)):
        return 0.0
    conf = float(keypoint_confidence)
    if conf <= 0:
        return 0.0
    return 1.0 if conf >= threshold else conf / max(threshold, 1e-9)

def unified_presence(native_presence: Optional[float] = None,
                     keypoint_confidence: Optional[float] = None) -> float:
    if native_presence is not None and np.isfinite(float(native_presence)):
        return clip01(float(native_presence))
    if keypoint_confidence is None or not np.isfinite(float(keypoint_confidence)):
        return 0.0
    return clip01(float(keypoint_confidence))

def upper_arm_metrics(world_points: np.ndarray, arm_len_ref_m: float) -> Tuple[float, float, float]:
    ls_idx, le_idx = UPPER_ARM
    if not np.isfinite(world_points[[ls_idx, le_idx]]).all():
        return float("nan"), float("nan"), float("nan")
    arm_len = float(np.linalg.norm(world_points[le_idx] - world_points[ls_idx]))
    abs_err = abs(arm_len - arm_len_ref_m)
    rel_err = (100.0 * abs_err / arm_len_ref_m) if arm_len_ref_m > 0 else float("nan")
    return arm_len, abs_err, rel_err

def list_from_obj(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)

def choose_first_person_prediction(predictions):
    if predictions is None:
        return None
    if isinstance(predictions, list):
        # Sometimes it's [instances], sometimes [[instances]]
        if len(predictions) == 0:
            return None
        if isinstance(predictions[0], list):
            return predictions[0][0] if predictions[0] else None
        return predictions[0]
    return predictions

def to_numpy_points(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def map_2d_to_unified(keypoints2d: np.ndarray,
                      scores: Optional[np.ndarray],
                      kpt_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out2d = np.full((NUM_LM, 3), np.nan, dtype=np.float64)
    vis = np.zeros(NUM_LM, dtype=np.float64)
    pres = np.zeros(NUM_LM, dtype=np.float64)

    count = min(len(keypoints2d), 17)
    for src_idx in range(count):
        dst_idx = COCO17_TO_UNIFIED[src_idx]
        xy = keypoints2d[src_idx]
        if np.shape(xy)[0] >= 2:
            out2d[dst_idx, 0] = float(xy[0])
            out2d[dst_idx, 1] = float(xy[1])
            out2d[dst_idx, 2] = 0.0
        conf = float(scores[src_idx]) if scores is not None and src_idx < len(scores) else float("nan")
        vis[dst_idx] = unified_visibility(keypoint_confidence=conf, threshold=kpt_thr)
        pres[dst_idx] = unified_presence(keypoint_confidence=conf)
    return out2d, vis, pres

def map_3d_to_unified(keypoints3d: np.ndarray,
                      scores: Optional[np.ndarray],
                      vis_in: Optional[np.ndarray],
                      pres_in: Optional[np.ndarray],
                      kpt_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out3d = np.full((NUM_LM, 3), np.nan, dtype=np.float64)
    vis = np.zeros(NUM_LM, dtype=np.float64)
    pres = np.zeros(NUM_LM, dtype=np.float64)

    count = min(len(keypoints3d), 17)
    for src_idx in range(count):
        dst_idx = COCO17_TO_UNIFIED[src_idx]
        xyz = keypoints3d[src_idx]
        if np.shape(xyz)[0] >= 3:
            out3d[dst_idx] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        conf = float(scores[src_idx]) if scores is not None and src_idx < len(scores) else float("nan")
        nv = float(vis_in[src_idx]) if vis_in is not None and src_idx < len(vis_in) else None
        npres = float(pres_in[src_idx]) if pres_in is not None and src_idx < len(pres_in) else None
        vis[dst_idx] = unified_visibility(native_visibility=nv, keypoint_confidence=conf, threshold=kpt_thr)
        pres[dst_idx] = unified_presence(native_presence=npres, keypoint_confidence=conf)
    return out3d, vis, pres

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

def iter_inferencer_results(inferencer, video_path: Path, device: str, bbox_thr: float, num_instances: int,
                            rebase_keypoint_height: bool):
    kwargs = dict(
        bbox_thr=bbox_thr,
        num_instances=num_instances,
        return_vis=False,
        show=False,
        draw_bbox=False,
        draw_heatmap=False,
        no_save_vis=True,
        no_save_pred=True,
    )
    if rebase_keypoint_height:
        kwargs["rebase_keypoint_height"] = True
    gen = inferencer(str(video_path), **kwargs)
    for item in gen:
        yield item

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default=cfg.get("videos_dir", "../../videos/single_camera"))
    ap.add_argument("--out_dir", default=str(project_root / "runs" / cfg.get("run_names", {}).get("baseline", "run")))
    ap.add_argument("--device", default=cfg.get("single_camera", {}).get("device", "cpu"))
    ap.add_argument("--arm_len_ref_m", type=float, default=float(cfg.get("arm_len_ref_m", 0.249)))
    ap.add_argument("--stride", type=int, default=int(cfg.get("single_camera", {}).get("stride", 1)))
    ap.add_argument("--max_frames", type=int, default=int(cfg.get("single_camera", {}).get("max_frames", 0)))
    ap.add_argument("--bbox_thr", type=float, default=float(cfg.get("single_camera", {}).get("bbox_thr", 0.3)))
    ap.add_argument("--kpt_thr", type=float, default=float(cfg.get("single_camera", {}).get("kpt_thr", 0.3)))
    ap.add_argument("--num_instances", type=int, default=int(cfg.get("single_camera", {}).get("num_instances", 1)))
    ap.add_argument("--pose3d_alias", default=str(cfg.get("model", {}).get("pose3d_alias", "human3d")))
    ap.add_argument("--rebase_keypoint_height", action="store_true",
                    default=bool(cfg.get("single_camera", {}).get("rebase_keypoint_height", True)))
    args = ap.parse_args()

    if MMPoseInferencer is None:
        raise RuntimeError(f"MMPose is not available: {IMPORT_ERROR}")

    videos_dir = (project_root / args.videos_dir).resolve() if not Path(args.videos_dir).is_absolute() else Path(args.videos_dir)
    out_root = Path(args.out_dir).resolve() if Path(args.out_dir).is_absolute() else (project_root / args.out_dir).resolve()
    safe_mkdir(out_root)

    videos = list_videos(videos_dir)
    if not videos:
        raise RuntimeError(f"No videos found in {videos_dir}")

    inferencer = MMPoseInferencer(pose3d=args.pose3d_alias, device=args.device)
    summaries: List[VideoRunSummary] = []

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        if not fps_src or fps_src <= 1e-3:
            fps_src = 30.0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        leaf = out_root / video.stem / args.pose3d_alias / "VIDEO" / args.device.upper()
        safe_mkdir(leaf)

        frame_csv = (leaf / "frame_metrics.csv").open("w", newline="", encoding="utf-8")
        lm2d_csv = (leaf / "landmarks_2d.csv").open("w", newline="", encoding="utf-8")
        lm3d_csv = (leaf / "landmarks_3d_world.csv").open("w", newline="", encoding="utf-8")
        bones_csv = (leaf / "bones_metrics.csv").open("w", newline="", encoding="utf-8")

        frame_w = csv.writer(frame_csv)
        lm2d_w = csv.writer(lm2d_csv)
        lm3d_w = csv.writer(lm3d_csv)
        bones_w = csv.writer(bones_csv)

        frame_w.writerow([
            "frame_idx","timestamp_ms","has_pose","inference_ms",
            "mean_visibility","mean_presence","jitter_frame_m",
            "left_upper_arm_len_m","left_upper_arm_abs_error_m","left_upper_arm_rel_error_pct",
        ])
        lm2d_w.writerow(["frame_idx","timestamp_ms","landmark_idx","landmark_name","x","y","z","visibility","presence"])
        lm3d_w.writerow(["frame_idx","timestamp_ms","landmark_idx","landmark_name","x_m","y_m","z_m","visibility","presence"])
        bones_w.writerow([
            "frame_idx","timestamp_ms","arm_len_ref_m",
            "left_upper_arm_len_m","left_upper_arm_abs_error_m","left_upper_arm_rel_error_pct"
        ])

        inference_times = []
        mean_vis_vals = []
        mean_pres_vals = []
        jitter_vals = []
        arm_lens = []
        arm_abs_errs = []
        arm_rel_errs = []
        has_pose_flags = []
        prev_world = None
        processed = 0
        wall_t0 = time.perf_counter()

        result_iter = iter_inferencer_results(
            inferencer=inferencer,
            video_path=video,
            device=args.device,
            bbox_thr=args.bbox_thr,
            num_instances=args.num_instances,
            rebase_keypoint_height=args.rebase_keypoint_height,
        )

        for raw_i, result in enumerate(tqdm(result_iter, desc=f"{video.name}")):
            if args.stride > 1 and (raw_i % args.stride != 0):
                continue
            processed += 1
            if args.max_frames > 0 and processed > args.max_frames:
                break

            frame_idx = raw_i
            timestamp_ms = int(round(frame_idx * 1000.0 / fps_src))

            # Result dict typically contains predictions and visualization
            predictions = result.get("predictions") if isinstance(result, dict) else list_from_obj(result, "predictions")
            person = choose_first_person_prediction(predictions)
            has_pose = person is not None
            has_pose_flags.append(int(has_pose))

            infer_ms = float("nan")
            mean_vis = float("nan")
            mean_pres = float("nan")
            jitter_frame = float("nan")
            arm_len = float("nan")
            arm_abs_err = float("nan")
            arm_rel_err = float("nan")

            if has_pose:
                k2d = list_from_obj(person, "keypoints")
                s2d = list_from_obj(person, "keypoint_scores")
                k3d = list_from_obj(person, "keypoints_3d")
                s3d = list_from_obj(person, "keypoint_scores")
                vis3d = list_from_obj(person, "visibility")
                pres3d = list_from_obj(person, "presence")
                # Optional inference timing from returned object if available
                pred_time = list_from_obj(person, "inference_time")
                if pred_time is not None:
                    try:
                        infer_ms = float(pred_time)
                    except Exception:
                        pass

                arr2d = to_numpy_points(k2d) if k2d is not None else np.empty((0, 2))
                scores2d = np.asarray(s2d, dtype=np.float64) if s2d is not None else None
                unified2d, vis2d_u, pres2d_u = map_2d_to_unified(arr2d, scores2d, args.kpt_thr)

                arr3d = to_numpy_points(k3d) if k3d is not None else np.empty((0, 3))
                scores3d = np.asarray(s3d, dtype=np.float64) if s3d is not None else None
                vis3d = np.asarray(vis3d, dtype=np.float64) if vis3d is not None else None
                pres3d = np.asarray(pres3d, dtype=np.float64) if pres3d is not None else None
                unified3d, vis3d_u, pres3d_u = map_3d_to_unified(arr3d, scores3d, vis3d, pres3d, args.kpt_thr)

                valid_idxs = [i for i in range(NUM_LM) if np.isfinite(unified3d[i]).all() or np.isfinite(unified2d[i, :2]).all()]
                if valid_idxs:
                    mean_vis = float(np.mean(vis3d_u[valid_idxs]))
                    mean_pres = float(np.mean(pres3d_u[valid_idxs]))
                    mean_vis_vals.append(mean_vis)
                    mean_pres_vals.append(mean_pres)

                for i in valid_idxs:
                    if np.isfinite(unified2d[i, :2]).all():
                        lm2d_w.writerow([
                            frame_idx, timestamp_ms, i, LANDMARK_NAMES[i],
                            unified2d[i, 0], unified2d[i, 1], unified2d[i, 2],
                            vis2d_u[i], pres2d_u[i],
                        ])
                    if np.isfinite(unified3d[i]).all():
                        lm3d_w.writerow([
                            frame_idx, timestamp_ms, i, LANDMARK_NAMES[i],
                            unified3d[i, 0], unified3d[i, 1], unified3d[i, 2],
                            vis3d_u[i], pres3d_u[i],
                        ])

                arm_len, arm_abs_err, arm_rel_err = upper_arm_metrics(unified3d, args.arm_len_ref_m)
                if np.isfinite(arm_len):
                    arm_lens.append(arm_len)
                    arm_abs_errs.append(arm_abs_err)
                    arm_rel_errs.append(arm_rel_err)
                bones_w.writerow([frame_idx, timestamp_ms, args.arm_len_ref_m, arm_len, arm_abs_err, arm_rel_err])

                if prev_world is not None:
                    stable_valid = [idx for idx in STABLE_IDXS if np.isfinite(unified3d[idx]).all() and np.isfinite(prev_world[idx]).all()]
                    if stable_valid:
                        diffs = unified3d[stable_valid] - prev_world[stable_valid]
                        jitter_frame = float(np.mean(np.linalg.norm(diffs, axis=1)))
                        jitter_vals.append(jitter_frame)
                prev_world = unified3d

            if not np.isfinite(infer_ms):
                # Fallback: estimate from wall-clock per yielded frame
                # not exact, but keeps frame_metrics complete
                infer_ms = 0.0
            inference_times.append(infer_ms)

            frame_w.writerow([
                frame_idx, timestamp_ms, int(has_pose), infer_ms,
                mean_vis, mean_pres, jitter_frame,
                arm_len, arm_abs_err, arm_rel_err,
            ])

        wall_s = max(1e-6, time.perf_counter() - wall_t0)

        frame_csv.close()
        lm2d_csv.close()
        lm3d_csv.close()
        bones_csv.close()

        summary = VideoRunSummary(
            set_name=out_root.name,
            method="mmpose_3d",
            video_or_session=video.name,
            model=args.pose3d_alias,
            mode="VIDEO",
            delegate=args.device.upper(),
            frames_total=frames_total,
            frames_processed=processed,
            effective_fps=float(processed / wall_s) if processed else 0.0,
            detection_rate=float(np.mean(has_pose_flags)) if has_pose_flags else 0.0,
            mean_visibility=float(np.mean(mean_vis_vals)) if mean_vis_vals else 0.0,
            mean_presence=float(np.mean(mean_pres_vals)) if mean_pres_vals else 0.0,
            jitter_world_m_mean=float(np.mean(jitter_vals)) if jitter_vals else 0.0,
            jitter_world_m_p95=percentile(jitter_vals, 95),
            arm_len_ref_m=float(args.arm_len_ref_m),
            left_upper_arm_len_m_mean=float(np.mean(arm_lens)) if arm_lens else 0.0,
            left_upper_arm_len_m_std=float(np.std(arm_lens)) if arm_lens else 0.0,
            left_upper_arm_abs_error_m_mean=float(np.mean(arm_abs_errs)) if arm_abs_errs else 0.0,
            left_upper_arm_abs_error_m_p95=percentile(arm_abs_errs, 95),
            left_upper_arm_rel_error_mean_pct=float(np.mean(arm_rel_errs)) if arm_rel_errs else 0.0,
        )
        (leaf / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    write_global_summary(out_root, summaries)
    print(f"DONE. Results in: {out_root}")

if __name__ == "__main__":
    main()
