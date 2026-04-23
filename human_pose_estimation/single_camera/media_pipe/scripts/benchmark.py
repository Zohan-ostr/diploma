
from __future__ import annotations
import csv, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import cv2, numpy as np, yaml, mediapipe as mp
LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}
NUM_LM = len(LANDMARK_NAMES)
STABLE_IDXS = [LM["left_shoulder"], LM["right_shoulder"], LM["left_hip"], LM["right_hip"]]
UPPER_ARM_IDXS = (LM["left_shoulder"], LM["left_elbow"])

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

def safe_mkdir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def percentile(values: Sequence[float], q: float) -> float:
    if not values: return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
def list_videos_in_dir(videos_dir: Path) -> List[Path]:
    exts = {'.mp4','.mov','.mkv','.avi','.webm'}
    return sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
def find_models(model_dir: Path) -> List[Path]: return sorted(model_dir.glob('*.task'))
def mp_image_from_bgr(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
def create_landmarker(model_path: Path, delegate: str, num_poses: int, min_pose_detection_conf: float, min_pose_presence_conf: float, min_tracking_conf: float):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    d = BaseOptions.Delegate.CPU if delegate == 'CPU' else BaseOptions.Delegate.GPU
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=d),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_conf,
        min_pose_presence_confidence=min_pose_presence_conf,
        min_tracking_confidence=min_tracking_conf,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)
def load_yaml_config(project_root: Path) -> dict:
    cfg = {}
    dp = project_root / 'configs' / 'default.yaml'
    lp = project_root / 'configs' / 'local.yaml'
    if dp.exists(): cfg = yaml.safe_load(dp.read_text(encoding='utf-8')) or {}
    if lp.exists():
        local = yaml.safe_load(lp.read_text(encoding='utf-8')) or {}
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
                else: a[k] = v
        merge(cfg, local)
    return cfg
def upper_arm_metrics(world: np.ndarray, arm_len_ref_m: float):
    ls, le = UPPER_ARM_IDXS
    if not np.isfinite(world[[ls, le]]).all(): return float('nan'), float('nan'), float('nan')
    arm_len = float(np.linalg.norm(world[le] - world[ls]))
    abs_err = abs(arm_len - arm_len_ref_m)
    rel_err = (100.0 * abs_err / arm_len_ref_m) if arm_len_ref_m > 0 else float('nan')
    return arm_len, abs_err, rel_err
def write_global_summary(out_root: Path, summaries: List[VideoRunSummary]) -> None:
    if not summaries: return
    header = list(asdict(summaries[0]).keys())
    with (out_root / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header)
        for s in summaries:
            d = asdict(s); w.writerow([d[h] for h in header])
    (out_root / 'summary.json').write_text(json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2), encoding='utf-8')

#!/usr/bin/env python3
from __future__ import annotations
import argparse

def run_on_video(video_path: Path, model_path: Path, out_dir: Path, set_name: str, delegate: str='CPU', stride: int=1, max_frames: int=0, num_poses: int=1,
                 min_pose_detection_conf: float=0.5, min_pose_presence_conf: float=0.5, min_tracking_conf: float=0.5, arm_len_ref_m: float=0.249,
                 keep_idxs: Optional[List[int]]=None, use_keep_for_metrics: bool=False, lm2d_name: str='landmarks_2d.csv', lm3d_name: str='landmarks_3d_world.csv') -> VideoRunSummary:
    safe_mkdir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {video_path}')
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_f = (out_dir / 'frame_metrics.csv').open('w', newline='', encoding='utf-8')
    lm2d_f = (out_dir / lm2d_name).open('w', newline='', encoding='utf-8')
    lm3d_f = (out_dir / lm3d_name).open('w', newline='', encoding='utf-8')
    bones_f = (out_dir / 'bones_metrics.csv').open('w', newline='', encoding='utf-8')
    frame_w, lm2d_w, lm3d_w, bones_w = csv.writer(frame_f), csv.writer(lm2d_f), csv.writer(lm3d_f), csv.writer(bones_f)
    frame_w.writerow(['frame_idx','timestamp_ms','has_pose','inference_ms','mean_visibility','mean_presence','jitter_frame_m','left_upper_arm_len_m','left_upper_arm_abs_error_m','left_upper_arm_rel_error_pct'])
    lm2d_w.writerow(['frame_idx','timestamp_ms','landmark_idx','landmark_name','x','y','z','visibility','presence'])
    lm3d_w.writerow(['frame_idx','timestamp_ms','landmark_idx','landmark_name','x_m','y_m','z_m','visibility','presence'])
    bones_w.writerow(['frame_idx','timestamp_ms','arm_len_ref_m','left_upper_arm_len_m','left_upper_arm_abs_error_m','left_upper_arm_rel_error_pct'])
    inference_times=[]; vis_list=[]; pres_list=[]; jitter_vals=[]; arm_lens=[]; arm_abs_errs=[]; arm_rel_errs=[]; has_pose_count=0
    prev_world=None; prev_ts_ms=None; frame_idx=-1; processed=0; wall_t0=time.perf_counter()
    try:
        landmarker_ctx = create_landmarker(model_path, delegate, num_poses, min_pose_detection_conf, min_pose_presence_conf, min_tracking_conf)
    except NotImplementedError as ex:
        if delegate == 'GPU' and 'GPU processing is disabled in build flags' in str(ex):
            print('[warn] This mediapipe build has no GPU support. Falling back to CPU.')
            delegate = 'CPU'
            landmarker_ctx = create_landmarker(model_path, delegate, num_poses, min_pose_detection_conf, min_pose_presence_conf, min_tracking_conf)
        else:
            raise
    with landmarker_ctx as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0): continue
            processed += 1
            if max_frames > 0 and processed > max_frames: break
            ts_ms = int(round(frame_idx * 1000.0 / fps_src))
            mp_image = mp_image_from_bgr(frame_bgr)
            t0 = time.perf_counter(); result = landmarker.detect_for_video(mp_image, ts_ms); infer_ms = (time.perf_counter() - t0) * 1000.0
            inference_times.append(infer_ms)
            has_pose = bool(getattr(result, 'pose_landmarks', None)) and len(result.pose_landmarks) > 0
            mean_vis=mean_pres=jitter_frame=arm_len=arm_abs_err=arm_rel_err=float('nan')
            if has_pose:
                has_pose_count += 1
                lm2d = result.pose_landmarks[0]
                lm3d = result.pose_world_landmarks[0] if getattr(result, 'pose_world_landmarks', None) else None
                vis = np.array([float(getattr(lm,'visibility',0.0)) for lm in lm2d], dtype=np.float64)
                pres = np.array([float(getattr(lm,'presence',0.0)) for lm in lm2d], dtype=np.float64)
                metric_idxs = keep_idxs if (use_keep_for_metrics and keep_idxs is not None) else list(range(NUM_LM))
                mean_vis = float(np.mean(vis[metric_idxs])); mean_pres = float(np.mean(pres[metric_idxs]))
                vis_list.append(mean_vis); pres_list.append(mean_pres)
                save_idxs = keep_idxs if keep_idxs is not None else list(range(NUM_LM))
                for i in save_idxs:
                    lm = lm2d[i]
                    lm2d_w.writerow([frame_idx, ts_ms, i, LANDMARK_NAMES[i], float(lm.x), float(lm.y), float(lm.z), float(getattr(lm,'visibility',0.0)), float(getattr(lm,'presence',0.0))])
                if lm3d is not None:
                    world = np.full((NUM_LM,3), np.nan, dtype=np.float64)
                    for i in save_idxs:
                        lm = lm3d[i]
                        x,y,z = float(lm.x), float(lm.y), float(lm.z)
                        world[i] = (x,y,z)
                        lm3d_w.writerow([frame_idx, ts_ms, i, LANDMARK_NAMES[i], x, y, z, float(getattr(lm,'visibility',0.0)), float(getattr(lm,'presence',0.0))])
                    arm_len, arm_abs_err, arm_rel_err = upper_arm_metrics(world, arm_len_ref_m)
                    if np.isfinite(arm_len):
                        arm_lens.append(arm_len); arm_abs_errs.append(arm_abs_err); arm_rel_errs.append(arm_rel_err)
                    bones_w.writerow([frame_idx, ts_ms, arm_len_ref_m, arm_len, arm_abs_err, arm_rel_err])
                    if prev_world is not None and prev_ts_ms is not None:
                        stable_idxs = [idx for idx in STABLE_IDXS if np.isfinite(world[idx]).all() and np.isfinite(prev_world[idx]).all()]
                        if stable_idxs:
                            diffs = world[stable_idxs] - prev_world[stable_idxs]
                            jitter_frame = float(np.mean(np.linalg.norm(diffs, axis=1)))
                            jitter_vals.append(jitter_frame)
                    prev_world = world; prev_ts_ms = ts_ms
            frame_w.writerow([frame_idx, ts_ms, int(has_pose), float(infer_ms), mean_vis, mean_pres, jitter_frame, arm_len, arm_abs_err, arm_rel_err])
    wall_s = max(1e-6, time.perf_counter() - wall_t0)
    cap.release(); frame_f.close(); lm2d_f.close(); lm3d_f.close(); bones_f.close()
    summary = VideoRunSummary(
        set_name=set_name, method='media_pipe', video_or_session=video_path.name, model=model_path.stem, mode='VIDEO', delegate=delegate,
        frames_total=frames_total, frames_processed=processed, effective_fps=float(processed / wall_s) if processed else 0.0,
        detection_rate=float(has_pose_count / max(1, processed)), mean_visibility=float(np.mean(vis_list)) if vis_list else 0.0,
        mean_presence=float(np.mean(pres_list)) if pres_list else 0.0, jitter_world_m_mean=float(np.mean(jitter_vals)) if jitter_vals else 0.0,
        jitter_world_m_p95=percentile(jitter_vals, 95), arm_len_ref_m=float(arm_len_ref_m), left_upper_arm_len_m_mean=float(np.mean(arm_lens)) if arm_lens else 0.0,
        left_upper_arm_len_m_std=float(np.std(arm_lens)) if arm_lens else 0.0, left_upper_arm_abs_error_m_mean=float(np.mean(arm_abs_errs)) if arm_abs_errs else 0.0,
        left_upper_arm_abs_error_m_p95=percentile(arm_abs_errs, 95), left_upper_arm_rel_error_mean_pct=float(np.mean(arm_rel_errs)) if arm_rel_errs else 0.0)
    (out_dir / 'summary.json').write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml_config(project_root)
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos_dir', default=cfg.get('videos_dir', '../../videos/single_camera'))
    ap.add_argument('--model_dir', default=cfg.get('models_dir', 'models'))
    ap.add_argument('--out_dir', default=str(project_root / 'runs' / cfg.get('run_names', {}).get('baseline', 'run')))
    ap.add_argument('--delegate', choices=['CPU','GPU'], default=cfg.get('single_camera', {}).get('delegate', 'CPU'))
    ap.add_argument('--stride', type=int, default=int(cfg.get('single_camera', {}).get('stride', 1)))
    ap.add_argument('--max_frames', type=int, default=int(cfg.get('single_camera', {}).get('max_frames', 0)))
    ap.add_argument('--num_poses', type=int, default=int(cfg.get('single_camera', {}).get('num_poses', 1)))
    ap.add_argument('--min_pose_detection_confidence', type=float, default=float(cfg.get('single_camera', {}).get('min_pose_detection_confidence', 0.5)))
    ap.add_argument('--min_pose_presence_confidence', type=float, default=float(cfg.get('single_camera', {}).get('min_pose_presence_confidence', 0.5)))
    ap.add_argument('--min_tracking_confidence', type=float, default=float(cfg.get('single_camera', {}).get('min_tracking_confidence', 0.5)))
    ap.add_argument('--arm_len_ref_m', type=float, default=float(cfg.get('arm_len_ref_m', 0.249)))
    ap.add_argument('--models', nargs='*', default=[])
    args = ap.parse_args()
    videos_dir = (project_root / args.videos_dir).resolve() if not Path(args.videos_dir).is_absolute() else Path(args.videos_dir)
    model_dir = (project_root / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    out_root = Path(args.out_dir).resolve() if Path(args.out_dir).is_absolute() else (project_root / args.out_dir).resolve()
    safe_mkdir(out_root)
    videos = list_videos_in_dir(videos_dir)
    if not videos: raise RuntimeError(f'No videos found in {videos_dir}')
    models = find_models(model_dir)
    if args.models:
        wanted = set(args.models); models = [m for m in models if m.stem in wanted or m.name in wanted]
    if not models: raise RuntimeError(f'No .task models found in {model_dir}')
    summaries=[]
    for video in videos:
        for model_path in models:
            run_dir = out_root / video.stem / model_path.stem / 'VIDEO' / args.delegate
            summaries.append(run_on_video(video, model_path, run_dir, out_root.name, args.delegate, args.stride, args.max_frames,
                                          args.num_poses, args.min_pose_detection_confidence, args.min_pose_presence_confidence,
                                          args.min_tracking_confidence, args.arm_len_ref_m))
    write_global_summary(out_root, summaries)
    print(f'DONE. Results in: {out_root}')
if __name__ == '__main__':
    main()
