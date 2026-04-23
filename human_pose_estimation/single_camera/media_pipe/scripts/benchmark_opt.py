#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark import run_on_video, load_yaml_config, LM, list_videos_in_dir, find_models, safe_mkdir, write_global_summary

def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml_config(project_root)
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos_dir', default=cfg.get('videos_dir', '../../videos/single_camera'))
    ap.add_argument('--model_dir', default=cfg.get('models_dir', 'models'))
    ap.add_argument('--out_dir', default=str(project_root / 'runs' / cfg.get('run_names', {}).get('optimized', 'run_opt')))
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
    keep_names = cfg.get('optimized', {}).get('keep_landmarks', [])
    keep_idxs = [LM[n] for n in keep_names]
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
                                          args.min_tracking_confidence, args.arm_len_ref_m, keep_idxs=keep_idxs, use_keep_for_metrics=True,
                                          lm2d_name='landmarks_2d_torso_arms.csv', lm3d_name='landmarks_3d_world_torso_arms.csv'))
    write_global_summary(out_root, summaries)
    print(f'DONE. Results in: {out_root}')
if __name__ == '__main__':
    main()
