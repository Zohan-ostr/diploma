#!/usr/bin/env python3
from __future__ import annotations
import csv, time
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
LANDMARK_NAMES = ["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]
LM = {name:i for i,name in enumerate(LANDMARK_NAMES)}
POSE_CONNECTIONS = [(LM['left_shoulder'],LM['right_shoulder']),(LM['left_shoulder'],LM['left_hip']),(LM['right_shoulder'],LM['right_hip']),(LM['left_hip'],LM['right_hip']),(LM['left_shoulder'],LM['left_elbow']),(LM['left_elbow'],LM['left_wrist']),(LM['left_wrist'],LM['left_thumb']),(LM['left_wrist'],LM['left_index']),(LM['left_wrist'],LM['left_pinky']),(LM['right_shoulder'],LM['right_elbow']),(LM['right_elbow'],LM['right_wrist']),(LM['right_wrist'],LM['right_thumb']),(LM['right_wrist'],LM['right_index']),(LM['right_wrist'],LM['right_pinky'])]
HIP_IDXS = [LM['left_hip'], LM['right_hip']]
def choose_from_list(items: List[Path], title: str) -> Path:
    print(f'
{title}')
    for i,item in enumerate(items, start=1): print(f'{i}) {item}')
    while True:
        s = input('Choose number: ').strip()
        if s.isdigit() and 1 <= int(s) <= len(items): return items[int(s)-1]
def list_sets(runs_root: Path) -> List[Path]: return sorted([p for p in runs_root.iterdir() if p.is_dir()])
def list_leaf_dirs(run_set: Path) -> List[Path]: return sorted(set([p.parent for p in run_set.rglob('landmarks_3d_world*.csv')]))
def tags_from_leaf(leaf_dir: Path, set_root: Path):
    rel = leaf_dir.relative_to(set_root); parts = rel.parts; return parts[0], parts[1], parts[2], parts[3]
def load_world_csv(path: Path):
    with path.open('r', newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
    frame_ids_sorted = sorted({int(r['frame_idx']) for r in rows}); landmark_ids_sorted = sorted({int(r['landmark_idx']) for r in rows})
    fi_to_idx = {fi:i for i,fi in enumerate(frame_ids_sorted)}; li_to_idx = {li:j for j,li in enumerate(landmark_ids_sorted)}
    ts_ms = np.zeros(len(frame_ids_sorted), dtype=np.int64); coords = np.full((len(frame_ids_sorted), len(landmark_ids_sorted), 3), np.nan); vis = np.full((len(frame_ids_sorted), len(landmark_ids_sorted)), np.nan); pres = np.full((len(frame_ids_sorted), len(landmark_ids_sorted)), np.nan)
    for r in rows:
        i = fi_to_idx[int(r['frame_idx'])]; j = li_to_idx[int(r['landmark_idx'])]; ts_ms[i] = int(r['timestamp_ms'])
        xk='x_m' if 'x_m' in r else 'x'; yk='y_m' if 'y_m' in r else 'y'; zk='z_m' if 'z_m' in r else 'z'
        coords[i,j,0]=float(r[xk]); coords[i,j,1]=float(r[yk]); coords[i,j,2]=float(r[zk]); vis[i,j]=float(r.get('visibility','nan')); pres[i,j]=float(r.get('presence','nan'))
    return np.array(frame_ids_sorted), ts_ms, np.array(landmark_ids_sorted), coords, vis, pres
def find_matching_video(video_name: str, videos_root: Path) -> Path:
    stem = Path(video_name).stem; candidates = list(videos_root.glob(f'{stem}.*'))
    if not candidates: raise RuntimeError(f'Video not found for {stem}')
    return candidates[0]
def main():
    project_root = Path(__file__).resolve().parents[1]; runs_root = project_root / 'runs'; videos_root = (project_root / '../../videos/single_camera').resolve()
    run_set = choose_from_list(list_sets(runs_root), 'Choose run set'); leaf = choose_from_list(list_leaf_dirs(run_set), 'Choose result dir')
    video_name, model, mode, delegate = tags_from_leaf(leaf, run_set); csv_path = sorted(leaf.glob('landmarks_3d_world*.csv'))[0]
    frame_ids, ts_ms, landmark_ids, coords, vis, pres = load_world_csv(csv_path); li_to_j = {li:j for j,li in enumerate(landmark_ids.tolist())}
    coords_draw = coords.copy()
    for t in range(coords_draw.shape[0]):
        hips=[]
        for h in HIP_IDXS:
            if h in li_to_j:
                p = coords_draw[t, li_to_j[h]]
                if np.isfinite(p).all(): hips.append(p)
        if hips: coords_draw[t] -= np.mean(np.asarray(hips), axis=0)
    pts_all = coords_draw.reshape(-1,3); pts_all = pts_all[np.isfinite(pts_all).all(axis=1)]
    mins = pts_all.min(axis=0); maxs = pts_all.max(axis=0); span = np.maximum(maxs-mins,1e-6); pad = 0.15*span; mins -= pad; maxs += pad
    video_path = find_matching_video(video_name, videos_root); cap = cv2.VideoCapture(str(video_path)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[0])); ok, frame_bgr = cap.read(); frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(15,7)); ax_video = fig.add_subplot(1,2,1); ax_3d = fig.add_subplot(1,2,2, projection='3d'); ax_video.axis('off'); ax_video.set_title(f'Video: {Path(video_name).stem}')
    ax_3d.set_title(f"3D: {model.replace('pose_landmarker_','')}"); ax_3d.set_xlim(mins[0], maxs[0]); ax_3d.set_ylim(mins[1], maxs[1]); ax_3d.set_zlim(mins[2], maxs[2])
    ax_3d.text2D(0.02,0.08,'blue = visibility', transform=ax_3d.transAxes, color='blue', fontsize=9); ax_3d.text2D(0.02,0.04,'red = presence', transform=ax_3d.transAxes, color='red', fontsize=9)
    img_artist = ax_video.imshow(frame_rgb); scat = ax_3d.scatter([],[],[],s=20); line_artists=[]
    for a,b in POSE_CONNECTIONS:
        ln, = ax_3d.plot([],[],[],linewidth=2); line_artists.append((ln,a,b))
    vis_texts=[]; pres_texts=[]; info_text = ax_3d.text2D(0.02,0.98,'', transform=ax_3d.transAxes, va='top')
    dts = np.diff(ts_ms.astype(np.float64))/1000.0; dts = dts if len(dts) else np.array([1/30], dtype=np.float64); frame_delays = np.concatenate([dts,[dts[-1]]], axis=0); t_wall_prev = None
    def get_video_frame(frame_number: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number); ok, frm = cap.read(); return cv2.cvtColor(frm, cv2.COLOR_BGR2RGB) if ok else None
    def update(k:int):
        nonlocal t_wall_prev
        if t_wall_prev is None: t_wall_prev = time.perf_counter()
        else:
            target = frame_delays[k]; now = time.perf_counter(); elapsed = now - t_wall_prev
            if elapsed < target: time.sleep(target-elapsed)
            t_wall_prev = time.perf_counter()
        frm = get_video_frame(int(frame_ids[k]));
        if frm is not None: img_artist.set_data(frm)
        pts = coords_draw[k]; scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        for txt in vis_texts: txt.remove()
        for txt in pres_texts: txt.remove()
        vis_texts.clear(); pres_texts.clear()
        for ln,a,b in line_artists:
            if a in li_to_j and b in li_to_j:
                pa = pts[li_to_j[a]]; pb = pts[li_to_j[b]]
                if np.isfinite(pa).all() and np.isfinite(pb).all(): ln.set_data([pa[0],pb[0]],[pa[1],pb[1]]); ln.set_3d_properties([pa[2],pb[2]])
                else: ln.set_data([],[]); ln.set_3d_properties([])
        for li,j in li_to_j.items():
            p = pts[j]
            if not np.isfinite(p).all(): continue
            vis_texts.append(ax_3d.text(p[0]-0.015,p[1],p[2], f"v:{vis[k,j]:.2f}", fontsize=6, color='blue', ha='right'))
            pres_texts.append(ax_3d.text(p[0]+0.015,p[1],p[2], f"p:{pres[k,j]:.2f}", fontsize=6, color='red', ha='left'))
        info_text.set_text(f"frame={int(frame_ids[k])} | t={ts_ms[k]} ms
video={Path(video_name).stem} | model={model.replace('pose_landmarker_','')}")
        return [img_artist, scat, info_text] + [ln for ln,_,_ in line_artists] + vis_texts + pres_texts
    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=1, blit=False, repeat=False); plt.tight_layout(); plt.show(); cap.release()
if __name__ == '__main__': main()
