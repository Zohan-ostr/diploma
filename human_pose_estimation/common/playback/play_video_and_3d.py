#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}
POSE_CONNECTIONS = [
    (LM["left_shoulder"], LM["right_shoulder"]), (LM["left_shoulder"], LM["left_hip"]), (LM["right_shoulder"], LM["right_hip"]),
    (LM["left_hip"], LM["right_hip"]), (LM["left_shoulder"], LM["left_elbow"]), (LM["left_elbow"], LM["left_wrist"]),
    (LM["right_shoulder"], LM["right_elbow"]), (LM["right_elbow"], LM["right_wrist"]),
]
HIP_IDXS = [LM["left_hip"], LM["right_hip"]]

def load_world_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    frame_ids = sorted({int(r["frame_idx"]) for r in rows})
    landmark_ids = sorted({int(r["landmark_idx"]) for r in rows})
    fi_to_i = {fi: i for i, fi in enumerate(frame_ids)}
    li_to_j = {li: j for j, li in enumerate(landmark_ids)}
    ts_ms = np.zeros(len(frame_ids), dtype=np.int64)
    xyz = np.full((len(frame_ids), len(landmark_ids), 3), np.nan)
    vis = np.full((len(frame_ids), len(landmark_ids)), np.nan)
    pres = np.full((len(frame_ids), len(landmark_ids)), np.nan)
    for r in rows:
        i = fi_to_i[int(r["frame_idx"])]
        j = li_to_j[int(r["landmark_idx"])]
        ts_ms[i] = int(r["timestamp_ms"])
        xk = "x_m" if "x_m" in r else "x"
        yk = "y_m" if "y_m" in r else "y"
        zk = "z_m" if "z_m" in r else "z"
        xyz[i, j] = [float(r[xk]), float(r[yk]), float(r[zk])]
        vis[i, j] = float(r.get("visibility", "nan"))
        pres[i, j] = float(r.get("presence", "nan"))
    return np.asarray(frame_ids), ts_ms, np.asarray(landmark_ids), xyz, vis, pres

def play_video_and_3d(video_path: Path, csv_path: Path):
    frame_ids, ts_ms, landmark_ids, coords, vis, pres = load_world_csv(csv_path)
    li_to_j = {li: j for j, li in enumerate(landmark_ids.tolist())}
    coords_draw = coords.copy()

    for t in range(coords_draw.shape[0]):
        hips = []
        for h in HIP_IDXS:
            if h in li_to_j:
                p = coords_draw[t, li_to_j[h]]
                if np.isfinite(p).all():
                    hips.append(p)
        if hips:
            center = np.mean(np.asarray(hips), axis=0)
            coords_draw[t] -= center

    pts_all = coords_draw.reshape(-1, 3)
    pts_all = pts_all[np.isfinite(pts_all).all(axis=1)]
    mins = pts_all.min(axis=0); maxs = pts_all.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.15 * span
    mins -= pad; maxs += pad

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[0]))
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(15, 7))
    ax_video = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_video.axis("off")
    ax_video.set_title(f"Video: {video_path.stem}")
    ax_3d.set_title("3D playback")
    ax_3d.set_xlim(mins[0], maxs[0]); ax_3d.set_ylim(mins[1], maxs[1]); ax_3d.set_zlim(mins[2], maxs[2])
    ax_3d.text2D(0.02, 0.08, "blue = visibility", transform=ax_3d.transAxes, color="blue", fontsize=9)
    ax_3d.text2D(0.02, 0.04, "red = presence", transform=ax_3d.transAxes, color="red", fontsize=9)

    img_artist = ax_video.imshow(frame_rgb)
    scat = ax_3d.scatter([], [], [], s=20)
    lines = []
    for a, b in POSE_CONNECTIONS:
        ln, = ax_3d.plot([], [], [], linewidth=2)
        lines.append((ln, a, b))
    vis_texts = []
    pres_texts = []
    info_text = ax_3d.text2D(0.02, 0.98, "", transform=ax_3d.transAxes, va="top")

    dts = np.diff(ts_ms.astype(np.float64)) / 1000.0
    if len(dts) == 0:
        dts = np.array([1/30], dtype=np.float64)
    delays = np.concatenate([dts, [dts[-1]]], axis=0)
    t_prev = None

    def get_video_frame(frame_number: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frm = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    def update(k: int):
        nonlocal t_prev
        if t_prev is None:
            t_prev = time.perf_counter()
        else:
            now = time.perf_counter()
            elapsed = now - t_prev
            target = delays[k]
            if elapsed < target:
                time.sleep(target - elapsed)
            t_prev = time.perf_counter()

        frm = get_video_frame(frame_ids[k])
        if frm is not None:
            img_artist.set_data(frm)

        pts = coords_draw[k]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        for txt in vis_texts:
            txt.remove()
        for txt in pres_texts:
            txt.remove()
        vis_texts.clear(); pres_texts.clear()

        for ln, a, b in lines:
            if a in li_to_j and b in li_to_j:
                pa = pts[li_to_j[a]]
                pb = pts[li_to_j[b]]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    ln.set_3d_properties([pa[2], pb[2]])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])

        for li, j in li_to_j.items():
            p = pts[j]
            if not np.isfinite(p).all():
                continue
            vis_texts.append(ax_3d.text(p[0] - 0.015, p[1], p[2], f"v:{vis[k,j]:.2f}", fontsize=6, color="blue", ha="right"))
            pres_texts.append(ax_3d.text(p[0] + 0.015, p[1], p[2], f"p:{pres[k,j]:.2f}", fontsize=6, color="red", ha="left"))

        info_text.set_text(f"frame={int(frame_ids[k])} | t={ts_ms[k]} ms")
        return [img_artist, scat, info_text] + [ln for ln, _, _ in lines] + vis_texts + pres_texts

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=1, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
    cap.release()

if __name__ == "__main__":
    print("Import this helper and call play_video_and_3d(video_path, csv_path).")
