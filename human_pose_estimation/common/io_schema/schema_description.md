# Common I/O schema for human_pose_estimation

This folder documents the unified schema used by all algorithms.

## Leaf output directory
Each algorithm should write one result leaf directory like:

```text
runs/run/<video_or_session>/<model>/<mode>/<delegate>/
```

Inside this leaf directory, the required files are:

- `frame_metrics.csv`
- `landmarks_2d.csv`
- `landmarks_3d_world.csv`
- `bones_metrics.csv`
- `summary.json`

## Global run directory
Each run directory should also contain:

- `summary.csv`
- `summary.json`

## Required `frame_metrics.csv` columns
- `frame_idx`
- `timestamp_ms`
- `has_pose`
- `inference_ms`
- `mean_visibility`
- `mean_presence`
- `jitter_frame_m`
- `left_upper_arm_len_m`
- `left_upper_arm_abs_error_m`
- `left_upper_arm_rel_error_pct`

## Required `landmarks_2d.csv` columns
- `frame_idx`
- `timestamp_ms`
- `landmark_idx`
- `landmark_name`
- `x`
- `y`
- `z`
- `visibility`
- `presence`

For stereo methods, `camera_id` may be added.

## Required `landmarks_3d_world.csv` columns
- `frame_idx`
- `timestamp_ms`
- `landmark_idx`
- `landmark_name`
- `x_m`
- `y_m`
- `z_m`
- `visibility`
- `presence`

## Required `bones_metrics.csv` columns
- `frame_idx`
- `timestamp_ms`
- `arm_len_ref_m`
- `left_upper_arm_len_m`
- `left_upper_arm_abs_error_m`
- `left_upper_arm_rel_error_pct`

## Required global summary columns
- `set_name`
- `method`
- `video_or_session`
- `model`
- `mode`
- `delegate`
- `frames_total`
- `frames_processed`
- `effective_fps`
- `detection_rate`
- `mean_visibility`
- `mean_presence`
- `jitter_world_m_mean`
- `jitter_world_m_p95`
- `arm_len_ref_m`
- `left_upper_arm_len_m_mean`
- `left_upper_arm_len_m_std`
- `left_upper_arm_abs_error_m_mean`
- `left_upper_arm_abs_error_m_p95`
- `left_upper_arm_rel_error_mean_pct`
