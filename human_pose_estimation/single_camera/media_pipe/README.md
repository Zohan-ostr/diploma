# media_pipe (reworked)

Reworked MediaPipe package for the new project standard:

- `configs/` – configuration placeholders
- `models/` – MediaPipe `.task` models
- `scripts/benchmark.py` – baseline run
- `scripts/benchmark_opt.py` – optimized torso+arms run
- `scripts/analyze_results.py` – result analysis
- `scripts/playback_3d.py` – 3D playback
- `runs/` – outputs (`run`, `run_opt`, `compare_out`)

## Expected video location

The scripts are configured for videos under:

`../../videos/single_camera`

relative to this folder.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/benchmark.py
python scripts/benchmark_opt.py
python scripts/analyze_results.py
python scripts/playback_3d.py
```

## Note

This archive is a migration scaffold for the `media_pipe` part of the new
`human_pose_estimation/single_camera` structure.
