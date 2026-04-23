# openpose_3d

Шаблон проекта `human_pose_estimation/double_cameras/openpose_3d`.

## Назначение
3D human pose estimation по двум откалиброванным камерам через OpenPose 3D reconstruction.

## Вход
Стереосессии:
`../../videos/double_cameras`

Ожидаемый формат одной сессии:
- `left.mp4`
- `right.mp4`
- `calib/intrinsics_left.yaml`
- `calib/intrinsics_right.yaml`
- `calib/extrinsics.yaml`
- `calib/stereo.yaml`
- `meta.yaml`

## Выход
- `frame_metrics.csv`
- `landmarks_2d.csv`
- `landmarks_3d_world.csv`
- `bones_metrics.csv`
- `summary.json`
- `summary.csv`

## arm_len_ref_m
Эталонное расстояние `left_shoulder -> left_elbow`.

## Текущее состояние
Это проектный шаблон под общий стандарт.
Нужно дописать:
- запуск OpenPose 3D reconstruction
- парсинг результатов OpenPose
- экспорт в общий schema

## Полезные шаги
```bash
python scripts/download_models.py
python scripts/prepare_sessions.py
python scripts/benchmark.py
python scripts/analyze_results.py
```
