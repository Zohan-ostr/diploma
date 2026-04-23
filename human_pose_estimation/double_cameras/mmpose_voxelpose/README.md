# mmpose_voxelpose

Шаблон проекта `human_pose_estimation/double_cameras/mmpose_voxelpose`.

## Назначение
Multi-view 3D human pose estimation через MMPose / VoxelPose для двух откалиброванных камер.

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

## Почему этот проект нужен
Это современный multi-view 3D baseline внутри экосистемы OpenMMLab.
Он полезен как исследовательский контраст к более простому stereo_triangulation.

## Текущее состояние
Это проектный шаблон.
Нужно дописать:
- конвертацию стереосессий в формат VoxelPose
- запуск реального VoxelPose inference
- экспорт в common schema

## Базовые шаги
```bash
python scripts/download_models.py
python scripts/prepare_sessions.py
python scripts/convert_sessions_to_mmpose_format.py
python scripts/benchmark.py
python scripts/analyze_results.py
```
