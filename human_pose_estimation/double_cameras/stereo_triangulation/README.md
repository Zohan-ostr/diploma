# stereo_triangulation

Шаблон проекта `human_pose_estimation/double_cameras/stereo_triangulation`.

## Назначение
Универсальный двухкамерный пайплайн:
1. 2D pose на левой камере
2. 2D pose на правой камере
3. триангуляция по откалиброванным камерам
4. экспорт в единый формат метрик

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

## Почему этот проект важен
Это самый универсальный stereo-подход для твоей ВКР.
Можно менять только 2D backend, а сама геометрическая часть остаётся общей.

## Текущее состояние
Это проектный шаблон.
Нужно дописать:
- левый 2D pipeline
- правый 2D pipeline
- triangulation backend
- экспорт в common schema

## Базовые шаги
```bash
python scripts/prepare_sessions.py
python scripts/run_pose2d_left.py
python scripts/run_pose2d_right.py
python scripts/run_triangulation.py
python scripts/benchmark.py
python scripts/analyze_results.py
```
