# mmpose_3d

Шаблон проекта `human_pose_estimation/single_camera/mmpose_3d`.

## Назначение
Single-camera 3D human pose estimation через пайплайн MMPose:
- 2D keypoint detection
- 2D-to-3D lifting
- экспорт в единый формат метрик

## Структура
- `configs/` — настройки проекта
- `models/` — чекпоинты detector / 2D / 3D
- `scripts/benchmark.py` — основной запуск
- `runs/run/` — результаты в общем schema
- `runs/compare_out/` — графики

## Вход
Видео:
`../../videos/single_camera`

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
Сейчас это подготовленный проектный шаблон.
Нужно дописать реальный inference MMPose и экспорт в общую схему.

## Базовые команды
```bash
python scripts/download_models.py
python scripts/benchmark.py
python scripts/analyze_results.py
```
