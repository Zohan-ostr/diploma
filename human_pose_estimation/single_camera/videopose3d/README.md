# videopose3d

Шаблон проекта `human_pose_estimation/single_camera/videopose3d`.

## Назначение
Single-camera 3D human pose estimation через двухэтапный пайплайн:
1. извлечение 2D keypoints
2. temporal lifting в 3D через VideoPose3D
3. экспорт в единый формат метрик

## Структура
- `configs/` — настройки проекта
- `models/` — чекпоинты VideoPose3D и/или 2D backend
- `input_2d/` — промежуточные 2D keypoints
- `output_3d/` — промежуточные 3D результаты lifting
- `scripts/extract_2d_keypoints.py` — 2D этап
- `scripts/run_lifting.py` — 3D lifting
- `scripts/benchmark.py` — общий запуск / экспорт summary
- `runs/run/` — результаты в общем schema

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
Это подготовленный проектный шаблон.
Нужно дописать:
- реальный 2D backend
- реальный запуск VideoPose3D
- экспорт результатов в общий формат

## Базовые команды
```bash
python scripts/download_models.py
python scripts/extract_2d_keypoints.py
python scripts/run_lifting.py
python scripts/benchmark.py
python scripts/analyze_results.py
```
