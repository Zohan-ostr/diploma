# mmpose_3d

Рабочая папка `human_pose_estimation/single_camera/mmpose_3d`.

## Что делает проект
Проект использует MMPose unified inferencer для 3D human pose estimation из одного входного видео и сохраняет результаты в общий формат проекта:
- `frame_metrics.csv`
- `landmarks_2d.csv`
- `landmarks_3d_world.csv`
- `bones_metrics.csv`
- `summary.json`
- `summary.csv`

## Вход
Видео:
`../../videos/single_camera`

## Выход
Результаты:
`runs/run/<video>/<model>/<mode>/<delegate>/`

## arm_len_ref_m
Эталонное расстояние `left_shoulder -> left_elbow`.

## Команды

Подготовка:
```bash
python scripts/download_models.py
```

Основной прогон:
```bash
python scripts/benchmark.py
```

Анализ:
```bash
python scripts/analyze_results.py
```

Playback:
```bash
python scripts/playback_3d.py
```

## Замечания
По умолчанию используется alias `human3d`.
Если окружение MMPose настроено корректно, inferencer может автоматически загрузить нужные checkpoint-файлы при первом запуске.
