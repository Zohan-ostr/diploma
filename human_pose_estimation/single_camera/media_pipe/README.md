# media_pipe

Переработанный проект MediaPipe под единый стандарт для `human_pose_estimation/single_camera/media_pipe`.

## Вход
Видео лежат в:
`../../videos/single_camera`

Модели лежат в:
`./models`

## Основные команды

Скачать модели:
```bash
python scripts/download_models.py
```

Полный прогон:
```bash
python scripts/benchmark.py
```

Оптимизированный прогон:
```bash
python scripts/benchmark_opt.py
```

Анализ:
```bash
python scripts/analyze_results.py
```

3D playback:
```bash
python scripts/playback_3d.py
```

## Основные метрики
- effective_fps
- detection_rate
- mean_visibility
- mean_presence
- jitter_world_m_mean
- jitter_world_m_p95
- left_upper_arm_len_m_mean
- left_upper_arm_abs_error_m_mean
- left_upper_arm_rel_error_mean_pct

## arm_len_ref_m
Это эталонная длина руки от `left_shoulder` до `left_elbow`, задаётся в `configs/default.yaml`.
