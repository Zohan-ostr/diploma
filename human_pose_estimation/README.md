# Human Pose Estimation

Эта папка является основной рабочей директорией для всех экспериментов по оценке позы человека в рамках дипломного проекта.

---

## Структура папки

```text
human_pose_estimation/
├── videos/
│   ├── single_camera/
│   └── double_cameras/
├── common/
│   ├── io_schema/
│   ├── metrics/
│   ├── playback/
│   ├── compare/
│   └── calibration/
├── docker/
├── single_camera/
│   ├── media_pipe/
│   ├── mmpose_3d/
│   └── videopose3d/
└── double_cameras/
    ├── openpose_3d/
    ├── stereo_triangulation/
    └── mmpose_voxelpose/
```

---

## Назначение папок

### `videos/`
Хранит все входные данные.

#### `videos/single_camera/`
Видео для алгоритмов, работающих с одной камерой.

#### `videos/double_cameras/`
Стереосессии для алгоритмов, работающих с двумя откалиброванными камерами.

---

### `common/`
Общие модули для всех алгоритмов:

---

### `docker/`
Docker-конфигурация для запуска алгоритмов в контейнерах.

---

### `single_camera/`
Проекты алгоритмов, работающих с одной камерой:
- `media_pipe`
- `mmpose_3d`
- `videopose3d`

---

### `double_cameras/`
Проекты алгоритмов, работающих с двумя камерами:
- `openpose_3d`
- `stereo_triangulation`
- `mmpose_voxelpose`

---

## Общий принцип работы

---

## Docker

### Сборка контейнера

Из папки `human_pose_estimation`:

```bash
docker compose -f docker/docker-compose.yml build
```

### Запуск основного контейнера

```bash
docker compose -f docker/docker-compose.yml run --rm hpe_runtime bash
```

### Запуск контейнера OpenPose

Если нужен отдельный контейнер под OpenPose:

```bash
docker compose -f docker/docker-compose.yml --profile openpose run --rm openpose_runtime bash
```

---

## Работа с single-camera алгоритмами

### MediaPipe

```bash
cd /workspace/single_camera/media_pipe
python scripts/download_models.py
python scripts/benchmark.py
python scripts/benchmark_opt.py
python scripts/analyze_results.py
python scripts/playback_3d.py
```

### MMPose 3D

```bash
cd /workspace/single_camera/mmpose_3d
python scripts/download_models.py
python scripts/benchmark.py
python scripts/analyze_results.py
```

### VideoPose3D

```bash
cd /workspace/single_camera/videopose3d
python scripts/download_models.py
python scripts/extract_2d_keypoints.py
python scripts/run_lifting.py
python scripts/benchmark.py
python scripts/analyze_results.py
```

---

## Работа с double-camera алгоритмами

### Общий порядок

Для двухкамерных алгоритмов сначала нужно:
1. подготовить сессию в `videos/double_cameras/`;
2. проверить, что есть `left.mp4`, `right.mp4`, `calib/*`, `meta.yaml`;
3. запустить `prepare_sessions.py`;
4. затем запускать основной pipeline метода.

---

### OpenPose 3D

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/download_models.py
python scripts/prepare_sessions.py
python scripts/benchmark.py
python scripts/analyze_results.py
```

---

### Stereo triangulation

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/prepare_sessions.py
python scripts/run_pose2d_left.py
python scripts/run_pose2d_right.py
python scripts/run_triangulation.py
python scripts/benchmark.py
python scripts/analyze_results.py
```

---

### MMPose VoxelPose

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/download_models.py
python scripts/prepare_sessions.py
python scripts/convert_sessions_to_mmpose_format.py
python scripts/benchmark.py
python scripts/analyze_results.py
```

---

## Единый формат результатов

Каждый алгоритм должен сохранять результаты в своей папке `runs/`.

---

## Основные метрики

Для всех алгоритмов используется единый набор метрик:

- `effective_fps`
- `detection_rate`
- `mean_visibility`
- `mean_presence`
- `jitter_world_m_mean`
- `jitter_world_m_p95`

Дополнительно для всех методов используется метрика длины руки:

- `arm_len_ref_m` — эталонное расстояние от `left_shoulder` до `left_elbow`, измеренное вручную;
- `left_upper_arm_len_m`
- `left_upper_arm_abs_error_m`
- `left_upper_arm_rel_error_pct`

---

## Сравнение методов

Для сравнения результатов всех методов используются утилиты из:

```text
common/compare/
```

После того как несколько алгоритмов сформировали свои `summary.csv`, можно запускать общий сравнительный анализ.

---

## Playback

Общие модули воспроизведения лежат в:

```text
common/playback/
```

Playback используется для синхронного просмотра:
- исходного видео;
- 2D/3D представления позы;
- служебных метрик `visibility` и `presence`.

---

## Калибровка для двух камер

Общие калибровочные утилиты лежат в:

```text
common/calibration/
```

Шаблоны файлов калибровки также дублируются внутри stereo-проектов.

Минимально нужны:
- `intrinsics_left.yaml`
- `intrinsics_right.yaml`
- `extrinsics.yaml`
- `stereo.yaml`

---
