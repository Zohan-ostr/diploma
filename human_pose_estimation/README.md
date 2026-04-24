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

## Docker

### Где запускать Docker

Все Docker-команды запускаются **из папки `human_pose_estimation`**.

### Сборка контейнера

```bash
docker compose -f docker/docker-compose.yml build
```

### Запуск основного контейнера

```bash
docker compose -f docker/docker-compose.yml run --rm hpe_runtime bash
```

### Запуск контейнера OpenPose

```bash
docker compose -f docker/docker-compose.yml --profile openpose run --rm openpose_runtime bash
```

---

## Входные данные

## Видео для одной камеры

Папка:

```text
videos/single_camera/
```

Ожидаемые файлы:

```text
videos/single_camera/
├── front_stay.mp4
├── side_stay.mp4
├── arms.mp4
└── fast_arms.mp4
```

Эти видео используются для:
- `single_camera/media_pipe`
- `single_camera/mmpose_3d`
- `single_camera/videopose3d`

---

## Видео для двух камер

Папка:

```text
videos/double_cameras/
```

Каждая сессия должна иметь вид:

```text
videos/double_cameras/session_01/
├── left.mp4
├── right.mp4
├── calib/
│   ├── intrinsics_left.yaml
│   ├── intrinsics_right.yaml
│   ├── extrinsics.yaml
│   └── stereo.yaml
└── meta.yaml
```

Эти данные используются для:
- `double_cameras/openpose_3d`
- `double_cameras/stereo_triangulation`
- `double_cameras/mmpose_voxelpose`

---

## Как запускать проекты

Ниже перечислены все основные варианты запуска.

---

## 1. Single-camera / MediaPipe

Папка:

```text
single_camera/media_pipe/
```

### Скачать модели

```bash
cd /workspace/single_camera/media_pipe
python scripts/download_models.py
```

### Полный прогон

```bash
cd /workspace/single_camera/media_pipe
python scripts/benchmark.py
```

### Оптимизированный прогон

```bash
cd /workspace/single_camera/media_pipe
python scripts/benchmark_opt.py
```

### Построить графики

```bash
cd /workspace/single_camera/media_pipe
python scripts/analyze_results.py
```

### Запустить playback

```bash
cd /workspace/single_camera/media_pipe
python scripts/playback_3d.py
```

### Что создаётся

В папке:
```text
single_camera/media_pipe/runs/
```

---

## 2. Single-camera / MMPose 3D

Папка:

```text
single_camera/mmpose_3d/
```

### Подготовить models

```bash
cd /workspace/single_camera/mmpose_3d
python scripts/download_models.py
```

### Базовый запуск

```bash
cd /workspace/single_camera/mmpose_3d
python scripts/benchmark.py
```

### Построить графики

```bash
cd /workspace/single_camera/mmpose_3d
python scripts/analyze_results.py
```

### Playback

```bash
cd /workspace/single_camera/mmpose_3d
python scripts/playback_3d.py
```

### Важно

Сейчас папка подготовлена как проектный шаблон. Для полноценной работы нужно дописать реальный inference MMPose 3D.

---

## 3. Single-camera / VideoPose3D

Папка:

```text
single_camera/videopose3d/
```

### Подготовить models

```bash
cd /workspace/single_camera/videopose3d
python scripts/download_models.py
```

### Извлечь 2D keypoints

```bash
cd /workspace/single_camera/videopose3d
python scripts/extract_2d_keypoints.py
```

### Выполнить 3D lifting

```bash
cd /workspace/single_camera/videopose3d
python scripts/run_lifting.py
```

### Сформировать общий прогон и summary

```bash
cd /workspace/single_camera/videopose3d
python scripts/benchmark.py
```

### Построить графики

```bash
cd /workspace/single_camera/videopose3d
python scripts/analyze_results.py
```

### Playback

```bash
cd /workspace/single_camera/videopose3d
python scripts/playback_3d.py
```

### Важно

Сейчас папка подготовлена как проектный шаблон. Для полноценной работы нужно дописать:
- 2D backend;
- запуск VideoPose3D;
- экспорт в общий формат.

---

## 4. Double-cameras / OpenPose 3D

Папка:

```text
double_cameras/openpose_3d/
```

### Проверить структуру stereo-сессий

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/prepare_sessions.py
```

### Подготовить models / runtime

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/download_models.py
```

### Базовый запуск

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/benchmark.py
```

### Построить графики

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/analyze_results.py
```

### Playback

```bash
cd /workspace/double_cameras/openpose_3d
python scripts/playback_3d.py
```

### Важно

Сейчас папка подготовлена как проектный шаблон. Для полноценной работы нужно интегрировать реальный OpenPose 3D reconstruction.

---

## 5. Double-cameras / Stereo Triangulation

Папка:

```text
double_cameras/stereo_triangulation/
```

### Проверить stereo-сессии

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/prepare_sessions.py
```

### Запуск 2D pose для левой камеры

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/run_pose2d_left.py
```

### Запуск 2D pose для правой камеры

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/run_pose2d_right.py
```

### Запуск триангуляции

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/run_triangulation.py
```

### Сформировать итоговый прогон

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/benchmark.py
```

### Построить графики

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/analyze_results.py
```

### Playback

```bash
cd /workspace/double_cameras/stereo_triangulation
python scripts/playback_3d.py
```

### Важно

Это основной универсальный stereo-подход. Здесь предполагается:
- отдельный 2D backend на левую камеру;
- отдельный 2D backend на правую камеру;
- триангуляция по калибровке;
- экспорт в единый формат.

---

## 6. Double-cameras / MMPose VoxelPose

Папка:

```text
double_cameras/mmpose_voxelpose/
```

### Проверить stereo-сессии

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/prepare_sessions.py
```

### Подготовить models

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/download_models.py
```

### Конвертировать входные stereo-сессии в формат MMPose/VoxelPose

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/convert_sessions_to_mmpose_format.py
```

### Базовый запуск

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/benchmark.py
```

### Построить графики

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/analyze_results.py
```

### Playback

```bash
cd /workspace/double_cameras/mmpose_voxelpose
python scripts/playback_3d.py
```

### Важно

Сейчас папка подготовлена как проектный шаблон. Для полноценной работы нужно дописать:
- конвертацию входных stereo-сессий;
- реальный VoxelPose inference;
- экспорт в общий формат.

---

## Common: что можно запускать в общей папке `common`

Ниже перечислены все общие подсистемы.

---

## `common/io_schema/`

### Назначение
Описание общего формата файлов, который должны использовать все алгоритмы.

### Что смотреть
- `schema_description.md`
- `examples/summary_example.csv`

### Когда использовать
Когда нужно:
- проверить правильность структуры выходных данных;
- привести новый алгоритм к общему стандарту;
- понять, какие CSV и JSON должен формировать каждый проект.

---

## `common/metrics/`

### Назначение
Общие функции расчёта метрик качества.

### Файлы
- `common_metrics.py`
- `bone_metrics.py`
- `confidence_adapters.py`

### Что можно использовать

#### `common_metrics.py`
Используется для:
- `effective_fps`
- `detection_rate`
- `jitter`
- средних и percentile

#### `bone_metrics.py`
Используется для:
- вычисления длины руки:
  - `left_shoulder -> left_elbow`
- абсолютной ошибки относительно `arm_len_ref_m`
- относительной ошибки

#### `confidence_adapters.py`
Используется для приведения confidence разных алгоритмов к единым полям:
- `mean_visibility`
- `mean_presence`

### Как запускать
Это не самостоятельные CLI-скрипты, а вспомогательные Python-модули, которые должны импортироваться из алгоритмов.

---

## `common/playback/`

### Назначение
Общие утилиты воспроизведения.

### Файлы
- `play_video_and_3d.py`
- `play_video_and_2d.py`

### Что делает `play_video_and_3d.py`
Позволяет воспроизводить:
- исходное видео;
- 3D-скелет;
- подписи `visibility` и `presence`.

### Когда использовать
Когда у алгоритма уже есть:
- видео;
- `landmarks_3d_world.csv`.

### Как использовать
Либо импортировать как модуль в конкретный проект, либо адаптировать под локальный playback-скрипт алгоритма.

---

## `common/compare/`

### Назначение
Общий сравнительный анализ между всеми алгоритмами.

### Основной файл
- `compare_all_methods.py`

### Что делает
Считывает `summary.csv` из разных алгоритмов и строит сравнительные графики по:
- `effective_fps`
- `mean_presence`
- `jitter_world_m_mean`
- `left_upper_arm_abs_error_m_mean`

### Как запускать

Из папки `human_pose_estimation`:

```bash
cd /workspace
python common/compare/compare_all_methods.py
```

### Когда запускать
После того как несколько алгоритмов уже сформировали свои `runs/run/summary.csv`.

---

## `common/calibration/`

### Назначение
Общие утилиты для работы с калибровкой stereo-камер и триангуляцией.

### Файлы
- `calibration_utils.py`
- `stereo_triangulation.py`

### Что делает `calibration_utils.py`
Позволяет:
- загрузить intrinsics;
- загрузить extrinsics;
- загрузить stereo projection matrices.

### Что делает `stereo_triangulation.py`
Позволяет:
- триангулировать одну точку;
- триангулировать набор keypoints.

### Когда использовать
Во всех двухкамерных проектах:
- `openpose_3d`
- `stereo_triangulation`
- `mmpose_voxelpose`

### Как запускать
Это не отдельные обязательные CLI-скрипты, а общие Python-модули для импорта.

---

## Единый формат результатов

Каждый алгоритм должен сохранять результаты в своей папке `runs/`.

Пример структуры:

```text
runs/run/<video_or_session>/<model>/<mode>/<delegate>/
├── frame_metrics.csv
├── landmarks_2d.csv
├── landmarks_3d_world.csv
├── bones_metrics.csv
└── summary.json
```

Также в корне каждого набора:

```text
runs/run/
├── summary.csv
└── summary.json
```

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

## Калибровка для двух камер

Минимально для каждой stereo-сессии нужны:

- `intrinsics_left.yaml`
- `intrinsics_right.yaml`
- `extrinsics.yaml`
- `stereo.yaml`

Дополнительно рекомендуется:
- `meta.yaml`

---

## Что сейчас реально готово, а что является шаблоном

### Наиболее рабочий baseline
- `single_camera/media_pipe`

### Проектные шаблоны под общий стандарт
- `single_camera/mmpose_3d`
- `single_camera/videopose3d`
- `double_cameras/openpose_3d`
- `double_cameras/stereo_triangulation`
- `double_cameras/mmpose_voxelpose`

Это значит:
- структура папок уже унифицирована;
- форматы файлов уже унифицированы;
- метрики уже унифицированы;
- но реальный inference у части методов ещё нужно дописать.
