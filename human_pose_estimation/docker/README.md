# docker

Заготовка папки `human_pose_estimation/docker` для запуска алгоритмов через контейнеры.

## Файлы
- `Dockerfile` — базовый контейнер для Python / MediaPipe / ONNX
- `Dockerfile.openpose` — отдельный контейнер под OpenPose
- `docker-compose.yml` — запуск контейнеров
- `entrypoint.sh` — общий entrypoint
- `requirements-base.txt` — общие Python-зависимости
- `requirements-hpe.txt` — зависимости для HPE-базовых методов
- `requirements-mmpose.txt` — место под OpenMMLab stack

## Быстрый старт

Из корня проекта `diploma/`:

```bash
docker compose -f human_pose_estimation/docker/docker-compose.yml build
docker compose -f human_pose_estimation/docker/docker-compose.yml run --rm hpe_runtime bash
```

Для OpenPose:

```bash
docker compose -f human_pose_estimation/docker/docker-compose.yml --profile openpose build
docker compose -f human_pose_estimation/docker/docker-compose.yml --profile openpose run --rm openpose_runtime bash
```

## Важно
`requirements-mmpose.txt` оставлен как заготовка. Для MMPose обычно удобнее ставить пакетный стек через `openmim` и закреплять версии отдельно под CUDA / torch.
