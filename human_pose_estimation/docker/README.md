# Docker profiles for human_pose_estimation

Здесь лежат два отдельных профиля Docker:

- `Dockerfile.cpu` и `docker-compose.cpu.yml` — для домашнего ноутбука без NVIDIA GPU
- `Dockerfile.gpu` и `docker-compose.gpu.yml` — для лабораторного компьютера с NVIDIA GPU
- `Dockerfile.openpose` — отдельный GPU-контейнер под OpenPose

## 1. Работа на домашнем ноутбуке

Из папки `human_pose_estimation`:

```bash
docker compose -f docker/docker-compose.cpu.yml build
docker compose -f docker/docker-compose.cpu.yml run --rm hpe_runtime_cpu bash
```

После входа в контейнер:
```bash
cd /workspace/single_camera/media_pipe
python scripts/benchmark.py
```

Этот контейнер:
- не тянет CUDA
- не ставит PyTorch GPU stack
- подходит для `media_pipe`, анализа результатов, playback и общей отладки

## 2. Работа на лабораторном компьютере

Из папки `human_pose_estimation`:

```bash
docker compose -f docker/docker-compose.gpu.yml build
docker compose -f docker/docker-compose.gpu.yml run --rm hpe_runtime_gpu bash
```

После входа в контейнер:
```bash
cd /workspace/single_camera/mmpose_3d
python scripts/benchmark.py
```

Этот контейнер:
- рассчитан на NVIDIA GPU
- ставит PyTorch + OpenMMLab stack
- нужен для тяжёлых алгоритмов:
  - `mmpose_3d`
  - `videopose3d`
  - `mmpose_voxelpose`

## 3. OpenPose

Если нужен OpenPose:

```bash
docker compose -f docker/docker-compose.gpu.yml --profile openpose build
docker compose -f docker/docker-compose.gpu.yml --profile openpose run --rm openpose_runtime bash
```

## 4. Что менять в проекте

В папке `docker/` лучше оставить только эти файлы и удалить старый единый `Dockerfile` и старый `docker-compose.yml`, чтобы не путаться.