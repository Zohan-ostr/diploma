# Unitree H1 Teleoperation Workspace

Проект для двух режимов разработки:

- **Домашний режим** — отладка кинематики и потока команд через **RViz**.
- **Лабораторный режим** — запуск упрощённой модели в **Gazebo Classic**.

Цель проекта:

1. Получить позу оператора из MediaPipe.
2. Преобразовать landmarks в углы суставов H1.
3. Публиковать команды верхней части тела через ROS 2.
4. Проверять движение сначала в RViz, затем в Gazebo.
5. Позже переключить тот же пайплайн на реального Unitree H1.

---

## 1. Структура репозитория

```text
h1_teleop_project/
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── scripts/
│   ├── check_host.sh
│   └── kill_gazebo.sh
├── src/
│   ├── h1_description_ros2/
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── launch/
│   │   │   ├── display.launch.py
│   │   │   └── gazebo_light.launch.py
│   │   └── urdf/
│   │       ├── h1.urdf
│   │       └── h1_gazebo_light.urdf
│   └── h1_teleop_test/
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       └── h1_teleop_test/
│           ├── __init__.py
│           └── upper_body_cmd_pub.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## 2. Что внутри

### `h1_description_ros2`

ROS 2 пакет с двумя вариантами модели:

- `h1.urdf` — модель для **RViz**.
- `h1_gazebo_light.urdf` — облегчённая модель для **Gazebo**, чтобы не ловить зависания на слабом железе.

### `h1_teleop_test`

Тестовый publisher, который публикует `sensor_msgs/JointState` в `/joint_states` и двигает:

- `torso_joint`
- `left_shoulder_pitch_joint`
- `left_shoulder_roll_joint`
- `left_shoulder_yaw_joint`
- `left_elbow_joint`
- `right_shoulder_pitch_joint`
- `right_shoulder_roll_joint`
- `right_shoulder_yaw_joint`
- `right_elbow_joint`

### `docker/`

Контейнер с ROS 2 Humble, Gazebo Classic, RViz, colcon и базовыми утилитами для сборки.

### `scripts/check_host.sh`

Проверка хоста перед запуском контейнера:

- установлен ли Docker
- работает ли Docker daemon
- есть ли `docker compose`
- есть ли GPU и `nvidia-smi`
- доступен ли X11/Wayland
- установлен ли `xhost`
- установлен ли ROS 2 / Gazebo на хосте (необязательно, просто для информации)

---

## 3. Два режима работы

### Дом

Используй **только RViz**:

- меньше нагрузка
- не нужна мощная видеокарта
- удобно отлаживать цепочку `команды -> суставы`
- можно запускать даже при софтварном OpenGL

### Лаборатория

Используй **Gazebo Classic**:

- нужен ноутбук или ПК помощнее
- удобно показывать симуляцию
- можно отлаживать более полный pipeline

---

## 4. Проверка хоста перед запуском

На хостовой машине в корне проекта:

```bash
chmod +x scripts/check_host.sh
./scripts/check_host.sh
```

Минимально желательно увидеть:

- Docker установлен
- `docker compose` работает
- Docker daemon активен
- пользователь в группе `docker` или ты готов запускать команды через `sudo`
- `DISPLAY` не пустой
- `xhost` установлен

Если `nvidia-smi` не найден, это **не ошибка**. Просто дома лучше использовать режим RViz.

---

## 5. Если Docker не установлен

Установи Docker Engine и Compose plugin по официальной инструкции Docker для Ubuntu.

После установки обычно нужны ещё шаги:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

И проверка:

```bash
docker --version
docker compose version
```

---

## 6. Разрешение GUI из контейнера

Перед первым запуском на хосте:

```bash
xhost +local:docker
```

После завершения работы можно вернуть ограничения:

```bash
xhost -local:docker
```

> На Ubuntu с **Wayland** GUI из контейнера иногда работает нестабильно. Для домашнего режима это обычно лечится переменной `LIBGL_ALWAYS_SOFTWARE=1`.

---

## 7. Сборка контейнера

В корне проекта:

```bash
docker compose build
```

---

## 8. Базовая сборка workspace в контейнере

```bash
docker compose run --rm h1-dev bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  colcon build &&
  source install/setup.bash
"
```

> Если `colcon` ругается на `h1_teleop_test` про marker/package.xml, это пока предупреждение, а не блокирующая ошибка.

---

## 9. Домашний режим: RViz

### Терминал 1 на хосте

```bash
xhost +local:docker
```

### Терминал 2 на хосте — запуск RViz

```bash
docker compose run --rm h1-dev bash -lc "
  export LIBGL_ALWAYS_SOFTWARE=1 &&
  export QT_X11_NO_MITSHM=1 &&
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  colcon build &&
  source install/setup.bash &&
  ros2 launch h1_description_ros2 display.launch.py
"
```

### Терминал 3 на хосте — запуск тестового publisher

```bash
docker compose run --rm h1-dev bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  source install/setup.bash &&
  ros2 run h1_teleop_test upper_body_cmd_pub
"
```

Ожидаемый результат:

- открывается RViz
- видна модель H1
- плечи, локти и корпус двигаются

---

## 10. Лабораторный режим: Gazebo

Перед запуском желательно закрыть старые процессы Gazebo:

```bash
chmod +x scripts/kill_gazebo.sh
./scripts/kill_gazebo.sh
```

### Терминал 1 на хосте

```bash
xhost +local:docker
```

### Терминал 2 на хосте — запуск Gazebo

```bash
docker compose run --rm h1-dev bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  colcon build &&
  source install/setup.bash &&
  ros2 launch h1_description_ros2 gazebo_light.launch.py
"
```

Ожидаемый результат:

- открывается Gazebo Classic
- в левой панели `Models` появляется `unitree_h1_light`
- Gazebo не зависает

> В текущей реализации Gazebo-модель облегчённая. Это сделано специально, чтобы на слабом железе не ловить зависания из-за тяжёлых visual meshes.

---

## 11. Что делать, если контейнерный RViz не запускается

### Проблема 1: `parent link [world] of joint [floating_base_joint] not found`

Причина: в `h1.urdf` есть `floating_base_joint`, но не определён link `world`.

Исправление:

```bash
python3 - <<'PY'
from pathlib import Path

p = Path('src/h1_description_ros2/urdf/h1.urdf')
text = p.read_text(encoding='utf-8')

if '<link name="world"/>' not in text and '<link name="world" />' not in text:
    start = text.find('<robot')
    start = text.find('>', start)
    text = text[:start+1] + '\n  <link name="world"/>\n' + text[start+1:]

p.write_text(text, encoding='utf-8')
print('patched', p)
PY
```

Потом пересобери workspace:

```bash
docker compose run --rm h1-dev bash -lc "
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  colcon build
"
```

### Проблема 2: `MESA`, `glx`, `iris`, `Failed to query drm device`

Это частая история для контейнера под Wayland/встроенную графику.

Попробуй запускать RViz так:

```bash
docker compose run --rm h1-dev bash -lc "
  export LIBGL_ALWAYS_SOFTWARE=1 &&
  export QT_X11_NO_MITSHM=1 &&
  source /opt/ros/humble/setup.bash &&
  cd /workspaces/h1_teleop_ws &&
  source install/setup.bash &&
  ros2 launch h1_description_ros2 display.launch.py
"
```

### Проблема 3: Gazebo пишет `Address already in use`

Остался старый `gzserver`.

Исправление:

```bash
./scripts/kill_gazebo.sh
```

Если не помогло:

```bash
pkill -9 -f gzserver || true
pkill -9 -f gzclient || true
pkill -9 -f "gazebo --verbose" || true
```

---

## 12. Запуск на мощной машине с GPU

Если на лабораторной машине установлен NVIDIA Container Toolkit, можно использовать GPU passthrough.

Пример отдельного запуска:

```bash
docker run --rm -it \
  --network host \
  --ipc host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspaces/h1_teleop_ws \
  h1_teleop_project-h1-dev bash
```

Если GPU passthrough не настроен, всё равно можно работать через CPU.

---

## 13. Следующие пакеты, которые стоит добавить

### `h1_pose_mapper`

Нода, которая будет:

- принимать landmarks из MediaPipe
- оценивать углы плеч, локтей и корпуса
- публиковать `JointState` или отдельное сообщение для H1

### `h1_filters`

Нода для:

- сглаживания углов
- ограничения скоростей изменения
- стабилизации дрожащих суставов

### `h1_robot_bridge`

Отдельный слой для переключения между:

- RViz-режимом
- Gazebo-режимом
- реальным роботом через Unitree ROS 2 API

Архитектура проекта в итоге:

```text
MediaPipe -> Pose Mapper -> Filters/Safety -> Robot Bridge -> RViz / Gazebo / Real H1
```

---

## 14. Рекомендуемый git workflow

```bash
git init
git add .
git commit -m "Initial H1 teleop workspace"
git remote add origin <your_repo_url>
git push -u origin main
```

---

## 15. Типовой цикл работы

### Дома

1. `./scripts/check_host.sh`
2. `xhost +local:docker`
3. `docker compose build`
4. контейнерный `colcon build`
5. запуск `display.launch.py`
6. запуск `upper_body_cmd_pub`

### В лаборатории

1. `./scripts/check_host.sh`
2. `xhost +local:docker`
3. `docker compose build`
4. контейнерный `colcon build`
5. `./scripts/kill_gazebo.sh`
6. запуск `gazebo_light.launch.py`

---

## 16. Важные замечания

- Дома основной режим — **RViz**.
- Gazebo дома можно запускать, но он не является обязательным.
- В лаборатории основной режим — **Gazebo**.
- Если тяжёлые модели снова начнут вешать Gazebo, лучше оставить облегчённую модель и отлаживать управление, а не графику.
- Для ВКР важнее устойчивый pipeline `MediaPipe -> углы -> ROS 2 -> H1`, чем тяжёлый фотореалистичный рендер.
