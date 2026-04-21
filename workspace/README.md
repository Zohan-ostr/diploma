# Unitree H1 Teleoperation Workspace

Проект для двух режимов разработки:

- **Домашний режим**: отладка кинематики и потока команд через **RViz**.
- **Лабораторный режим**: запуск упрощённой видимой модели в **Gazebo Classic**.

Основная идея проекта:

1. Получить позу оператора из MediaPipe.
2. Преобразовать landmarks в углы суставов H1.
3. Публиковать команды верхней части тела через ROS 2.
4. Проверять движение сначала в RViz, затем в Gazebo.

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

- `h1.urdf` — упрощённая видимая модель для **RViz**.
- `h1_gazebo_light.urdf` — облегчённая видимая модель для **Gazebo**.

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

Контейнер с ROS 2 Humble, Gazebo Classic, RViz, colcon и утилитами для сборки.

### `scripts/check_host.sh`

Проверка хоста перед запуском контейнера:

- установлен ли Docker
- работает ли Docker daemon
- есть ли `docker compose`
- есть ли `nvidia-smi`
- доступен ли X11/Wayland
- установлен ли `xhost`

---

## 3. Быстрый сценарий работы

### Дом

Используй только RViz:

- меньше нагрузка
- не нужна мощная видеокарта
- удобно отлаживать цепочку `команды -> суставы`

### Лаборатория

Используй Gazebo:

- нужен ноутбук/ПК помощнее
- удобно показывать "симуляцию робота"
- можно демонстрировать видимую модель в мире Gazebo

---

## 4. Проверка хоста перед запуском

На хостовой машине в корне проекта:

```bash
chmod +x scripts/check_host.sh
./scripts/check_host.sh
```

Что желательно увидеть:

- `docker: FOUND`
- `docker compose: FOUND`
- Docker daemon активен
- пользователь в группе `docker` или ты готов запускать через `sudo`
- `DISPLAY` не пустой
- `xhost` установлен

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

## 6. Разрешение X11 для GUI из контейнера

На хосте перед первым запуском:

```bash
xhost +local:docker
```

После работы можно закрыть доступ:

```bash
xhost -local:docker
```

---

## 7. Сборка контейнера

В корне проекта:

```bash
docker compose build
```

---

## 8. Запуск контейнера

```bash
docker compose run --rm h1-dev
```

Внутри контейнера:

```bash
source /opt/ros/humble/setup.bash
cd /workspaces/h1_teleop_ws
colcon build
source install/setup.bash
```

---

## 9. Домашний режим: RViz

### Терминал 1 внутри контейнера

```bash
source /opt/ros/humble/setup.bash
cd /workspaces/h1_teleop_ws
colcon build
source install/setup.bash
ros2 launch h1_description_ros2 display.launch.py
```

### Терминал 2 внутри контейнера

```bash
source /opt/ros/humble/setup.bash
cd /workspaces/h1_teleop_ws
source install/setup.bash
ros2 run h1_teleop_test upper_body_cmd_pub
```

Ожидаемый результат:

- открывается RViz
- видна упрощённая модель H1
- плечи, локти и корпус двигаются

---

## 10. Лабораторный режим: Gazebo

Перед запуском желательно закрыть старые процессы Gazebo:

```bash
./scripts/kill_gazebo.sh
```

### Терминал 1 внутри контейнера

```bash
source /opt/ros/humble/setup.bash
cd /workspaces/h1_teleop_ws
colcon build
source install/setup.bash
ros2 launch h1_description_ros2 gazebo_light.launch.py
```

Ожидаемый результат:

- открывается Gazebo Classic
- в левой панели `Models` появляется `unitree_h1_light`
- Gazebo не зависает

> Примечание: сейчас Gazebo-версия — упрощённая и нужна для устойчивой отладки на слабом или среднем железе.

---

## 11. Как запускать на мощной машине с GPU

Если на лабораторной машине установлен NVIDIA Container Toolkit, можно запускать контейнер так:

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

## 12. Что добавить следующим шагом

Следующие логические блоки проекта:

### `h1_pose_mapper`

Нода, которая будет:

- принимать landmarks из MediaPipe
- оценивать углы плеч, локтей и корпуса
- публиковать `JointState` или свой тип сообщения для H1

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

Это даст красивую архитектуру для ВКР:

```text
MediaPipe -> Pose Mapper -> Filters/Safety -> Robot Bridge -> RViz / Gazebo / Real H1
```

---

## 13. Рекомендуемый git workflow

```bash
git init
git add .
git commit -m "Initial H1 teleop workspace"
git remote add origin <your_repo_url>
git push -u origin main
```

---

## 14. Типовой цикл работы

### Дома

1. `./scripts/check_host.sh`
2. `xhost +local:docker`
3. `docker compose build`
4. `docker compose run --rm h1-dev`
5. `colcon build`
6. запуск `display.launch.py`
7. запуск `upper_body_cmd_pub`

### В лаборатории

1. `./scripts/check_host.sh`
2. `xhost +local:docker`
3. `docker compose build`
4. `docker compose run --rm h1-dev`
5. `colcon build`
6. `./scripts/kill_gazebo.sh`
7. запуск `gazebo_light.launch.py`

---

## 15. Важные замечания

- Gazebo может зависать на тяжёлых mesh-моделях без хорошей графики.
- Поэтому для отладки лучше держать **облегчённую Gazebo-модель**.
- Для домашнего режима RViz достаточно и обычно устойчивее.
- Для демонстрации в лаборатории удобнее Gazebo.

