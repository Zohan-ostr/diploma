# Upper Body Teleoperation — Home Mode

Проект для первого домашнего теста контура:

```text
webcam/mock pose -> MediaPipe/landmarks -> retargeting -> /joint_states + /tf -> RViz
```

В Home-режиме **симулятор не запускается**. Загружается полный робот, но команды публикуются только для верхней части тела:

- `torso_joint`
- `left_shoulder_pitch_joint`
- `left_shoulder_roll_joint`
- `left_shoulder_yaw_joint`
- `left_elbow_joint`
- `right_shoulder_pitch_joint`
- `right_shoulder_roll_joint`
- `right_shoulder_yaw_joint`
- `right_elbow_joint`

Ноги в Home-режиме не управляются этим контуром. В будущей лабораторной версии нижняя часть тела должна оставаться под штатным locomotion/balance-контуром.

---

## 1. Проверка хоста

```bash
bash scripts/check_home_host.sh
```

Нужно, чтобы были:

- Docker
- Docker Compose
- `DISPLAY`
- доступ к X11 через `xhost`
- вебкамера `/dev/video0` для camera-режима

---

## 2. Сборка контейнера

```bash
bash scripts/home_build.sh
```

Контейнер основан на ROS2 Humble и содержит:

- RViz2
- robot_state_publisher
- OpenCV
- MediaPipe
- ROS2 workspace tools

---

## 3. Первый тест без камеры: mock mode

```bash
bash scripts/home_run_mock.sh
```

Что должно произойти:

- откроется RViz;
- загрузится полный робот;
- `mock_pose_source` начнёт генерировать тестовую позу;
- `retarget_node` будет преобразовывать её в команды верхней части тела;
- верхняя часть робота будет двигаться.

Этот режим нужен только для проверки, что ROS2, Docker, RViz, TF и retargeting работают.

---

## 4. Тест с вебкамерой и MediaPipe

```bash
bash scripts/home_run_camera.sh 0
```

Где `0` — номер камеры `/dev/video0`.

Если камера другая:

```bash
bash scripts/home_run_camera.sh 2
```

Что должно открыться:

1. RViz с полным роботом.
2. Отдельное окно `Home webcam + MediaPipe pose`, где показывается изображение с камеры и скелет MediaPipe.

---

## 5. Как управлять видом в RViz

В RViz:

- левая кнопка мыши — вращение;
- колёсико — приблизить/отдалить;
- средняя кнопка мыши — сдвиг;
- меню `Panels -> Views` — открыть панель вида;
- `F` — фокус на выбранный объект.

В конфиге уже включены:

- `RobotModel`
- `TF`
- `Grid`
- Fixed Frame = `world`

---

## 6. Основные топики

```bash
/pose/landmarks
/upper_body/command
/retarget/joint_states_debug
/joint_states
/tf
/tf_static
```

Проверить:

```bash
ros2 topic list
ros2 topic echo /upper_body/command --once
ros2 topic echo /joint_states --once
```

---

## 7. Запись эксперимента

Внутри контейнера или на хосте с ROS2:

```bash
ros2 bag record /pose/landmarks /upper_body/command /joint_states /tf /tf_static
```

`rosbag2` используется только для записи и воспроизведения экспериментов. Онлайн-обмен идёт через ROS2 topics.

---

## 8. Что является заглушкой

Пакет `g1_mujoco_backend` пока является заглушкой. Он нужен для будущего лабораторного режима:

```text
Jetson + Ubuntu 24 + Unitree G1 + MuJoCo simulation
```

Home-режим его не использует.

---

## 9. Частые проблемы

### RViz открылся, но робот не двигается

Проверь:

```bash
ros2 topic echo /joint_states --once
ros2 topic echo /pose/landmarks --once
```

### Нет окна камеры

Ты, вероятно, запустил mock-режим:

```bash
bash scripts/home_run_mock.sh
```

Для камеры нужно:

```bash
bash scripts/home_run_camera.sh 0
```

### Камера не найдена

Проверь:

```bash
ls -la /dev/video*
```

### Permission denied на скриптах

Можно запускать через `bash`, тогда `chmod` не нужен:

```bash
bash scripts/home_run_mock.sh
```

Или можно сделать файл исполняемым:

```bash
chmod +x scripts/home_run_mock.sh
./scripts/home_run_mock.sh
```

`chmod +x` только разрешает запуск файла как программы. Сам файл запускается отдельной командой.
