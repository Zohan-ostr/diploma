# Запуск верхнего телоуправления Unitree H1 через ноутбук с камерой и unitree_sdk2

Эта инструкция описывает рабочую схему, которую мы получили для реального Unitree H1.

Главная идея: не отправлять команды в приводы через ROS `/lowcmd` с ноутбука. Рабочий канал для H1 — `unitree_sdk2` напрямую в DDS-топик `rt/lowcmd` на компьютере робота. Ноутбук с камерой считает позу и отправляет готовые команды суставов на робота обычным UDP.

Итоговая схема:

```text
НОУТБУК
webcam_mediapipe_node
  -> /pose/landmarks
h1_geometric_retarget_node
  -> /upper_body/command_geom
laptop_upper_body_udp_forwarder.py
  -> UDP 192.168.123.162:50051

РОБОТ H1
sdk2_h1_udp_lowcmd_sender
  -> unitree_sdk2
  -> rt/lowcmd
  -> H1 arm motors
```

Почему так: на роботе `rclcpp` и `unitree_sdk2` нельзя стабильно держать в одном процессе — оба инициализируют CycloneDDS и возникает ошибка `Precondition Not Met - Failed to create domain explicitly`. Поэтому ROS и SDK2 разделены, а между ноутбуком и роботом используется простой UDP.

---

## 0. Ожидаемая сеть

В текущей конфигурации:

```text
робот H1:  192.168.123.162, интерфейс eth0
ноутбук:   192.168.123.200, интерфейс USB/Ethernet enx00e04c36022c
UDP порт:  50051
```

На ноутбуке проверить маршрут к роботу:

```bash
ip route get 192.168.123.162
```

Должно быть примерно:

```text
192.168.123.162 dev enx00e04c36022c src 192.168.123.200
```

---

## 1. Подготовка репозитория на роботе

Терминал робота:

```bash
cd ~/WS_OZ
# Если есть интернет:
git clone https://github.com/Zohan-ostr/diploma.git diploma
```

Если на роботе нет DNS/интернета, скопировать репозиторий с ноутбука через `scp`, флешку или архив.

---

## 2. Обязательно: склонировать и собрать unitree_sdk2 на роботе

Терминал робота:

```bash
cd ~/WS_OZ
bash ~/WS_OZ/diploma/scripts/robot_h1/robot_install_unitree_sdk2.sh
```

Этот скрипт делает:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
sudo apt install ...
cmake ..
make
sudo make install
```

Если у робота нет интернета, на ноутбуке:

```bash
cd ~/Downloads
git clone https://github.com/unitreerobotics/unitree_sdk2.git
tar -czf unitree_sdk2.tar.gz unitree_sdk2
scp unitree_sdk2.tar.gz unitree@192.168.123.162:~/WS_OZ/
```

На роботе:

```bash
cd ~/WS_OZ
tar -xzf unitree_sdk2.tar.gz
cd ~/WS_OZ/diploma
bash scripts/robot_h1/robot_install_unitree_sdk2.sh
```

После установки должны существовать:

```bash
ls /usr/local/include/unitree
ls /usr/local/lib | grep unitree
```

Ожидаемо:

```text
/usr/local/include/unitree/...
/usr/local/lib/libunitree_sdk2.a
```

---

## 3. Сборка robot-side bridge на роботе

Терминал робота:

```bash
cd ~/WS_OZ/diploma
bash scripts/robot_h1/sdk2_upper_body_bridge/build_on_robot.sh
```

Скрипт собирает два executable:

```text
sdk2_h1_udp_lowcmd_sender      # основной процесс: UDP -> unitree_sdk2 -> rt/lowcmd
ros_upper_body_udp_forwarder   # optional/local test: ROS topic -> UDP localhost
```

Для боевого запуска с ноутбуком нужен только:

```text
sdk2_h1_udp_lowcmd_sender
```

---

## 4. Проверочный тест правого плеча через SDK2

Этот тест нужен один раз, чтобы убедиться, что SDK2 действительно двигает руку.

Терминал робота:

```bash
cd ~/WS_OZ/diploma/scripts/robot_h1/sdk2_right_arm_test
bash build.sh
bash run_right_shoulder_test.sh
```

Скрипт попросит:

```text
Type YES to start:
```

Ввести:

```text
YES
```

По умолчанию тест двигает мотор `12`, правое плечо pitch, амплитуда `0.10 rad`.

Для локтя можно запустить так:

```bash
MOTOR_ID=15 AMPLITUDE=0.10 KP=20.0 KD=1.0 bash run_right_shoulder_test.sh
```

Если плечо шевелится — канал `unitree_sdk2 -> rt/lowcmd` работает.

---

## 5. Боевой запуск. Терминал 1 на роботе: SDK2 UDP sender

На роботе открыть терминал 1:

```bash
cd ~/WS_OZ/diploma
bash scripts/robot_h1/sdk2_upper_body_bridge/run_sdk2_udp_sender_on_robot.sh
```

Скрипт попросит подтверждение:

```text
WARNING: this sends commands to rt/lowcmd. Type YES to start:
```

Ввести:

```text
YES
```

Ожидаемый старт:

```text
rt/lowstate OK
Initial arm q: [...]
timeout=1 seq=0 ...
```

`timeout=1` на этом этапе нормально: ноутбук ещё не отправляет команды.

Параметры можно менять переменными окружения:

```bash
IFACE=eth0 \
KP=25.0 \
KD=1.5 \
MAX_STEP=0.012 \
TIMEOUT_SEC=0.35 \
UDP_PORT=50051 \
bash scripts/robot_h1/sdk2_upper_body_bridge/run_sdk2_udp_sender_on_robot.sh
```

Если движения слишком медленные, можно поднять:

```bash
MAX_STEP=0.02
```

Если движения резкие — уменьшить:

```bash
MAX_STEP=0.006
```

---

## 6. Терминал 1 на ноутбуке: запустить камеру

На ноутбуке, на хосте, НЕ внутри контейнера:

```bash
cd ~/diploma/workspace/upper_body_teleop_home_full
xhost +local:docker

ROS_DOMAIN_ID_VALUE=0 \
ROS_LOCALHOST_ONLY_VALUE=0 \
bash scripts/home_run_camera.sh 0
```

Важно: `home_run_camera.sh` — это host-скрипт, он сам вызывает Docker. Если запустить его внутри контейнера, будет ошибка:

```text
docker: command not found
```

Внутри контейнера камеры должны появиться топики:

```text
/pose/landmarks
/upper_body/command
/upper_body/command_geom
```

Проверка из отдельного терминала ноутбука:

```bash
docker exec -it h1_camera_pipeline bash
```

Внутри контейнера:

```bash
cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 topic list | grep -E 'pose|upper_body'
ros2 topic echo /upper_body/command_geom --once
```

Если `echo --once` зависает, значит retarget не публикует команды. Тогда отдельно проверить:

```bash
ros2 topic hz /pose/landmarks
ros2 topic info /upper_body/command_geom -v
ps aux | grep -E 'media|pose|retarget|geom|h1|webcam|camera' | grep -v grep
```

---

## 7. Терминал 2 на ноутбуке: UDP forwarder в контейнере камеры

На ноутбуке, в новом терминале хоста:

```bash
docker exec -it h1_camera_pipeline bash
```

Внутри контейнера:

```bash
cd /workspace
bash scripts/robot_h1/run_laptop_udp_forwarder_in_container.sh
```

Ожидаемые логи:

```text
Laptop UDP forwarder started
input_topic: /upper_body/command_geom
udp target:  192.168.123.162:50051
seq=0 mapped=8 q_r=[...] q_l=[...]
seq=15 mapped=8 ...
seq=30 mapped=8 ...
```

Если `seq` растёт, ноутбук отправляет UDP на робота.

На роботе в терминале `sdk2_h1_udp_lowcmd_sender` должно смениться:

```text
timeout=0 seq=...
```

Если на роботе всё ещё:

```text
timeout=1 seq=1
```

значит поток UDP-команд не приходит. Обычно причины:

1. `laptop_upper_body_udp_forwarder.py` не запущен в контейнере `h1_camera_pipeline`.
2. `/upper_body/command_geom` есть в списке, но фактически не публикует сообщения.
3. IP робота не `192.168.123.162`.
4. Firewall режет UDP 50051.

---

## 8. Быстрый UDP/command тест с ноутбука без камеры

Если нужно проверить только UDP-цепочку, запустить на роботе терминал 1 из пункта 5, а на ноутбуке в контейнере камеры:

```bash
cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# В одном терминале контейнера запустить forwarder:
bash scripts/robot_h1/run_laptop_udp_forwarder_in_container.sh
```

В другом терминале контейнера:

```bash
cd /workspace
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

ros2 topic pub --once /upper_body/command_geom upper_body_msgs/msg/UpperBodyCommand "{
  header: {frame_id: 'manual_laptop_udp'},
  joint_names: ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow'],
  position: [0.20, 0.0, 1.30, 1.57, 0.0, 0.0, -1.30, 1.57],
  confidence: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  valid: true
}"
```

На роботе должно появиться:

```text
timeout=0 seq=...
```

---

## 9. Диагностика

### 9.1. На роботе проверить входящие UDP-пакеты

```bash
sudo tcpdump -ni eth0 udp port 50051
```

Если пакеты видны, но `sdk2_h1_udp_lowcmd_sender` не меняет `seq`, значит sender слушает не тот порт или не пересобран после патча `INADDR_ANY`.

### 9.2. На ноутбуке проверить маршрут к роботу

```bash
ip route get 192.168.123.162
```

Ожидаемо:

```text
192.168.123.162 dev enx00e04c36022c src 192.168.123.200
```

### 9.3. Проверить, что ноутбук реально публикует команды

В контейнере `h1_camera_pipeline`:

```bash
ros2 topic echo /upper_body/command_geom --once
ros2 topic hz /upper_body/command_geom
```

### 9.4. Ошибка `ddsi_udp_conn_write ... failed`

Сообщения вида:

```text
ddsi_udp_conn_write to udp/172.18.0.1:7414 failed
```

относятся к CycloneDDS/ROS discovery по лишним интерфейсам. В текущей UDP-схеме между ноутбуком и роботом они не являются главным признаком ошибки. Главный признак работы — на роботе в `sdk2_h1_udp_lowcmd_sender` должно быть:

```text
timeout=0 seq=растёт
```

---

## 10. Что запускать последовательно

### Робот, терминал 1

```bash
cd ~/WS_OZ/diploma
bash scripts/robot_h1/sdk2_upper_body_bridge/run_sdk2_udp_sender_on_robot.sh
# ввести YES
```

### Ноутбук, терминал 1, хост

```bash
cd ~/diploma/workspace/upper_body_teleop_home_full
xhost +local:docker
ROS_DOMAIN_ID_VALUE=0 ROS_LOCALHOST_ONLY_VALUE=0 bash scripts/home_run_camera.sh 0
```

### Ноутбук, терминал 2, хост -> контейнер

```bash
docker exec -it h1_camera_pipeline bash
```

Внутри контейнера:

```bash
cd /workspace
bash scripts/robot_h1/run_laptop_udp_forwarder_in_container.sh
```

После этого на роботе должно быть:

```text
timeout=0 seq=...
```

и руки должны реагировать на движения перед камерой.
