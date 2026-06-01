#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SIM 4. VECTOR FABRIK RETARGET
# ============================================================
#
# Назначение:
#   запустить главный алгоритм копирующего управления верхней частью H1.
#
# Вход:
#   /pose/landmarks
#
# Выход:
#   /upper_body/command_geom
#
# Основной файл алгоритма:
#   /workspace/scripts/robot_h1/vector_fabrik_retarget.py
#
# Учебные блоки внутри vector_fabrik_retarget.py:
#   1. Выделение базиса X/Y/Z тела оператора.
#   2. Формирование векторов:
#        shoulder -> elbow
#        elbow    -> wrist
#   3. Pitch/Roll: FABRIK-подобный сегментный подбор локтя.
#   4. Elbow: угол между двумя сегментами руки.
#   5. Yaw: сегментный подбор для позиционирования кисти.
#
# Что студентам можно менять в симуляции:
#   - landmark_alpha;
#   - joint_alpha;
#   - pitch_grid / roll_grid / yaw_grid;
#   - pitch_window / roll_window;
#   - веса ошибок;
#   - elbow_gain / deadzone / response_gain;
#   - max_joint_step / yaw_max_step / elbow_max_step.
#
# Что нельзя менять без понимания:
#   - input_topic;
#   - output_topic;
#   - порядок суставов в UpperBodyCommand;
#   - знаки left/right yaw;
#   - стандартный базис тела.
#
# Для реального робота:
#   этот файл только считает углы.
#   PD-регуляторы находятся в SDK2 sender и не меняются.
# ============================================================

CONTAINER_NAME="${CONTAINER_NAME:-h1_camera_pipeline}"

echo "============================================================"
echo " SIM 3: START VECTOR FABRIK RETARGET"
echo "============================================================"
echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "INPUT:          /pose/landmarks"
echo "OUTPUT:         /upper_body/command_geom"
echo "ROS_DOMAIN_ID:  42"
echo "============================================================"

for i in $(seq 1 60); do
  if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "ERROR: container is not running: $CONTAINER_NAME"
    exit 1
  fi

  sleep 1
done

docker exec -it "$CONTAINER_NAME" bash -lc '
set -e

cd /workspace

export PYTHONPATH=/workspace/src:/workspace/src/home_pipeline:$PYTHONPATH

export PYTHONPATH=/workspace/src/upper_body_teleop_runtime:$PYTHONPATH

source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

echo "Stopping old retarget processes safely..."

python3 - <<PY
import os
import signal
import subprocess

me = os.getpid()
out = subprocess.check_output(["ps", "-eo", "pid=,comm=,args="], text=True)

targets = [
    "sim_geometric_retarget_tpose_yaw.py",
    "sim_geometric_retarget_old_with_tpose_yaw.py",
    "sim_geometric_retarget_ik2.py",
    "realsense_ik_upper_body_retarget.py",
    "vector_fabrik_retarget.py",
]

for line in out.splitlines():
    parts = line.strip().split(None, 2)
    if len(parts) < 3:
        continue

    pid_s, comm, args = parts
    pid = int(pid_s)

    if pid == me:
        continue

    if comm.startswith("python") and any(t in args for t in targets):
        print(f"killing pid={pid} args={args}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
PY

sleep 1

# ============================================================
# Параметры ниже разделены на учебные группы.
# Логику менять не нужно: это тот же vector_fabrik_retarget.py.
# ============================================================

python3 -m upper_body_teleop_runtime.vector_fabrik_retarget \
  --ros-args \
  \
  -p input_topic:=/pose/landmarks \
  -p output_topic:=/upper_body/command_geom \
  -p calibration_topic:=/upper_body/start_calibration \
  \
  `# ---------- Калибровка ложного сгиба локтя ----------` \
  -p calibration_frames:=45 \
  -p use_calibrated_elbow_bias:=true \
  \
  `# ---------- Сглаживание входных точек и суставов ----------` \
  -p landmark_alpha:=0.35 \
  -p joint_alpha:=0.75 \
  \
  `# ---------- Геометрические длины руки робота ----------` \
  -p robot_upper_len:=0.31 \
  -p robot_fore_len:=0.31 \
  \
  `# ---------- FABRIK/сегментный подбор pitch и roll ----------` \
  -p pitch_grid:=21 \
  -p roll_grid:=31 \
  -p pitch_window:=1.30 \
  -p roll_window:=1.60 \
  -p upper_direction_weight:=1.0 \
  -p upper_continuity_weight:=0.015 \
  -p pitch_geom_gain:=1.0 \
  \
  `# ---------- Сегментный подбор yaw для кисти ----------` \
  -p yaw_grid:=81 \
  -p yaw_direction_weight:=1.0 \
  -p yaw_wrist_position_weight:=4.0 \
  -p yaw_refine_window:=0.18 \
  -p yaw_refine_grid:=17 \
  -p yaw_continuity_weight:=0.05 \
  \
  `# ---------- Диапазоны yaw для левой и правой руки ----------` \
  -p left_yaw_down:=-1.30 \
  -p left_yaw_up:=1.74 \
  -p right_yaw_down:=1.30 \
  -p right_yaw_up:=-1.74 \
  \
  `# ---------- Elbow по углу между shoulder->elbow и elbow->wrist ----------` \
  -p left_elbow_straight:=1.57 \
  -p right_elbow_straight:=1.57 \
  -p elbow_gain:=1.55 \
  -p elbow_bend_deadzone:=0.015 \
  -p elbow_bend_response_gain:=1.75 \
  \
  `# ---------- Ограничение резкости команд ----------` \
  -p max_joint_step:=0.180 \
  -p yaw_max_step:=0.450 \
  -p elbow_max_step:=0.320 \
  \
  `# ---------- Направление оси Z телесного базиса ----------` \
  -p standard_z_sign:=1.0
'
